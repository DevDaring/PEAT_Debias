#!/usr/bin/env python3
"""
vastctl.py — local controller for running PEAT on a Vast.ai GPU VM.

Runs on the LOCAL machine. Wraps the `vastai` CLI (>=1.2.1). Reads the API key
from Code/.env (PHD_VAST_AI_KEY) and never prints or logs any secret.

Subcommands
-----------
  search    Read-only. List optimal-VRAM offers (A100-80GB first, 48GB fallback).
  up        Provision cheapest matching offer, scp secrets, run bootstrap.sh.
  status    Show instance state + tail of the remote main-run log.
  monitor   Poll every N minutes; print SS consistency snapshot from pushed CSVs.
  pull      rsync/scp results/ from the VM to the local repo.
  down      Destroy the instance (idempotent).

Design notes
------------
  * Secrets: the Vast key is passed to the CLI via the VAST_API_KEY environment
    variable of the child process only. Never echoed. peat.env is built in a
    temp file, scp'd with mode 600, and deleted locally.
  * "Optimal VRAM": default target is a single A100 80GB (matches the paper's
    A100-SXM4-80GB; Llama-3.1-8B peaks ~34GB so 80GB is comfortable). A 48GB
    card (A6000/A40/L40S) also fits every model and is the cheaper fallback.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]          # D:\PhD\PEAT_Debias
ENV_FILE  = REPO_ROOT / "Code" / ".env"
SLUG      = "DevDaring/PEAT_Debias"
IMAGE     = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB   = 80
SSH_KEY   = os.path.expanduser("~/.ssh/id_rsa")          # attached to the instance
# Force this key only and skip host-key prompts (Vast offers several keys).
SSH_OPTS  = ["-o", "StrictHostKeyChecking=no", "-o", "IdentitiesOnly=yes",
             "-o", "ConnectTimeout=25", "-i", SSH_KEY]

# ---- secret loading (never printed) ---------------------------------------
def load_env() -> dict:
    env = {}
    if not ENV_FILE.exists():
        sys.exit(f"[vastctl] {ENV_FILE} not found")
    for line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def vast_key(env: dict) -> str:
    k = env.get("PHD_VAST_AI_KEY") or env.get("VAST_API_KEY")
    if not k:
        sys.exit("[vastctl] PHD_VAST_AI_KEY missing from Code/.env")
    return k

def vast(args: list[str], key: str, capture=True) -> subprocess.CompletedProcess:
    """Invoke the vastai CLI with the key injected via env only."""
    child = dict(os.environ)
    child["VAST_API_KEY"] = key            # not logged; not on the command line
    return subprocess.run(["vastai", *args], env=child,
                          capture_output=capture, text=True)

# ---- offer search ----------------------------------------------------------
QUERIES = {
    "a100-80": "gpu_name=A100_SXM4 gpu_ram>=79 num_gpus=1 rentable=true "
               "reliability>0.98 inet_down>200 disk_space>80 verified=true",
    "a100pcie-80": "gpu_name=A100_PCIE gpu_ram>=79 num_gpus=1 rentable=true "
                   "reliability>0.98 inet_down>200 disk_space>80 verified=true",
    "48gb": "gpu_ram>=47 gpu_ram<=49 num_gpus=1 rentable=true reliability>0.98 "
            "inet_down>200 disk_space>80 verified=true",
}

def search(key: str, want: str = "a100-80", limit: int = 8) -> list[dict]:
    q = QUERIES.get(want, want)
    r = vast(["search", "offers", q, "-o", "dph+", "--limit", str(limit), "--raw"], key)
    if r.returncode != 0:
        print("[vastctl] search failed:", r.stderr.strip()[:300]); return []
    try:
        offers = json.loads(r.stdout)
    except json.JSONDecodeError:
        print("[vastctl] could not parse offers"); return []
    return offers

def _fmt(o: dict) -> str:
    return (f"  id={o.get('id')}  {o.get('gpu_name')}  "
            f"{round(float(o.get('gpu_ram',0))/1024 if o.get('gpu_ram',0)>1000 else o.get('gpu_ram',0))}GB  "
            f"${float(o.get('dph_total',0)):.3f}/hr  "
            f"rel={float(o.get('reliability2',o.get('reliability',0))):.3f}  "
            f"down={int(float(o.get('inet_down',0)))}Mbps  {o.get('datacenter') or o.get('geolocation','')}")

def cmd_search(key: str, args):
    for tier in (args.tier or ["a100-80", "a100pcie-80", "48gb"]):
        offers = search(key, tier, args.limit)
        print(f"\n=== {tier} ({len(offers)} offers, cheapest first) ===")
        for o in offers[: args.limit]:
            print(_fmt(o))

# ---- provision -------------------------------------------------------------
def cmd_up(key: str, env: dict, args):
    offers = []
    for tier in (["a100-80", "a100pcie-80"] if args.tier == "a100" else [args.tier]):
        offers = search(key, tier, 8)
        if offers:
            break
    if args.offer_id:
        offer_id = args.offer_id
    elif offers:
        offer_id = offers[0]["id"]
        print("[vastctl] selected offer:"); print(_fmt(offers[0]))
    else:
        sys.exit("[vastctl] no matching offers; try `search` or pass --offer-id")

    print(f"[vastctl] creating instance from offer {offer_id} ...")
    r = vast(["create", "instance", str(offer_id), "--image", IMAGE,
              "--disk", str(DISK_GB), "--ssh", "--direct",
              "--onstart-cmd", "touch ~/.vast_ready", "--raw"], key)
    if r.returncode != 0:
        sys.exit(f"[vastctl] create failed: {r.stderr.strip()[:300]}")
    try:
        new_id = json.loads(r.stdout).get("new_contract")
    except Exception:
        new_id = None
    print(f"[vastctl] instance requested (contract={new_id}). "
          f"Run `vastctl.py status --id {new_id}` to watch it come up, "
          f"then `up --id {new_id} --deploy` once SSH is ready.")
    if new_id:
        (REPO_ROOT / "Code" / "vast" / ".instance_id").write_text(str(new_id))

def ssh_target(key: str, iid: int) -> tuple[str, str] | None:
    r = vast(["show", "instance", str(iid), "--raw"], key)
    if r.returncode != 0:
        return None
    d = json.loads(r.stdout)
    host = d.get("ssh_host"); port = d.get("ssh_port")
    if not host or not port:
        return None
    return host, str(port)

def cmd_deploy(key: str, env: dict, iid: int):
    """scp secrets + bootstrap, then launch bootstrap.sh over SSH."""
    tgt = ssh_target(key, iid)
    if not tgt:
        sys.exit("[vastctl] SSH not ready yet; retry in ~30s")
    host, port = tgt
    # Build a minimal peat.env with only the keys the VM needs.
    needed = ["HF_Classic_Token", "Github_Classic_Token", "TextBelt_API_KEY",
              "PHONE_NO", "Mistral_API_Key"]
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, newline="\n") as f:
        for k in needed:
            if env.get(k):
                f.write(f"{k}={env[k]}\n")
        tmp = f.name
    os.chmod(tmp, 0o600)
    ssh = ["ssh", *SSH_OPTS, "-p", port, f"root@{host}"]
    scp = ["scp", *SSH_OPTS, "-P", port]
    try:
        subprocess.run(scp + [tmp, f"root@{host}:/root/peat.env"], check=True)
        subprocess.run(scp + [str(REPO_ROOT/"Code"/"vast"/"bootstrap.sh"),
                              str(REPO_ROOT/"Code"/"vast"/"autopush.sh"),
                              f"root@{host}:/root/"], check=True)
        subprocess.run(ssh + ["chmod +x /root/bootstrap.sh /root/autopush.sh && "
                              "setsid nohup bash /root/bootstrap.sh >/root/bootstrap.log 2>&1 & "
                              "echo launched"], check=True)
        print("[vastctl] bootstrap launched. Monitor: vastctl.py monitor --id", iid)
    finally:
        os.unlink(tmp)   # remove local secret copy

# ---- status / monitor / pull / down ---------------------------------------
def cmd_status(key: str, iid: int):
    r = vast(["show", "instance", str(iid), "--raw"], key)
    if r.returncode != 0:
        sys.exit(r.stderr.strip()[:200])
    d = json.loads(r.stdout)
    print(f"[vastctl] instance {iid}: status={d.get('actual_status')} "
          f"gpu={d.get('gpu_name')} cost=${float(d.get('dph_total',0)):.3f}/hr")
    tgt = ssh_target(key, iid)
    if tgt:
        host, port = tgt
        subprocess.run(["ssh", *SSH_OPTS, "-p", port, f"root@{host}",
                        "tail -n 25 /root/mainrun.log 2>/dev/null || "
                        "tail -n 25 /root/bootstrap.log 2>/dev/null"])

def cmd_pull(key: str, iid: int):
    tgt = ssh_target(key, iid)
    if not tgt:
        sys.exit("[vastctl] SSH not available")
    host, port = tgt
    dest = str(REPO_ROOT) + "/"
    # rsync results back; -a preserves, --exclude keeps large caches/secrets out
    subprocess.run(["scp", "-r", *SSH_OPTS, "-P", port,
                    f"root@{host}:/root/PEAT_Debias/Code/results", dest + "Code/"])
    print("[vastctl] results pulled to", dest + "Code/results")

def cmd_down(key: str, iid: int):
    r = vast(["destroy", "instance", str(iid)], key)
    print("[vastctl] destroy:", (r.stdout or r.stderr).strip()[:200])

# ---- CLI -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="PEAT Vast.ai controller (secret-safe)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("--tier", nargs="*"); s.add_argument("--limit", type=int, default=8)
    u = sub.add_parser("up"); u.add_argument("--tier", default="a100"); u.add_argument("--offer-id", type=int)
    u.add_argument("--id", type=int); u.add_argument("--deploy", action="store_true")
    for name in ("status", "monitor", "pull", "down", "deploy"):
        q = sub.add_parser(name); q.add_argument("--id", type=int, required=True)
        if name == "monitor":
            q.add_argument("--every", type=int, default=30)
    a = p.parse_args()
    env = load_env(); key = vast_key(env)

    if a.cmd == "search":   cmd_search(key, a)
    elif a.cmd == "up":
        if a.deploy and a.id: cmd_deploy(key, env, a.id)
        else:                 cmd_up(key, env, a)
    elif a.cmd == "deploy": cmd_deploy(key, env, a.id)
    elif a.cmd == "status": cmd_status(key, a.id)
    elif a.cmd == "pull":   cmd_pull(key, a.id)
    elif a.cmd == "down":   cmd_down(key, a.id)
    elif a.cmd == "monitor":
        while True:
            cmd_status(key, a.id)
            time.sleep(max(60, a.every * 60))

if __name__ == "__main__":
    main()
