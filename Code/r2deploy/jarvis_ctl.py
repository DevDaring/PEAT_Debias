#!/usr/bin/env python3
"""
Jarvis Labs controller for the revision-2 workers (local machine).

  python Code/r2deploy/jarvis_ctl.py balance
  python Code/r2deploy/jarvis_ctl.py gpus
  python Code/r2deploy/jarvis_ctl.py create --gpu A100 --name peat-r2-a
  python Code/r2deploy/jarvis_ctl.py list
  python Code/r2deploy/jarvis_ctl.py push-env --id <machine_id>
  python Code/r2deploy/jarvis_ctl.py run --id <machine_id> -- "<remote command>"
  python Code/r2deploy/jarvis_ctl.py destroy --id <machine_id>

The API key is read from Code/.env (JARVAIS_LAB_2009_API_KEY) and passed to
the SDK in memory only. push-env copies HF and GitHub tokens to the VM as
~/peat_r2.env (mode 600) without printing them.
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values

ENV = dotenv_values(Path(__file__).resolve().parents[1] / ".env")


def client():
    from jarvislabs import Client
    return Client(api_key=ENV["JARVAIS_LAB_2009_API_KEY"].strip())


def ssh_parts(inst):
    """Parse 'ssh -p PORT user@host' into (user@host, port)."""
    toks = shlex.split(inst.ssh_command)
    port = toks[toks.index("-p") + 1] if "-p" in toks else "22"
    target = [t for t in toks if "@" in t][0]
    return target, port


def ssh(inst, command: str, timeout=3600):
    target, port = ssh_parts(inst)
    return subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
                           "-p", port, target, command], capture_output=True, text=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action")
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--name", default="peat-r2")
    ap.add_argument("--storage", type=int, default=100)
    ap.add_argument("--id", type=int)
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    with client() as c:
        if a.action == "balance":
            print(c.account.balance())
        elif a.action == "gpus":
            for g in c.account.gpu_availability():
                d = g.model_dump()
                print(d["gpu_type"], d["region"], d["vram"], d["workload_type"], "free", d["num_free_devices"],
                      "INR/h", d["price_per_hour"])
        elif a.action == "create":
            inst = c.instances.create(gpu_type=a.gpu, name=a.name, storage=a.storage)
            print(inst.machine_id, inst.status, inst.ssh_command)
        elif a.action == "list":
            for i in c.instances.list():
                print(i.machine_id, i.name, i.status, i.gpu_type, getattr(i, "ssh_command", ""))
        elif a.action == "push-env":
            inst = c.instances.get(a.id)
            body = f"HF_KEY={ENV['HUGGINGFACE_TOKEN'].strip()}\nGH_TOKEN={ENV['Github_Classic_Token'].strip()}\n"
            target, port = ssh_parts(inst)
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".env", newline="\n") as f:
                f.write(body)
                tmp = f.name
            try:
                r = subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no", "-P", port, tmp,
                                    f"{target}:peat_r2.env"], capture_output=True, text=True)
                print("scp rc", r.returncode, r.stderr[-300:])
                print(ssh(inst, "chmod 600 ~/peat_r2.env && wc -l ~/peat_r2.env").stdout)
            finally:
                os.unlink(tmp)
        elif a.action == "run":
            inst = c.instances.get(a.id)
            r = ssh(inst, " ".join(a.rest))
            print(r.stdout[-6000:])
            print(r.stderr[-2000:], file=sys.stderr)
        elif a.action == "destroy":
            print(c.instances.destroy(a.id))
        else:
            ap.error(a.action)


if __name__ == "__main__":
    main()
