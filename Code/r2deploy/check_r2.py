#!/usr/bin/env python3
"""
Hourly correctness audit of the revision-2 run (execution-monitoring rule).

  python Code/r2deploy/check_r2.py

1. liveness per worker host (process, GPU utilisation, log age);
2. error scan of worker logs and failed cells;
3. progress and ETA from the shared GitHub state;
4. semantic checks on every finished cell (row counts, ranges, NaN/inf,
   training-loss finiteness, adapter present for trained cells);
5. disk and credit (Jarvis balance).
Hosts are listed in Code/r2deploy/hosts.json as {"name": "user@ip", ...}.
"""

import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from peat.r2 import sync  # noqa: E402
from peat.r2.common import ADAPTER_DIR, CELLS_DIR  # noqa: E402
from peat.r2.plan import build_plan  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
EXPECT_N = {"crows_en_n": 1508, "delta_val_n": 211, "cloze_english_n": 1411}


def ssh(host, cmd, timeout=60):
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no", host, cmd],
                           capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout.strip()
    except Exception as e:
        return f"SSH-ERROR {e}"


def check_cell(c) -> list[str]:
    probs = []
    m = json.loads((CELLS_DIR / c["id"] / "meta.json").read_text())
    s = m.get("summary", {})
    for k, v in s.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) and not k.endswith("_sd"):
            probs.append(f"{k}=nan/inf")
    for k, n in EXPECT_N.items():
        if k in s and s[k] != n:
            probs.append(f"{k}={s[k]} (expected {n})")
    for k, v in s.items():
        if (k.endswith("_ss") or k.endswith("_ss_fp32")) and isinstance(v, (int, float)) and not (5 <= v <= 95):
            probs.append(f"{k}={v:.1f} out of range")
    if "wikitext_ppl" in s and not (2 < s["wikitext_ppl"] < 200):
        probs.append(f"ppl={s['wikitext_ppl']}")
    if c["kind"] in ("peat", "lora"):
        if not (ADAPTER_DIR / f"{c['id']}.safetensors").exists():
            probs.append("adapter missing")
        st = CELLS_DIR / c["id"] / "train_steps.csv"
        if c["kind"] == "peat" and st.exists():
            import pandas as pd
            d = pd.read_csv(st)
            if not d["L_total"].apply(math.isfinite).all():
                probs.append("non-finite training loss")
            if len(d) and d["L_neut"].tail(20).mean() > d["L_neut"].head(20).mean() * 1.5:
                probs.append("L_neut increased over training")
    if m.get("smoke"):
        probs.append("SMOKE OUTPUT IN MAIN RESULTS")
    return probs


def main():
    now = datetime.now(IST)
    print(f"=== PEAT r2 audit {now:%Y-%m-%d %H:%M} IST ===")
    sync.pull()
    plan = build_plan()
    done = [c for c in plan if sync.is_done(c["id"])]
    failed = []
    for c in plan:
        mp = CELLS_DIR / c["id"] / "meta.json"
        if mp.exists() and json.loads(mp.read_text()).get("status") == "failed":
            failed.append(c["id"])
    rem = [c for c in plan if c not in done]
    print(f"progress: {len(done)}/{len(plan)} done; failed-status: {failed or 'none'}; "
          f"remaining est {sum(c['est_min'] for c in rem) / 60:.1f} GPU-h (plan estimate)")
    # Observed rate
    times = []
    for c in done:
        m = json.loads((CELLS_DIR / c["id"] / "meta.json").read_text())
        if m.get("finished") and m.get("started"):
            times.append((m["finished"], (m["finished"] - m["started"]) / 60, c["est_min"]))
    if times:
        ratio = sum(t[1] for t in times) / max(sum(t[2] for t in times), 1e-9)
        print(f"observed/planned time ratio {ratio:.2f}")
    bad = {}
    for c in done:
        p = check_cell(c)
        if p:
            bad[c["id"]] = p
    print("semantic problems:", json.dumps(bad, indent=1) if bad else "none")
    hosts_f = HERE / "hosts.json"
    hosts = json.loads(hosts_f.read_text()) if hosts_f.exists() else {}
    for name, host in hosts.items():
        out = ssh(host, "pgrep -fc '[r]un_r2.py' ; nvidia-smi --query-gpu=utilization.gpu,memory.used "
                        "--format=csv,noheader; df -h /home | tail -1 | awk '{print $5}'; "
                        f"tail -2 /home/worker_{name}.log; stat -c %Y /home/worker_{name}.log; date +%s; "
                        f"grep -ciE 'Traceback|out of memory|Killed' /home/worker_{name}.log")
        print(f"--- {name} ({host})\n{out}")
    try:
        from dotenv import dotenv_values
        from jarvislabs import Client
        k = dotenv_values(HERE.parent / ".env")["JARVAIS_LAB_2009_API_KEY"].strip()
        with Client(api_key=k) as cl:
            print("jarvis:", cl.account.balance(), [(i.machine_id, i.name, i.status) for i in cl.instances.list()])
    except Exception as e:
        print("jarvis check error", e)


if __name__ == "__main__":
    main()
