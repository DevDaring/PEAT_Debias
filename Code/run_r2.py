#!/usr/bin/env python3
"""
PEAT revision-2 experiments — single entry point.

  python3 run_r2.py --smoke                 # 2-sample end-to-end test of every cell type
  python3 run_r2.py --worker jarvis-a       # claim and run cells until none remain
  python3 run_r2.py --status                # progress from the shared GitHub state
  python3 run_r2.py --cell peat_qwen_s42    # run one cell in this process (used internally)

Each cell runs in its own subprocess (clean GPU memory, crash isolation),
writes its artefacts incrementally, and is pushed to GitHub on completion, so
any number of workers on any provider can share the plan and a lost VM costs
at most the cell in flight.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def gpu_mem_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def run_subprocess(cell_id: str, timeout_min: float, log_path: Path, smoke: bool) -> int:
    env = dict(os.environ, PEAT_R2_SMOKE="1" if smoke else "0", PYTHONUNBUFFERED="1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {cell_id} start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        lf.flush()
        try:
            p = subprocess.run([sys.executable, str(HERE / "run_r2.py"), "--cell", cell_id],
                               cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=timeout_min * 60)
            return p.returncode
        except subprocess.TimeoutExpired:
            lf.write(f"\n===== TIMEOUT after {timeout_min} min\n")
            return 124


def trim_log(path: Path, keep: int = 400) -> None:
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        path.write_text("\n".join(lines[-keep:]) + "\n", encoding="utf-8")


def cmd_cell(cell_id: str) -> int:
    import torch
    from peat.r2.cells import run_cell
    from peat.r2.plan import get_cell
    meta = run_cell(get_cell(cell_id), device="cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"cell": cell_id, "status": meta["status"]}))
    return 0


def cmd_smoke(only=None) -> int:
    os.environ["PEAT_R2_SMOKE"] = "1"
    from peat.r2.common import R2_DIR
    from peat.r2.plan import SMOKE_CELLS
    cells = only or SMOKE_CELLS
    fails = []
    for cid in cells:
        t0 = time.time()
        log = R2_DIR / "smoke" / "cells" / cid / "log.txt"
        rc = run_subprocess(cid, 60, log, smoke=True)
        meta_p = R2_DIR / "smoke" / "cells" / cid / "meta.json"
        status = json.loads(meta_p.read_text()).get("status") if meta_p.exists() else "no-meta"
        print(f"[smoke] {cid:28s} rc={rc} status={status} {time.time() - t0:6.0f}s", flush=True)
        if rc != 0 or status != "done":
            fails.append(cid)
    print("SMOKE RESULT:", "PASS" if not fails else f"FAIL {fails}", flush=True)
    return 0 if not fails else 1


def cmd_worker(worker: str, prefixes=None, max_attempts: int = 2) -> int:
    from peat.r2 import sync
    from peat.r2.common import CELLS_DIR, ensure_r2_dirs
    from peat.r2.plan import build_plan
    ensure_r2_dirs()
    mem = gpu_mem_gb()
    attempts: dict[str, int] = {}
    while True:
        sync.pull()
        plan = sorted(build_plan(), key=lambda c: (c["priority"], c["model"], c["seed"]))
        todo = [c for c in plan if not sync.is_done(c["id"]) and c["vram"] <= mem - 1
                and attempts.get(c["id"], 0) < max_attempts
                and (not prefixes or any(c["id"].startswith(p) for p in prefixes))]
        n_done = sum(sync.is_done(c["id"]) for c in plan)
        sync.heartbeat(worker, {"gpu_mem_gb": round(mem, 1), "done": n_done, "total": len(plan),
                                "todo_for_me": len(todo)})
        claimed = None
        for c in [c for c in todo if sync.claim_owner(c["id"]) in (None, worker)]:
            if sync.try_claim(c["id"], worker):
                claimed = c
                break
        if claimed is None:
            if todo:
                time.sleep(120)  # remaining cells are held by other workers
                continue
            print(f"[{worker}] nothing left to run ({n_done}/{len(plan)} done)", flush=True)
            return 0
        cid = claimed["id"]
        attempts[cid] = attempts.get(cid, 0) + 1
        t0 = time.time()
        print(f"[{worker}] start {cid} (attempt {attempts[cid]})", flush=True)
        log = CELLS_DIR / cid / "log.txt"
        rc = run_subprocess(cid, max(4 * claimed["est_min"], 90), log, smoke=False)
        trim_log(log)
        status = "done" if sync.is_done(cid) else f"failed(rc={rc})"
        sync.heartbeat(worker, {"gpu_mem_gb": round(mem, 1), "last_cell": cid, "last_status": status,
                                "last_minutes": round((time.time() - t0) / 60, 1)})
        ok = sync.publish(cid, worker, status)
        print(f"[{worker}] {status} {cid} in {(time.time() - t0) / 60:.1f} min (pushed={ok})", flush=True)


def cmd_status() -> int:
    from peat.r2 import sync
    from peat.r2.common import R2_DIR
    from peat.r2.plan import build_plan
    sync.pull()
    plan = build_plan()
    done = [c for c in plan if sync.is_done(c["id"])]
    running = [(c["id"], sync.claim_owner(c["id"])) for c in plan
               if not sync.is_done(c["id"]) and sync.claim_owner(c["id"])]
    print(f"done {len(done)}/{len(plan)}; est. remaining "
          f"{sum(c['est_min'] for c in plan if c not in done) / 60:.1f} GPU-h")
    for cid, w in running:
        print(f"  running: {cid} on {w}")
    for f in sorted((R2_DIR / "workers").glob("*.json")):
        print("  worker", f.read_text().replace("\n", " ")[:300])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-cells", default="")
    ap.add_argument("--worker")
    ap.add_argument("--only", default="", help="comma-separated cell-id prefixes")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.cell:
        return cmd_cell(a.cell)
    if a.smoke:
        return cmd_smoke([c for c in a.smoke_cells.split(",") if c] or None)
    if a.worker:
        return cmd_worker(a.worker, [p for p in a.only.split(",") if p] or None)
    if a.status:
        return cmd_status()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
