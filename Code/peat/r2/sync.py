"""
GitHub as the shared, crash-safe state for the distributed revision-2 run.

A worker claims a cell by pushing results/r2/claims/<cell>.json, runs it, and
pushes the cell's directory (and adapter) as soon as it finishes. A cell is
complete when results/r2/cells/<cell>/meta.json says status "done". Claims
older than CLAIM_TTL_H hours are treated as abandoned (VM lost or credit
exhausted) and may be re-claimed. Only paths under Code/results/r2 are ever
staged, so no secret or unrelated file can be committed by a worker.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from peat.r2.common import CELLS_DIR, CLAIMS_DIR, R2_DIR
from peat.utils import PROJECT_ROOT

REPO = PROJECT_ROOT.parent
CLAIM_TTL_H = float(os.environ.get("PEAT_R2_CLAIM_TTL_H", "3"))
REL = "Code/results/r2"


def git(*args, check=True) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", *args], cwd=REPO, env=env, capture_output=True, text=True,
                          check=check)


def pull() -> None:
    r = git("pull", "--rebase", "--autostash", "-q", check=False)
    if r.returncode != 0:
        git("rebase", "--abort", check=False)
        git("fetch", "-q", "origin", check=False)
        # Local commits only ever touch results/r2 and were already pushed or
        # are re-creatable; align with the remote and keep untracked results.
        git("reset", "--mixed", "-q", "origin/main", check=False)


def push(paths: list[str], message: str, retries: int = 6) -> bool:
    for p in paths:
        git("add", "-f", "--", p, check=False)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return True
    git("commit", "-q", "-m", message, check=False)
    for k in range(retries):
        if git("push", "-q", "origin", "HEAD:main", check=False).returncode == 0:
            return True
        time.sleep(3 + 4 * k)
        pull()
    return False


def is_done(cell_id: str) -> bool:
    m = CELLS_DIR / cell_id / "meta.json"
    if not m.exists():
        return False
    try:
        return json.loads(m.read_text()).get("status") == "done"
    except Exception:
        return False


def claim_owner(cell_id: str):
    p = CLAIMS_DIR / f"{cell_id}.json"
    if not p.exists():
        return None
    try:
        c = json.loads(p.read_text())
    except Exception:
        return None
    if time.time() - c.get("t", 0) > CLAIM_TTL_H * 3600:
        return None
    return c.get("worker")


def try_claim(cell_id: str, worker: str) -> bool:
    pull()
    if is_done(cell_id):
        return False
    owner = claim_owner(cell_id)
    if owner not in (None, worker):
        return False
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    p = CLAIMS_DIR / f"{cell_id}.json"
    p.write_text(json.dumps({"worker": worker, "t": time.time()}))
    if not push([f"{REL}/claims/{cell_id}.json"], f"r2: {worker} claims {cell_id}"):
        return False
    pull()
    return claim_owner(cell_id) == worker


def publish(cell_id: str, worker: str, status: str) -> bool:
    paths = [f"{REL}/cells/{cell_id}"]
    ad = R2_DIR / "adapters" / f"{cell_id}.safetensors"
    if ad.exists():
        paths.append(f"{REL}/adapters/{cell_id}.safetensors")
    hb = R2_DIR / "workers" / f"{worker}.json"
    if hb.exists():
        paths.append(f"{REL}/workers/{worker}.json")
    return push(paths, f"r2: {worker} {status} {cell_id}")


def heartbeat(worker: str, info: dict) -> None:
    p = R2_DIR / "workers" / f"{worker}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(info, worker=worker, t=time.time(),
                                 t_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=2))
