#!/usr/bin/env python3
"""Release the claims of every cell whose meta.json says "failed", so any
worker can retry it after a code fix.   python Code/r2deploy/release_failed.py"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peat.r2 import sync  # noqa: E402
from peat.r2.common import CELLS_DIR, CLAIMS_DIR  # noqa: E402

sync.pull()
paths = []
for m in CELLS_DIR.glob("*/meta.json"):
    cid = m.parent.name
    if json.loads(m.read_text()).get("status") == "failed":
        c = CLAIMS_DIR / f"{cid}.json"
        if c.exists():
            c.unlink()
            sync.git("rm", "-q", "--cached", "--", f"{sync.REL}/claims/{cid}.json", check=False)
            paths.append(cid)
print("released:", paths)
if paths:
    sync.git("commit", "-q", "-m", f"r2: release claims of failed cells after fix ({len(paths)})", check=False)
    for _ in range(5):
        if sync.git("push", "-q", "origin", "HEAD:main", check=False).returncode == 0:
            print("pushed")
            break
        sync.pull()
