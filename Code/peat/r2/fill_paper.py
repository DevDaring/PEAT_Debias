"""
Insert the generated LaTeX tables (results/r2/latex/<name>.tex) into the
manuscript at the markers "%%TAB:<name>%%". The inserted block is enclosed
by "%%TAB:<name>%%" ... "%%END:<name>%%", so re-running replaces it in place.
The journal requires a single .tex file, hence no \\input.

    python3 -m peat.r2.fill_paper ../Submission/AI_sn-article.tex
"""

import re
import sys
from pathlib import Path

from peat.r2.common import R2_DIR


def fill(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    done = []
    for name in sorted(set(re.findall(r"%%TAB:([a-z0-9_]+)%%", text))):
        body = (R2_DIR / "latex" / f"{name}.tex").read_text(encoding="utf-8").rstrip("\n")
        block = f"%%TAB:{name}%%\n{body}\n%%END:{name}%%"
        pat = re.compile(rf"%%TAB:{name}%%(?:\n.*?%%END:{name}%%)?", re.S)
        text = pat.sub(lambda m: block, text, count=1)
        done.append(name)
    path.write_text(text, encoding="utf-8", newline="\n")
    return done


if __name__ == "__main__":
    print(fill(Path(sys.argv[1])))
