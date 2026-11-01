"""
Sign and magnitude of the CrowS-Pairs gap in the published (R1) per-pair
results (Reviewer #5, weakness 1). Uses results/raw/*: Base (seed 42),
PEAT and LoRA-Vanilla-SFT (five seeds each, per-pair seed means).

    python3 -m peat.r2.published_delta   ->  results/r2/tables/published_delta_sign.csv
"""

import glob

import numpy as np
import pandas as pd

from peat.r2.common import R2_DIR
from peat.utils import RAW_DIR


def _delta(path):
    d = pd.read_csv(path).drop_duplicates("idx", keep="last").sort_values("idx").set_index("idx")
    st = d["stereo_antistereo"].eq("stereo").values
    d["delta"] = np.where(st, d.score_more - d.score_less, d.score_less - d.score_more)
    return d


def main():
    rows = []
    for m in ["bert-base", "modernbert-base", "nomicbert", "qwen2.5-1.5b"]:
        b = _delta(glob.glob(str(RAW_DIR / f"baselines/base/{m}/seed_42/*.csv"))[0])
        d0 = b["delta"].values
        for name, pat in (("Base", None), ("LoRA-Vanilla-SFT", f"baselines/lora_vanilla_sft/{m}/seed_*/*.csv"),
                          ("PEAT", f"peat/{m}/seed_*/*.csv")):
            if pat is None:
                Ds = [d0]
            else:
                Ds = [_delta(f)["delta"].loc[b.index].values for f in sorted(glob.glob(str(RAW_DIR / pat)))]
            per = []
            for D in Ds:  # statistics per seed, then averaged over seeds
                ch = D - d0
                per.append({
                    "SS": 100 * (D > 0).mean(), "mean": D.mean(), "median": np.median(D),
                    "mean_abs": np.abs(D).mean(),
                    "band_plus_0.5": 100 * ((D > 0) & (D < 0.5)).mean(),
                    "band_minus_0.5": 100 * ((D > -0.5) & (D <= 0)).mean(),
                    "shift_on_base_pos": ch[d0 > 0].mean(), "shift_on_base_neg": ch[d0 < 0].mean(),
                    "corr_with_base": float(np.corrcoef(d0, D)[0, 1]),
                })
            row = {"model": m, "method": name, "n_seeds": len(Ds)}
            row.update(pd.DataFrame(per).mean().to_dict())
            rows.append(row)
    out = pd.DataFrame(rows)
    (R2_DIR / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(R2_DIR / "tables" / "published_delta_sign.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
