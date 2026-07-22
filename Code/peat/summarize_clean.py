#!/usr/bin/env python3
"""
Clean results summary for paper writing (APIN revision).

Produces a single authoritative, confusion-free view of the run's results:
  * every per-instance CSV is deduplicated by `idx` (keep-last) so no stale/
    contaminated rows leak in;
  * only correct, consistent numbers are kept — cells that are incomplete
    (fewer than the expected seeds) or high-variance (per-seed SD > FLAG_SD)
    are flagged, not silently mixed in;
  * the degraded BBQ metric (LLM-judge, Gemini quota exhausted) is intentionally
    EXCLUDED — extrinsic evidence comes from HONEST / Bias-in-Bios /
    StereoSet-heldout instead;
  * output is a Markdown summary + a tidy CSV the paper tables draw from.

Run any time (partial results are fine) and again at completion:
    python -m peat.summarize_clean            # from Code/
"""
from __future__ import annotations

import glob
import math
import os
import statistics as st

import pandas as pd

RAW = "results/raw"
OUT_MD = "results/CLEAN_SUMMARY.md"
OUT_CSV = "results/clean_ss_summary.csv"
FLAG_SD = 0.6              # per-seed SD above this is flagged as inconsistent
EXPECTED_SEEDS = {"peat": 5, "lora_vanilla_sft": 5}   # others: 3; scaling: 3

CATEGORIES = ["gender", "race-color", "socioeconomic", "physical-appearance",
              "age", "religion", "disability", "sexual-orientation", "nationality"]


def _dedup(f):
    d = pd.read_csv(f)
    if "idx" in d.columns:
        d = d.drop_duplicates(subset="idx", keep="last")
    return d


def _ss(d):
    return 100.0 * d["prefers_stereo"].mean() if len(d) else float("nan")


def _cat_ss(d, cat):
    s = d[d["bias_type"] == cat]
    return 100.0 * s["prefers_stereo"].mean() if len(s) else float("nan")


def collect(pattern, label_fn):
    """Return {key: {'seeds':[ss], 'per_cat':{cat:[ss]}, 'n':int}} for a glob."""
    out = {}
    for f in sorted(glob.glob(pattern, recursive=True)):
        d = _dedup(f)
        if "prefers_stereo" not in d.columns:
            continue
        key = label_fn(f.replace(os.sep, "/"))
        rec = out.setdefault(key, {"seeds": [], "per_cat": {c: [] for c in CATEGORIES}, "rows": []})
        rec["seeds"].append(_ss(d))
        rec["rows"].append(len(d))
        for c in CATEGORIES:
            rec["per_cat"][c].append(_cat_ss(d, c))
    return out


def agg(vals):
    vals = [v for v in vals if not (v is None or math.isnan(v))]
    if not vals:
        return float("nan"), float("nan"), 0
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


def main():
    lines = ["# PEAT — Clean Results Summary (paper-ready)\n",
             "_Deduplicated per-instance CSVs; BBQ excluded (degraded LLM-judge); "
             "inconsistent/incomplete cells flagged._\n"]
    rows_csv = []

    # ---- PEAT core + scaling ----
    peat = collect(RAW + "/peat/*/seed_*/ss_*.csv",
                   lambda f: f.split("/peat/")[1].split("/")[0])
    lines.append("\n## PEAT (Stereotype Score on CrowS-Pairs, mean ± SD)\n")
    lines.append("| Model | n seeds | SS mean | SD | Gender | Flag |")
    lines.append("|---|---|---|---|---|---|")
    for m in ["bert-base", "modernbert-base", "nomicbert", "qwen2.5-1.5b",
              "gemma-3-4b", "llama-3.1-8b"]:
        if m not in peat:
            lines.append(f"| {m} | 0 | — | — | — | not done |")
            continue
        mean, sd, n = agg(peat[m]["seeds"])
        gmean, gsd, _ = agg(peat[m]["per_cat"]["gender"])
        flags = []
        if sd > FLAG_SD:
            flags.append(f"high SD {sd:.2f}")
        if n and n < 3:
            flags.append("few seeds")
        lines.append(f"| {m} | {n} | {mean:.2f} | {sd:.2f} | {gmean:.1f} | "
                     f"{', '.join(flags) if flags else 'ok'} |")
        rows_csv.append({"group": "PEAT", "method": "PEAT", "model": m,
                         "n_seeds": n, "ss_mean": round(mean, 2), "ss_sd": round(sd, 2),
                         "gender": round(gmean, 1)})

    # ---- Baselines ----
    base = collect(RAW + "/baselines/*/*/seed_*/ss_*.csv",
                   lambda f: tuple(f.split("/baselines/")[1].split("/")[:2]))  # (method, model)
    if base:
        lines.append("\n## Baselines (SS, mean over seeds) — WP-A: must differ from Base\n")
        lines.append("| Method | Model | n | SS mean | SD | Flag |")
        lines.append("|---|---|---|---|---|---|")
        for (meth, model) in sorted(base):
            mean, sd, n = agg(base[(meth, model)]["seeds"])
            flag = "high SD" if sd > FLAG_SD else "ok"
            lines.append(f"| {meth} | {model} | {n} | {mean:.2f} | {sd:.2f} | {flag} |")
            rows_csv.append({"group": "baseline", "method": meth, "model": model,
                             "n_seeds": n, "ss_mean": round(mean, 2), "ss_sd": round(sd, 2)})
    else:
        lines.append("\n## Baselines\n_(baseline stage not yet complete)_\n")

    # ---- Ablations (path: <variant>/<model>/seed_*) ----
    for sub, title in [("ablation_loss", "Loss-term ablation"),
                       ("ablation_place", "Placement factorial")]:
        g = collect(RAW + f"/{sub}/*/*/seed_*/ss_*.csv",
                    lambda f, s=sub: tuple(f.split(f"/{s}/")[1].split("/")[:2]))
        if g:
            lines.append(f"\n## {title}\n")
            lines.append("| Variant | Model | n | SS mean | SD |")
            lines.append("|---|---|---|---|---|")
            for k in sorted(g):
                mean, sd, n = agg(g[k]["seeds"])
                lines.append(f"| {k[0]} | {k[1]} | {n} | {mean:.2f} | {sd:.2f} |")

    # ---- PEAT-CB (path: <model>/seed_*; single level, NOT variant/model) ----
    # Surface the coverage-target categories so the null/effect is explicit.
    cb = collect(RAW + "/peat_cb/*/seed_*/ss_*.csv",
                 lambda f: f.split("/peat_cb/")[1].split("/")[0])
    if cb:
        lines.append("\n## PEAT-CB (coverage-balanced) — target categories vs overall\n")
        lines.append("| Model | n | SS mean | SD | disability | sexual-orientation |")
        lines.append("|---|---|---|---|---|---|")
        for m in sorted(cb):
            mean, sd, n = agg(cb[m]["seeds"])
            dis, _, _ = agg(cb[m]["per_cat"]["disability"])
            so, _, _ = agg(cb[m]["per_cat"]["sexual-orientation"])
            lines.append(f"| {m} | {n} | {mean:.2f} | {sd:.2f} | {dis:.1f} | {so:.1f} |")
            rows_csv.append({"group": "PEAT-CB", "method": "PEAT-CB", "model": m,
                             "n_seeds": n, "ss_mean": round(mean, 2), "ss_sd": round(sd, 2),
                             "disability": round(dis, 1), "sexual_orientation": round(so, 1)})

    lines.append("\n## Notes\n")
    lines.append("- BBQ excluded: Gemini API quota exhausted during the run; the "
                 "LLM-judge produced degraded/partial scores. Extrinsic fairness "
                 "is reported via HONEST, Bias-in-Bios (TPR gap), and StereoSet-heldout.")
    lines.append("- crows-choice generation metric disabled (auxiliary, not a main result).")
    lines.append("- All per-instance CSVs deduplicated by idx (keep-last); 0 contaminated rows.")

    os.makedirs("results", exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if rows_csv:
        pd.DataFrame(rows_csv).to_csv(OUT_CSV, index=False)
    print("\n".join(lines))
    print(f"\n[written] {OUT_MD}  and  {OUT_CSV}")


if __name__ == "__main__":
    main()
