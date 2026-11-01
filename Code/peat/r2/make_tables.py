"""
LaTeX tables for the revision from results/r2/tables/*.csv (written by
peat.r2.analysis). Every number in the new tables of the manuscript comes
from here, so nothing is transcribed by hand.

    python3 -m peat.r2.analysis && python3 -m peat.r2.make_tables
    -> results/r2/latex/*.tex
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from peat.r2.common import R2_DIR

TAB = R2_DIR / "tables"
OUT = R2_DIR / "latex"
NAME = {"bert": "BERT-base", "modernbert": "ModernBERT", "nomic": "NomicBERT", "qwen": "Qwen2.5-1.5B",
        "gemma": "Gemma-3-4B", "llama": "Llama-3.1-8B"}
METH = {"base": "Base", "peat": "\\PEAT{}", "lora": "LoRA", "peat_cb": "\\PEAT{}-CB"}


def f(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "---"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return f"${s}$" if sign else s


def pm(mu, sd, d=2):
    if mu is None or (isinstance(mu, float) and math.isnan(mu)):
        return "---"
    if sd is None or (isinstance(sd, float) and (math.isnan(sd) or sd == 0)):
        return f(mu, d)
    return f"{mu:.{d}f}\\,{{\\scriptsize$\\pm${sd:.{d}f}}}"


def pval(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def load(name):
    p = TAB / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def wrap(rows, header, colspec, size="\\footnotesize"):
    body = "\n".join("    " + " & ".join(r) + " \\\\" if r != "MID" else "    \\midrule" for r in rows)
    return (f"  {size}\n  \\begin{{adjustbox}}{{max width=\\textwidth}}\n  \\begin{{tabular}}{{{colspec}}}\n"
            f"    \\toprule\n    {header} \\\\\n    \\midrule\n{body}\n    \\bottomrule\n  \\end{{tabular}}\n"
            f"  \\end{{adjustbox}}\n")


def t_surrogates():
    d = load("surrogates")
    lab = {"huber": "Huber (\\PEAT{})", "l2": "squared", "l1": "absolute", "logcosh": "symmetric logistic",
           "logistic": "one-sided logistic", "hinge": "one-sided hinge"}
    rows = []
    for m in ["bert", "qwen"]:
        g = d[d.model == m]
        if g.empty:
            continue
        if rows:
            rows.append("MID")
        for _, r in g.iterrows():
            rows.append([NAME[m] if r.surrogate == "huber" else "", lab[r.surrogate],
                         pm(r.crows_en_ss, r.crows_en_ss_sd), pm(r.cloze_english_ss, r.cloze_english_ss_sd),
                         pm(r.delta_val_ss, r.delta_val_ss_sd), f(r.delta_val_median), f(r.delta_val_mean_abs),
                         pval(r.get("perm_p_vs_huber", float("nan")))])
    return wrap(rows, "Model & Surrogate & CrowS SS & Cloze SS (en) & Held-out SS & median $\\Delta$ & mean $|\\Delta|$ & $p$ vs Huber",
                "@{}llcccccc@{}")


def t_bounds():
    d = load("bounds")
    d = d[(d.method == "peat") & (d.split == "val")]
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        g = d[d.model == m]
        for _, r in g.iterrows():
            if r.eps not in (0.25, 0.5, 1.0):
                continue
            rows.append([NAME[m] if r.eps == 0.25 else "", f(r.eps), f(r.L_neut, 3), f(r.SS_eps_realised, 1),
                         f(r.markov_bound, 1), f(r.cantelli_bound, 1), f(abs(r.SS - 50), 1),
                         f(r.tail_term, 1), f(r.band_imbalance_term, 1)])
    return wrap(rows, "Model & $\\varepsilon$ & $\\Lneut$ & $\\mathrm{SS}_\\varepsilon$ & Markov & Cantelli & $|\\mathrm{SS}-50|$ & tail term & band term",
                "@{}lcccccccc@{}")


def t_utility_dec():
    d = load("utility")
    tasks = [("arc_easy_acc_norm", "ARC-e"), ("arc_challenge_acc_norm", "ARC-c"), ("hellaswag_acc_norm", "HS"),
             ("piqa_acc_norm", "PIQA"), ("winogrande_acc", "WG"), ("boolq_acc", "BoolQ"),
             ("lambada_openai_acc", "LAMB.")]
    rows = []
    for m in ["qwen", "gemma", "llama"]:
        g = d[d.model == m]
        if g.empty:
            continue
        if rows:
            rows.append("MID")
        for meth in ["base", "lora", "peat"]:
            r = g[g.method == meth]
            if r.empty:
                continue
            r = r.iloc[0]
            accs = [r.get(k, float("nan")) for k, _ in tasks]
            rows.append([NAME[m] if meth == "base" else "", METH[meth], pm(r.get("wikitext_ppl"), r.get("wikitext_ppl_sd")),
                         f(r.get("wikitext_kl", float("nan")), 4), f(r.get("wikitext_tv", float("nan")), 4)]
                        + [f(100 * a, 1) for a in accs] + [f(100 * np.nanmean(accs), 1)])
    hdr = "Model & Method & PPL & KL & TV & " + " & ".join(t for _, t in tasks) + " & Avg."
    return wrap(rows, hdr, "@{}ll" + "c" * (4 + len(tasks)) + "@{}")


def t_utility_enc():
    d = load("utility")
    rows = []
    for m in ["bert", "modernbert", "nomic"]:
        g = d[d.model == m]
        for meth in ["base", "lora", "peat"]:
            r = g[g.method == meth]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([NAME[m] if meth == "base" else "", METH[meth],
                         pm(r.get("wikitext_pppl"), r.get("wikitext_pppl_sd")),
                         f(r.get("wikitext_kl", float("nan")), 4), f(r.get("wikitext_tv", float("nan")), 4)])
    return wrap(rows, "Model & Method & pseudo-PPL & KL & TV", "@{}llccc@{}")


def t_multilingual():
    d = load("multilingual")
    rows = []
    for m in ["qwen", "gemma", "llama"]:
        g = d[d.model == m]
        if g.empty:
            continue
        if rows:
            rows.append("MID")
        for meth in ["base", "lora", "peat"]:
            r = g[g.method == meth]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([NAME[m] if meth == "base" else "", METH[meth],
                         pm(r.crows_fr_ss, r.crows_fr_ss_sd), pm(r.mcrows_hindi_ss, r.mcrows_hindi_ss_sd),
                         pm(r.mcrows_bengali_ss, r.mcrows_bengali_ss_sd), pm(r.cloze_hindi_ss, r.cloze_hindi_ss_sd),
                         pm(r.cloze_bengali_ss, r.cloze_bengali_ss_sd),
                         pm(r.bbq_s_amb, r.bbq_s_amb_sd, 3), pm(r.bbq_acc_ambig, r.bbq_acc_ambig_sd, 3),
                         pm(r.bold_gender_gap, r.bold_gender_gap_sd, 3), pm(r.bold_race_gap, r.bold_race_gap_sd, 3)])
    hdr = ("Model & Method & FR & HI & BN & HI cloze & BN cloze & BBQ $s_{\\mathrm{amb}}$ & BBQ acc & BOLD gender & BOLD race")
    return wrap(rows, hdr, "@{}llccccccccc@{}")


def t_dynamics():
    d = load("loss_dynamics")
    rows = []
    for _, r in d.iterrows():
        rows.append([NAME.get(r.model, r.model), f(r.L_neut_start, 3), f(r.L_neut_end, 3), f(r.w_pair_end, 4),
                     f(r.w_kl_end, 4), f(r.g_neut_end, 3), f(r.g_pair_end, 3), f(r.g_kl_end, 3),
                     f(r.cos_neut_pair_end, 2), f(r.cos_neut_kl_end, 2)])
    hdr = ("Model & $\\Lneut$ start & $\\Lneut$ end & $\\lambda_1\\Lpair$ & $\\lambda_2\\Lkl$ & "
           "$\\|g_{\\mathrm{neut}}\\|$ & $\\|g_{\\mathrm{pair}}\\|$ & $\\|g_{\\mathrm{kl}}\\|$ & "
           "$\\cos(g_{\\mathrm{neut}},g_{\\mathrm{pair}})$ & $\\cos(g_{\\mathrm{neut}},g_{\\mathrm{kl}})$")
    return wrap(rows, hdr, "@{}lccccccccc@{}")


def t_placement():
    d = load("placement")
    lab = {"first2": "first two", "middle2": "middle two", "last2": "last two (\\PEAT{})", "all": "all blocks"}
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen"]:
        g = d[d.model == m]
        if g.empty:
            continue
        if rows:
            rows.append("MID")
        for pl in ["first2", "middle2", "last2", "all"]:
            r = g[g.placement == pl]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([NAME[m] if pl == "first2" else "", lab[pl], str(int(r.n)),
                         pm(r.crows_en_ss, r.crows_en_ss_sd), pm(r.cloze_english_ss, r.cloze_english_ss_sd),
                         pm(r.delta_val_ss, r.delta_val_ss_sd)])
    return wrap(rows, "Model & Placement & seeds & CrowS SS & Cloze SS (en) & Held-out SS", "@{}llcccc@{}")


def t_hp():
    d = load("hp")
    lab = {"r2": "$r=2$", "r4": "$r=4$ (\\PEAT{})", "r8": "$r=8$", "r16": "$r=16$", "lr5e-5": "lr $5\\times10^{-5}$",
           "lr1e-4": "lr $10^{-4}$ (\\PEAT{})", "lr2e-4": "lr $2\\times10^{-4}$"}
    rows = []
    for m in ["bert", "qwen"]:
        g = d[d.model == m]
        if g.empty:
            continue
        if rows:
            rows.append("MID")
        first = True
        for s in ["r2", "r4", "r8", "r16", "lr5e-5", "lr1e-4", "lr2e-4"]:
            r = g[g.setting == s]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([NAME[m] if first else "", lab[s], pm(r.crows_en_ss, r.crows_en_ss_sd),
                         pm(r.cloze_english_ss, r.cloze_english_ss_sd), pm(r.delta_val_ss, r.delta_val_ss_sd)])
            first = False
    return wrap(rows, "Model & Setting & CrowS SS & Cloze SS (en) & Held-out SS", "@{}llccc@{}")


def t_cb():
    d, t = load("cb"), load("cb_tests")
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma"]:
        g = d[d.model == m]
        if g.empty:
            continue
        for meth in ["peat", "peat_cb"]:
            r = g[g.method == meth]
            if r.empty:
                continue
            r = r.iloc[0]
            p_dis = p_so = float("nan")
            if meth == "peat_cb" and len(t):
                tt = t[(t.model == m)]
                p_dis = tt[tt.category == "disability"]["p_holm"].values[0] if (tt.category == "disability").any() else float("nan")
                p_so = tt[tt.category == "sexual-orientation"]["p_holm"].values[0] if (tt.category == "sexual-orientation").any() else float("nan")
            rows.append([NAME[m] if meth == "peat" else "", METH[meth], pm(r.crows_en_ss, r.crows_en_ss_sd),
                         pm(r["crows_en_ss_disability"], r["crows_en_ss_disability_sd"], 1),
                         pval(p_dis) if meth == "peat_cb" else "",
                         pm(r["crows_en_ss_sexual-orientation"], r["crows_en_ss_sexual-orientation_sd"], 1),
                         pval(p_so) if meth == "peat_cb" else "",
                         pm(r["crows_en_ss_gender"], r["crows_en_ss_gender_sd"], 1)])
    return wrap(rows, "Model & Variant & Overall & Disability & $p_{\\mathrm{Holm}}$ & Sexual orient. & $p_{\\mathrm{Holm}}$ & Gender",
                "@{}llcccccc@{}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("surrogates", t_surrogates), ("bounds", t_bounds), ("utility_dec", t_utility_dec),
                     ("utility_enc", t_utility_enc), ("multilingual", t_multilingual), ("dynamics", t_dynamics),
                     ("placement", t_placement), ("hp", t_hp), ("cb", t_cb)]:
        try:
            (OUT / f"{name}.tex").write_text(fn(), encoding="utf-8")
            print("wrote", name)
        except Exception as e:
            print("skip", name, repr(e)[:160])


if __name__ == "__main__":
    main()
