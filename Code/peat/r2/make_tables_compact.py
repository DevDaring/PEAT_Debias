"""
Compact versions of the revision tables for the main text (narrow enough for
the single-column Springer layout without scaling). Wider per-task and
per-seed versions stay in make_tables.py / supp_sections.py for the supplement.

    python3 -m peat.r2.make_tables_compact  ->  results/r2/latex/c_*.tex
"""

import numpy as np

from peat.r2.make_tables import METH, NAME, OUT, f, load, pm, pval, wrap


def c_surrogates():
    d = load("surrogates")
    lab = {"huber": "Huber (\\PEAT{})", "l2": "squared", "l1": "absolute", "logcosh": "sym.\\ logistic",
           "logistic": "one-sided logistic", "hinge": "one-sided hinge"}
    rows = []
    for m in ["bert", "qwen"]:
        g = d[d.model == m]
        if rows:
            rows.append("MID")
        for _, r in g.iterrows():
            rows.append([NAME[m] if r.surrogate == "huber" else "", lab[r.surrogate],
                         pm(r.crows_en_ss, r.crows_en_ss_sd), f(r.cloze_english_ss), f(r.delta_val_ss),
                         f(r.delta_val_mean_abs), "---" if r.surrogate == "huber" else pval(r.perm_p_vs_huber)])
    return wrap(rows, "Model & Surrogate & CrowS & Cloze & Held-out & $|\\Delta|$ & $p$", "@{}llccccc@{}")


def c_signheld():
    d = load("delta_sign")
    d = d[d.format == "stereoset_val"]
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        g = d[d.model == m]
        if rows:
            rows.append("MID")
        for meth in ["base", "lora", "peat"]:
            r = g[g.method == meth]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([NAME[m] if meth == "base" else "", METH[meth], f(r.SS), f(r["median"]), f(r.mean_abs),
                         "---" if meth == "base" else f(r.shift_on_pos, 2, True),
                         "---" if meth == "base" else f(r.shift_on_neg, 2, True)])
    return wrap(rows, "Model & Method & SS & med.\\ $\\Delta$ & $|\\Delta|$ & shift$^{+}$ & shift$^{-}$",
                "@{}llrrrrr@{}")


def c_dynamics():
    d = load("loss_dynamics").set_index("model")
    from peat.r2.make_tables import g3
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        r = d.loc[m]
        rows.append([NAME[m], f"{g3(r.L_neut_start)}$\\to${g3(r.L_neut_end)}", g3(r.w_pair_end), g3(r.w_kl_end),
                     g3(r.g_neut_end), g3(r.g_pair_end), g3(r.g_kl_end), f(r.cos_neut_pair_end, 2, True),
                     f(r.cos_neut_kl_end, 2, True)])
    hdr = ("Model & $\\Lneut$ & $\\lambda_1\\Lpair$ & $\\lambda_2\\Lkl$ & $\\|g_{\\mathrm{n}}\\|$ & "
           "$\\|g_{\\mathrm{p}}\\|$ & $\\|g_{\\mathrm{k}}\\|$ & $\\cos_{\\mathrm{np}}$ & $\\cos_{\\mathrm{nk}}$")
    return wrap(rows, hdr, "@{}lcrrrrrrr@{}", sep="3pt")


TASKS = ["arc_easy_acc_norm", "arc_challenge_acc_norm", "hellaswag_acc_norm", "piqa_acc_norm", "winogrande_acc",
         "boolq_acc", "lambada_openai_acc"]


def c_utility():
    d = load("utility")
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        g = d[d.model == m]
        if rows:
            rows.append("MID")
        for meth in ["base", "lora", "peat"]:
            r = g[g.method == meth]
            if r.empty:
                continue
            r = r.iloc[0]
            ppl = r.get("wikitext_ppl") if m in ("qwen", "gemma", "llama") else r.get("wikitext_pppl")
            zs = np.nanmean([r.get(k, np.nan) for k in TASKS]) if m in ("qwen", "gemma", "llama") else np.nan
            rows.append([NAME[m] if meth == "base" else "", METH[meth], f(ppl), f(r.get("wikitext_kl", np.nan), 4),
                         f(r.get("wikitext_tv", np.nan), 3), "---" if np.isnan(zs) else f(100 * zs, 1)])
    return wrap(rows, "Model & Method & (pseudo-)PPL & KL & TV & Zero-shot", "@{}llrrrr@{}")


def c_multilingual():
    d = load("multilingual")
    rows = []
    for m in ["qwen", "gemma", "llama"]:
        g = d[d.model == m]
        if rows:
            rows.append("MID")
        for meth in ["base", "lora", "peat"]:
            r = g[g.method == meth]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([NAME[m] if meth == "base" else "", METH[meth], f(r.crows_fr_ss_fp32), f(r.mcrows_hindi_ss_fp32),
                         f(r.mcrows_bengali_ss_fp32), f(r.cloze_hindi_ss), f(r.cloze_bengali_ss), f(r.cloze_english_ss)])
    return wrap(rows, "Model & Method & FR & HI & BN & HI$_{\\mathrm{c}}$ & BN$_{\\mathrm{c}}$ & EN$_{\\mathrm{c}}$",
                "@{}llrrrrrr@{}")


def c_boldbbq():
    d = load("bold_bbq_paired")
    lab = {"gender": "BOLD gender", "race": "BOLD race", "religious_ideology": "BOLD religion",
           "BBQ s_amb": "BBQ $s_{\\mathrm{amb}}$"}
    rows = []
    for _, r in d.iterrows():
        rows.append([NAME[r.model], METH[r.method], lab[r.domain], f(r.gap_base, 3), f(r["diff"], 3, True),
                     f"$[{r.ci_lo:+.3f},\\,{r.ci_hi:+.3f}]$"])
    return wrap(rows, "Model & Method & Measure & Base & Change & 95\\% CI", "@{}lllrrc@{}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("c_surrogates", c_surrogates), ("c_signheld", c_signheld), ("c_dynamics", c_dynamics),
                     ("c_utility", c_utility), ("c_multilingual", c_multilingual), ("c_boldbbq", c_boldbbq)]:
        (OUT / f"{name}.tex").write_text(fn(), encoding="utf-8", newline="\n")
        print("wrote", name)


if __name__ == "__main__":
    main()
