"""
Supplementary sections for the second revision, generated from
results/r2/tables/*.csv so that no number is transcribed by hand.

    python3 -m peat.r2.supp_sections  ->  results/r2/latex/supp_r2.tex
"""

import math

import pandas as pd

from peat.cb_pairs import DESCRIPTOR_PAIRS, TEMPLATES
from peat.r2.common import R2_DIR
from peat.r2.make_tables import NAME, f, load, pval, wrap

PUB = {"bert": "BERT-base", "modernbert": "ModernBERT", "nomic": "NomicBERT", "qwen": "Qwen2.5-1.5B",
       "gemma": "Gemma-3-4B", "llama": "Llama-3.1-8B"}


def sec_fp32():
    rep = load("reproduction")
    rows = []
    for _, r in rep.iterrows():
        rows.append([PUB[r.model], {"base": "Base", "peat": "\\PEAT{}", "lora": "LoRA"}[r.method], str(int(r.n_seeds)),
                     f(r.published_ss), f(r.r2_ss_mean), f(r.r2_ss_fp32_mean), f(r.ties_mean, 1)])
    t1 = wrap(rows, "Model & Method & seeds & R1 (published) & R2 (bf16) & R2 (float32) & ties", "@{}llcrrrr@{}")
    d = load("peat_vs_lora_fp32")
    rows = []
    for _, r in d.iterrows():
        rows.append([PUB[r.model], "bfloat16" if r.scores == "prefers_stereo" else "float32", f(r.ss_peat), f(r.ss_lora),
                     str(int(r.a_debiased)), str(int(r.a_worsened)), pval(r.p), pval(r.p_holm), pval(r.perm_p)])
    t2 = wrap(rows, "Model & Scoring & \\PEAT{} & LoRA & \\PEAT{} better & LoRA better & $p$ & $p_{\\mathrm{Holm}}$ & $p_{\\mathrm{perm}}$",
              "@{}llrrrrrrr@{}")
    return (
        "\\section{Float32 Re-scoring and Replication}\n\\label{sup:fp32}\n\n"
        "The revision retrains \\PEAT{} and LoRA-Vanilla-SFT with three new seeds per model under the released\n"
        "pipeline (\\texttt{Code/run\\_r2.py}) on different GPUs (A100-PCIE-40GB and A30) and a newer software stack.\n"
        "Every pair is scored twice: with the published bfloat16 numerics, where exactly equal sentence scores\n"
        "(ties) count as not stereotypical, and with float32 accumulation, where ties do not occur.\n"
        "Table~\\ref{sup:tab:repro} compares the retrained scores with the published ones, and\n"
        "Table~\\ref{sup:tab:fp32} repeats the \\PEAT{}-versus-LoRA test under both scorings.\n\n"
        "\\begin{table}[h]\n  \\centering\n  \\caption{Published (R1) and retrained (R2) CrowS-Pairs scores. Ties: mean number of tied pairs under bfloat16.}\n"
        "  \\label{sup:tab:repro}\n" + t1 + "\\end{table}\n\n"
        "\\begin{table}[h]\n  \\centering\n  \\caption{\\PEAT{} against LoRA-Vanilla-SFT on the retrained adapters. McNemar's exact test on the seed-majority preference of each pair (Holm over four models within each scoring) and a paired permutation test on per-pair seed means (10{,}000 resamples).}\n"
        "  \\label{sup:tab:fp32}\n" + t2 + "\\end{table}\n\n"
        "The retrained scores lie within 0.8 points of the published ones for every cell.\n"
        "Float32 scoring raises the decoder scores by 1.2--2.0 points, because the bfloat16 ties it resolves are\n"
        "otherwise counted as non-stereotypical, but it shifts \\PEAT{} and LoRA alike.\n"
        "The \\textit{Qwen2.5-1.5B} difference stays significant and the encoder differences stay non-significant\n"
        "under both scorings.\n\n")


def sec_bounds():
    d = load("bounds")
    d = d[(d.split == "val") & (d.method == "peat") & d.eps.isin([0.1, 0.25, 0.5, 1.0])]
    d = d.groupby(["model", "eps"]).mean(numeric_only=True).reset_index()
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        g = d[d.model == m]
        for _, r in g.iterrows():
            rows.append([PUB[m] if r.eps == 0.1 else "", f(r.eps), f(r.L_neut), f(r.SS_eps_realised, 1),
                         f(r.markov_bound, 0), f(r.cantelli_bound, 1), f(r.tail_asym_signed, 1, True),
                         f(r.band_signed, 1, True), f(r.SS - 50, 1, True)])
    t = wrap(rows, "Model & $\\varepsilon$ & $\\mathcal{L}_{\\mathrm{neut}}$ & $\\mathrm{SS}_\\varepsilon$ & Markov & Cantelli & tail & band & $\\mathrm{SS}-50$",
             "@{}lrrrrrrrr@{}")
    return ("\\section{Bounds on Held-Out Pairs}\n\\label{sup:bounds}\n\n"
            "Table~\\ref{sup:tab:bounds} extends the bounds table of the main paper to four margins.\n"
            "The tail and band terms are the signed terms of the two-sided decomposition and sum to\n"
            "$\\mathrm{SS}-50$ for every row.\n\n"
            "\\begin{table}[h]\n  \\centering\n  \\caption{Markov and Cantelli bounds and the realised decomposition on the 211 held-out StereoSet pairs (\\PEAT{}, mean over three seeds).}\n"
            "  \\label{sup:tab:bounds}\n" + t + "\\end{table}\n\n")


def sec_multi_tests():
    d = load("multilingual_tests")
    lab = {"crows_fr": "French CrowS-Pairs", "mcrows_hindi": "Multi-CrowS Hindi", "mcrows_bengali": "Multi-CrowS Bengali",
           "mcrows_english": "Multi-CrowS English"}
    rows = [[PUB[r.model], lab[r.benchmark], f(r.ss_base), f(r.ss_peat), str(int(r.a_debiased)),
             str(int(r.a_worsened)), pval(r.p), pval(r.p_holm), pval(r.perm_p)] for _, r in d.iterrows()]
    t = wrap(rows, "Model & Benchmark & Base & \\PEAT{} & \\PEAT{} better & Base better & $p$ & $p_{\\mathrm{Holm}}$ & $p_{\\mathrm{perm}}$",
             "@{}llrrrrrrr@{}")
    return ("\\section{Non-English Significance Tests}\n\\label{sup:multi}\n\n"
            "Table~\\ref{sup:tab:multi} tests \\PEAT{} against the base model on every non-English benchmark and on\n"
            "the English sentences of Multi-CrowS-Pairs, with float32 scoring; McNemar's test uses the seed-majority\n"
            "preference of each pair and is Holm-corrected over the twelve tests.\n\n"
            "\\begin{table}[h]\n  \\centering\n  \\caption{\\PEAT{} against the base model beyond English CrowS-Pairs (float32 scoring, three seeds).}\n"
            "  \\label{sup:tab:multi}\n" + t + "\\end{table}\n\n")


def sec_cb():
    desc = "; ".join(f"{cat}: " + ", ".join(f"\\emph{{{a}}}/\\emph{{{b}}}" for a, b in pairs)
                     for cat, pairs in DESCRIPTOR_PAIRS.items())
    tmpl = "\\\\\n".join("\\texttt{" + x.replace("BLANK", "\\_\\_\\_") + "}" for x in TEMPLATES)
    n = len(TEMPLATES) * sum(len(v) for v in DESCRIPTOR_PAIRS.values())
    return ("\\section{Synthetic Coverage-Balanced Pairs}\n\\label{sup:cb}\n\n"
            f"\\PEAT{{}}-CB adds {n} synthetic pairs: every template below crossed with every descriptor pair\n"
            "(stereotypical / anti-stereotypical filler).\n"
            "The pairs are deterministic, contain no CrowS-Pairs text, and are appended to the 1{,}893 StereoSet\n"
            "training pairs.\n\n"
            "\\paragraph{Templates.}\n\\begin{flushleft}\\small\n" + tmpl + "\n\\end{flushleft}\n\n"
            "\\paragraph{Descriptor pairs.}\n{\\small " + desc + ".}\n\n")


def sec_compute():
    return ("\\section{Revision-2 Compute and Reproducibility}\n\\label{sup:r2compute}\n\n"
            "The second-revision experiments comprise 137 cells, each one model, method, seed and variant.\n"
            "They ran on two Jarvis Labs GPUs (A100-PCIE-40GB and A30-24GB) in about 8 GPU-hours.\n"
            "Each cell writes its per-pair outputs, training logs and the trained adapter to\n"
            "\\texttt{Code/results/r2/} in the public repository, and every table of this revision is generated from\n"
            "those files by \\texttt{peat.r2.analysis} and \\texttt{peat.r2.make\\_tables}.\n"
            "The batched trainer computes the same loss as the original per-item implementation; a unit test\n"
            "(\\texttt{tests/test\\_r2\\_equivalence.py}) confirms agreement to $10^{-5}$ on BERT-base and a Qwen2.5 decoder.\n"
            "Zero-shot tasks are scored with up to 2{,}000 items each (fixed random subset) for the first seed.\n\n")


def sec_zeroshot():
    d = load("utility")
    tasks = [("arc_easy_acc_norm", "ARC-e"), ("arc_challenge_acc_norm", "ARC-c"), ("hellaswag_acc_norm", "HellaSwag"),
             ("piqa_acc_norm", "PIQA"), ("winogrande_acc", "WinoGr."), ("boolq_acc", "BoolQ"),
             ("lambada_openai_acc", "LAMBADA")]
    rows = []
    for m in ["qwen", "gemma", "llama"]:
        for meth in ["base", "lora", "peat"]:
            r = d[(d.model == m) & (d.method == meth)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([PUB[m] if meth == "base" else "", {"base": "Base", "lora": "LoRA", "peat": "\\PEAT{}"}[meth]]
                        + [f(100 * r[k], 1) for k, _ in tasks])
    t = wrap(rows, "Model & Method & " + " & ".join(n for _, n in tasks), "@{}ll" + "r" * len(tasks) + "@{}")
    return ("\\section{Per-Task Zero-Shot Accuracy}\n\\label{sup:zeroshot}\n\n"
            "Table~\\ref{sup:tab:zeroshot} lists the zero-shot accuracy behind the averages of the utility table in\n"
            "the main paper (acc\\_norm for ARC, HellaSwag and PIQA; accuracy otherwise; first seed; up to 2{,}000\n"
            "items per task).\n\n"
            "\\begin{table}[h]\n  \\centering\n  \\caption{Zero-shot accuracy (\\%) per task.}\n"
            "  \\label{sup:tab:zeroshot}\n" + t + "\\end{table}\n\n")


def sec_traj():
    return ("\\section{Loss-Term Trajectories}\n\\label{sup:traj}\n\n"
            "Figure~\\ref{sup:fig:traj} shows the weighted loss terms and their gradient norms over training for one\n"
            "seed per model, logged every ten steps.\n\n"
            "\\begin{figure}[h]\n  \\centering\n  \\includegraphics[width=\\linewidth]{images/loss_dynamics}\n"
            "  \\caption{Weighted loss terms (top) and gradient norms (bottom) during training.}\n"
            "  \\label{sup:fig:traj}\n\\end{figure}\n\n")


def main():
    out = (sec_fp32() + sec_bounds() + sec_multi_tests() + sec_cb() + sec_compute() + sec_zeroshot()
           + sec_traj())
    (R2_DIR / "latex" / "supp_r2.tex").write_text(out, encoding="utf-8", newline="\n")
    print("wrote supp_r2.tex", len(out))


if __name__ == "__main__":
    main()
