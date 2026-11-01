"""
Revision-2 analysis: turns results/r2/cells/* into paper-ready tables
(results/r2/tables/*.csv) and figures (results/r2/figures/*.pdf).

    python3 -m peat.r2.analysis

Statistics
  * paired tests over CrowS-Pairs pairs: exact McNemar on the seed-majority
    preference of each pair and a paired permutation test on the per-pair
    seed-mean preference (10,000 resamples); Holm correction within a table;
  * bounds (Reviewer #4 c.2): Markov (Prop. 1), Cantelli (one-sided
    Chebyshev) and the two-sided band decomposition evaluated on the realised
    per-pair Delta; empirical-Bernstein half-width (Maurer & Pontil 2009) for
    the finite-sample gap between empirical and population SS.

# Reference: McNemar, Psychometrika 12(2), 1947.
# Reference: Holm, Scand. J. Statistics 6(2), 1979.
# Reference: Cantelli, "Sui confini della probabilita", Atti Congr. Int. Mat. 1928.
# Reference: Maurer & Pontil, "Empirical Bernstein Bounds and Sample-Variance
#   Penalization", COLT 2009. arXiv: 0907.3740
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from peat.r2.common import CELLS_DIR, R2_DIR
from peat.r2.plan import CORE, DECODERS, ENCODERS, SEEDS, _short

TAB = R2_DIR / "tables"
FIG = R2_DIR / "figures"
SHORT2TAG = {_short(m): m for m in ENCODERS + DECODERS}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def meta(cid):
    p = CELLS_DIR / cid / "meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m if m.get("status") == "done" else None


def summ(cid):
    m = meta(cid)
    return m["summary"] if m else None


def csv(cid, name):
    p = CELLS_DIR / cid / f"{name}.csv"
    return pd.read_csv(p) if p.exists() and meta(cid) else None


def seeds_of(prefix, seeds=SEEDS):
    return [f"{prefix}_s{s}" for s in seeds if meta(f"{prefix}_s{s}")]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: binary stereotype-preference vectors over the same pairs."""
    n01 = int(((a == 0) & (b == 1)).sum())  # b stereotypical, a not
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    p = binomtest(min(n01, n10), n, 0.5).pvalue if n > 0 else 1.0
    return {"a_debiased": n01, "a_worsened": n10, "p": p}


def perm_test(x: np.ndarray, y: np.ndarray, n=10000, seed=0) -> float:
    d = np.asarray(x, float) - np.asarray(y, float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, d.size))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def emp_bernstein(p_hat: float, n: int, delta: float = 0.05, rng_width: float = 1.0) -> float:
    """Two-sided empirical-Bernstein half-width for a mean of [0, rng] variables."""
    V = p_hat * (1 - p_hat) * rng_width ** 2
    L = math.log(4 / delta)
    return math.sqrt(2 * V * L / n) + 7 * rng_width * L / (3 * (n - 1))


def bound_rows(d: np.ndarray, eps_list=(0.1, 0.25, 0.5, 1.0), tau=1.0) -> list[dict]:
    d = np.asarray(d, float)
    a = np.abs(d)
    L = float(np.where(a <= tau, d * d, 2 * tau * a - tau * tau).mean())
    mu, sd = float(d.mean()), float(d.std())
    rows = []
    for e in eps_list:
        cant = sd ** 2 / (sd ** 2 + (e - mu) ** 2) if e > mu else 1.0
        bplus = float(((d > 0) & (d < e)).mean())
        bminus = float(((d > -e) & (d <= 0)).mean())
        rows.append({"eps": e, "L_neut": L, "mean": mu, "sd": sd,
                     "SS_eps_realised": 100 * float((d >= e).mean()),
                     "markov_bound": min(100.0, 100 * L / e ** 2),
                     "cantelli_bound": 100 * cant,
                     "tail_term": 50 * min(1.0, L / e ** 2),
                     "band_plus": 100 * bplus, "band_minus": 100 * bminus,
                     "band_imbalance_term": 50 * abs(bplus - bminus),
                     "SS": 100 * float((d > 0).mean())})
    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _pref_matrix(cids, col="prefers_stereo"):
    mats = [csv(c, "crows_en")[col].values for c in cids if csv(c, "crows_en") is not None]
    return np.vstack(mats) if mats else None


def table_reproduction():
    """Retrained Base/PEAT/LoRA vs. published SS (sanity) + fp32 robustness."""
    published = {("peat", "bert"): 58.04, ("peat", "modernbert"): 58.18, ("peat", "nomic"): 56.96,
                 ("peat", "qwen"): 53.42, ("peat", "gemma"): 55.84, ("peat", "llama"): 55.75,
                 ("lora", "bert"): 58.69, ("lora", "modernbert"): 57.45, ("lora", "nomic"): 56.98,
                 ("lora", "qwen"): 56.33, ("base", "bert"): 57.89, ("base", "modernbert"): 59.08,
                 ("base", "nomic"): 56.70, ("base", "qwen"): 54.51}
    rows = []
    for (meth, m), pub in sorted(published.items()):
        cids = [f"base_{m}"] if meth == "base" else seeds_of(f"{meth}_{m}")
        ss = [summ(c)["crows_en_ss"] for c in cids if summ(c)]
        s32 = [summ(c)["crows_en_ss_fp32"] for c in cids if summ(c)]
        ties = [summ(c)["crows_en_ties_orig"] for c in cids if summ(c)]
        if not ss:
            continue
        rows.append({"method": meth, "model": m, "n_seeds": len(ss), "published_ss": pub,
                     "r2_ss_mean": np.mean(ss), "r2_ss_sd": np.std(ss, ddof=1) if len(ss) > 1 else 0,
                     "r2_ss_fp32_mean": np.mean(s32), "ties_mean": np.mean(ties)})
    return pd.DataFrame(rows)


def table_peat_vs_lora_fp32():
    """Re-test the headline comparison on tie-free fp32 scores."""
    rows, ps = [], []
    for m in ["bert", "modernbert", "nomic", "qwen"]:
        for col in ("prefers_stereo", "prefers_stereo_fp32"):
            A, B = _pref_matrix(seeds_of(f"peat_{m}"), col), _pref_matrix(seeds_of(f"lora_{m}"), col)
            if A is None or B is None:
                continue
            a = (A.mean(0) > 0.5).astype(int)
            b = (B.mean(0) > 0.5).astype(int)
            r = mcnemar(a, b)
            r.update({"model": m, "scores": col, "ss_peat": 100 * A.mean(), "ss_lora": 100 * B.mean(),
                      "perm_p": perm_test(A.mean(0), B.mean(0))})
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        for col in df["scores"].unique():
            mask = df["scores"] == col
            df.loc[mask, "p_holm"] = holm(df.loc[mask, "p"].tolist())
    return df


def table_delta_sign():
    """Reviewer #5 w.1: sign and magnitude of the per-pair gap, Base vs LoRA vs PEAT."""
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        base = csv(f"base_{m}", "crows_en")
        for meth in ("base", "lora", "peat"):
            cids = [f"base_{m}"] if meth == "base" else seeds_of(f"{meth}_{m}")
            for src, (fname, col) in {"crows": ("crows_en", "delta_fp32"),
                                      "cloze_en": ("cloze_english", "delta"),
                                      "stereoset_val": ("delta_val", "delta")}.items():
                ds = [csv(c, fname) for c in cids]
                ds = [x for x in ds if x is not None]
                if not ds:
                    continue
                D = np.mean([x[col].values for x in ds], axis=0)
                if meth == "base":
                    D0 = D
                else:
                    D0 = ds[0]["delta0"].values if "delta0" in ds[0] else (
                        base[col].values if (base is not None and src == "crows") else None)
                eps = 0.5
                row = {"model": m, "method": meth, "format": src, "n": D.size,
                       "SS": 100 * (D > 0).mean(), "mean": D.mean(), "median": np.median(D),
                       "mean_abs": np.abs(D).mean(), "band_plus_0.5": 100 * ((D > 0) & (D < eps)).mean(),
                       "band_minus_0.5": 100 * ((D > -eps) & (D <= 0)).mean()}
                if D0 is not None and meth != "base" and len(D0) == len(D):
                    ch = D - D0
                    row.update({"shift_on_pos": ch[D0 > 0].mean(), "shift_on_neg": ch[D0 < 0].mean(),
                                "flips_to_anti": int(((D0 > 0) & (D <= 0)).sum()),
                                "flips_to_stereo": int(((D0 <= 0) & (D > 0)).sum())})
                rows.append(row)
    return pd.DataFrame(rows)


def table_bounds():
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        for meth in ("base", "peat"):
            cids = [f"base_{m}"] if meth == "base" else seeds_of(f"peat_{m}")[:1]
            for c in cids:
                for split in ("val", "train"):
                    t = csv(c, f"delta_{split}")
                    if t is None:
                        continue
                    for r in bound_rows(t["delta"].values):
                        r.update({"model": m, "method": meth, "split": split, "cell": c})
                        rows.append(r)
    for n, p in ((1508, 0.534),):
        rows.append({"model": "bernstein", "method": f"n={n}, p={p}", "split": "crows",
                     "eps": float("nan"), "SS": 100 * emp_bernstein(p, n)})
    return pd.DataFrame(rows)


def _seed_stats(cids, key):
    v = [summ(c).get(key) for c in cids if summ(c) and summ(c).get(key) is not None]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v)) if v else (np.nan, np.nan, 0)


def table_surrogates():
    rows = []
    for m in ["bert", "qwen"]:
        ref = _pref_matrix(seeds_of(f"peat_{m}"), "prefers_stereo_fp32")
        for kind in ["huber", "l2", "l1", "logcosh", "logistic", "hinge"]:
            pre = f"peat_{m}" if kind == "huber" else f"surr_{kind}_{m}"
            cids = seeds_of(pre)
            if not cids:
                continue
            row = {"model": m, "surrogate": kind, "n": len(cids)}
            for key in ("crows_en_ss", "crows_en_ss_fp32", "cloze_english_ss", "delta_val_ss",
                        "delta_val_median", "delta_val_mean_abs", "delta_val_mean"):
                mu, sd, _ = _seed_stats(cids, key)
                row[key] = mu
                row[key + "_sd"] = sd
            A = _pref_matrix(cids, "prefers_stereo_fp32")
            if kind != "huber" and A is not None and ref is not None:
                row["perm_p_vs_huber"] = perm_test(A.mean(0), ref.mean(0))
            rows.append(row)
    return pd.DataFrame(rows)


def table_utility():
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        for meth in ("base", "peat", "lora"):
            cids = [f"base_{m}"] if meth == "base" else seeds_of(f"{meth}_{m}")
            if not cids:
                continue
            keys = set().union(*[summ(c).keys() for c in cids if summ(c)])
            row = {"model": m, "method": meth, "n": len(cids)}
            for k in sorted(keys):
                if k.startswith(("wikitext_", "arc_", "hellaswag_", "piqa_", "winogrande_", "boolq_",
                                 "lambada_")) and not k.endswith(("_n", "_error", "_tokens")):
                    row[k], row[k + "_sd"], _ = _seed_stats(cids, k)
            rows.append(row)
    return pd.DataFrame(rows)


def table_multilingual_extrinsic():
    rows, tests = [], []
    for m in ["qwen", "gemma", "llama", "bert", "modernbert", "nomic"]:
        for meth in ("base", "peat", "lora"):
            cids = [f"base_{m}"] if meth == "base" else seeds_of(f"{meth}_{m}")
            if not cids:
                continue
            row = {"model": m, "method": meth, "n": len(cids)}
            for k in ("crows_fr_ss", "mcrows_hindi_ss", "mcrows_bengali_ss", "mcrows_english_ss",
                      "crows_fr_ss_fp32", "mcrows_hindi_ss_fp32", "mcrows_bengali_ss_fp32",
                      "cloze_english_ss", "cloze_hindi_ss", "cloze_bengali_ss",
                      "bbq_acc_ambig", "bbq_s_amb", "bold_regard_mean", "bold_gender_gap",
                      "bold_race_gap", "bold_religious_ideology_gap"):
                row[k], row[k + "_sd"], _ = _seed_stats(cids, k)
            rows.append(row)
        for bench in ("crows_fr", "mcrows_hindi", "mcrows_bengali", "mcrows_english"):
            b = csv(f"base_{m}", bench)
            ps = [csv(c, bench) for c in seeds_of(f"peat_{m}")]
            ps = [x for x in ps if x is not None]
            if b is None or not ps:
                continue
            P = np.vstack([x["prefers_stereo_fp32"].values for x in ps])
            r = mcnemar((P.mean(0) > 0.5).astype(int), b["prefers_stereo_fp32"].values)
            r.update({"model": m, "benchmark": bench, "ss_base": 100 * b["prefers_stereo_fp32"].mean(),
                      "ss_peat": 100 * P.mean(), "perm_p": perm_test(P.mean(0), b["prefers_stereo_fp32"].values)})
            tests.append(r)
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t["p"].tolist())
    return pd.DataFrame(rows), t


def table_loss_dynamics():
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        for c in seeds_of(f"peat_{m}"):
            st = csv(c, "train_steps")
            if st is None or not len(st):
                continue
            last = st[st["epoch"] == st["epoch"].max()]
            first = st.head(max(1, len(st) // 20))
            g = last.dropna(subset=["g_neut"])
            rows.append({"model": m, "cell": c,
                         "L_neut_start": first["L_neut"].mean(), "L_neut_end": last["L_neut"].mean(),
                         "L_pair_end": last["L_pair"].mean(), "L_kl_end": last["L_kl"].mean(),
                         "w_pair_end": last["w_pair"].mean(), "w_kl_end": last["w_kl"].mean(),
                         "g_neut_end": g["g_neut"].mean(), "g_pair_end": g["g_pair"].mean(),
                         "g_kl_end": g["g_kl"].mean(), "cos_neut_pair_end": g["cos_neut_pair"].mean(),
                         "cos_neut_kl_end": g["cos_neut_kl"].mean(), "cos_pair_kl_end": g["cos_pair_kl"].mean()})
    df = pd.DataFrame(rows)
    return df.groupby("model").mean(numeric_only=True).reset_index() if len(df) else df


def table_placement_hp():
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen"]:
        for pl in ["first2", "middle2", "last2", "all"]:
            pre = f"peat_{m}" if pl == "last2" else f"place_{pl}_{m}"
            seeds = SEEDS if pl == "last2" else [42, 123]
            cids = seeds_of(pre, seeds)
            if not cids:
                continue
            row = {"model": m, "placement": pl, "n": len(cids)}
            for k in ("crows_en_ss", "crows_en_ss_fp32", "cloze_english_ss", "delta_val_ss", "delta_val_mean_abs"):
                row[k], row[k + "_sd"], _ = _seed_stats(cids, k)
            rows.append(row)
    place = pd.DataFrame(rows)
    rows = []
    for m in ["bert", "qwen"]:
        for name in ["r2", "r4", "r8", "r16", "lr5e-5", "lr1e-4", "lr2e-4"]:
            pre = f"peat_{m}" if name in ("r4", "lr1e-4") else f"hp_{name}_{m}"
            cids = seeds_of(pre, SEEDS if pre.startswith("peat") else [42, 123])
            if not cids:
                continue
            row = {"model": m, "setting": name, "n": len(cids)}
            for k in ("crows_en_ss", "crows_en_ss_fp32", "cloze_english_ss", "delta_val_ss"):
                row[k], row[k + "_sd"], _ = _seed_stats(cids, k)
            rows.append(row)
    return place, pd.DataFrame(rows)


def table_cb():
    rows, tests = [], []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma"]:
        for meth, pre in (("peat", f"peat_{m}"), ("peat_cb", f"cb_{m}")):
            cids = seeds_of(pre)
            if not cids:
                continue
            row = {"model": m, "method": meth, "n": len(cids)}
            for k in ("crows_en_ss", "crows_en_ss_disability", "crows_en_ss_sexual-orientation",
                      "crows_en_ss_gender", "crows_en_ss_age", "crows_en_ss_physical-appearance",
                      "crows_en_ss_socioeconomic", "crows_en_ss_nationality"):
                row[k], row[k + "_sd"], _ = _seed_stats(cids, k)
            rows.append(row)
        A, B = seeds_of(f"cb_{m}"), seeds_of(f"peat_{m}")
        if A and B:
            ca, cb = csv(A[0], "crows_en"), None
            for cat in ("disability", "sexual-orientation"):
                mask = ca["bias_type"].values == cat
                PA = np.vstack([csv(c, "crows_en")["prefers_stereo"].values[mask] for c in A])
                PB = np.vstack([csv(c, "crows_en")["prefers_stereo"].values[mask] for c in B])
                r = mcnemar((PA.mean(0) > 0.5).astype(int), (PB.mean(0) > 0.5).astype(int))
                r.update({"model": m, "category": cat, "n_pairs": int(mask.sum()),
                          "ss_cb": 100 * PA.mean(), "ss_peat": 100 * PB.mean(),
                          "perm_p": perm_test(PA.mean(0), PB.mean(0))})
                tests.append(r)
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t["perm_p"].tolist())
    return pd.DataFrame(rows), t


def table_diag():
    rows = []
    for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"]:
        gp, pr = csv(f"diag_{m}", "grad_profile"), csv(f"diag_{m}", "probe_layers")
        if gp is None or pr is None:
            continue
        h = gp[gp["kind"] == "hidden"].sort_values("layer")
        g = h["grad_norm"].values
        bl = gp[gp["kind"] == "block_param"]
        bl = bl[bl["layer"] >= 0].sort_values("layer")
        share = (bl["grad_norm"] ** 2).values
        n = len(pr)
        rows.append({"model": m, "layers": n - 1,
                     "grad_param_share_last2": float(share[-2:].sum() / share.sum()) if share.size else np.nan,
                     "best_dir_layer": int(pr["direction_acc"].idxmax()),
                     "best_dir_acc": pr["direction_acc"].max(), "final_dir_acc": pr["direction_acc"].iloc[-1],
                     "best_r2_layer": int(pr["delta_r2"].idxmax()), "best_r2": pr["delta_r2"].max(),
                     "final_r2": pr["delta_r2"].iloc[-1],
                     "rel_depth_best_r2": int(pr["delta_r2"].idxmax()) / max(n - 1, 1)})
    return pd.DataFrame(rows)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    # Loss / gradient dynamics
    ms = [m for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"] if seeds_of(f"peat_{m}")]
    if ms:
        fig, axes = plt.subplots(2, len(ms), figsize=(3.0 * len(ms), 5.2), squeeze=False)
        for j, m in enumerate(ms):
            st = csv(seeds_of(f"peat_{m}")[0], "train_steps")
            ax = axes[0, j]
            for k, lab in (("L_neut", r"$\mathcal{L}_{neut}$"), ("w_pair", r"$\lambda_1\mathcal{L}_{pair}$"),
                           ("w_kl", r"$\lambda_2\mathcal{L}_{KL}$")):
                ax.plot(st["step"], st[k].rolling(10, min_periods=1).mean(), label=lab)
            ax.set_yscale("log"); ax.set_title(m); ax.set_xlabel("step")
            g = st.dropna(subset=["g_neut"])
            ax = axes[1, j]
            for k, lab in (("g_neut", "neut"), ("g_pair", "pair"), ("g_kl", "KL")):
                ax.plot(g["step"], g[k], label=lab)
            ax.set_yscale("log"); ax.set_xlabel("step")
        axes[0, 0].set_ylabel("loss term"); axes[1, 0].set_ylabel("gradient norm")
        axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "loss_dynamics.pdf"); plt.close(fig)
    # Layer diagnostics
    ms = [m for m in ["bert", "modernbert", "nomic", "qwen", "gemma", "llama"] if meta(f"diag_{m}")]
    if ms:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for m in ms:
            gp, pr = csv(f"diag_{m}", "grad_profile"), csv(f"diag_{m}", "probe_layers")
            h = gp[gp["kind"] == "hidden"].sort_values("layer")
            x = h["layer"].values / max(h["layer"].max(), 1)
            axes[0].plot(x, h["grad_norm"].values / h["grad_norm"].max(), label=m)
            xp = pr["layer"].values / max(pr["layer"].max(), 1)
            axes[1].plot(xp, pr["direction_acc"].values, label=m)
            axes[2].plot(xp, pr["delta_r2"].values, label=m)
        axes[0].set_title("equaliser gradient norm (rel.)"); axes[1].set_title("stereotype-direction accuracy")
        axes[2].set_title(r"$R^2$ of output $\Delta$")
        for ax in axes:
            ax.set_xlabel("relative depth")
        axes[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "layer_diagnostics.pdf"); plt.close(fig)
    # Delta distributions (Base vs PEAT vs LoRA) for Qwen and BERT
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, m in zip(axes, ["bert", "qwen"]):
        for meth in ("base", "lora", "peat"):
            cids = [f"base_{m}"] if meth == "base" else seeds_of(f"{meth}_{m}")[:1]
            t = csv(cids[0], "delta_val") if cids else None
            if t is not None:
                ax.hist(np.clip(t["delta"], -6, 6), bins=60, histtype="step", density=True, label=meth)
        ax.axvline(0, color="k", lw=0.6); ax.set_title(f"{m}: held-out StereoSet $\\Delta$"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "delta_distributions.pdf"); plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    out = {
        "reproduction": table_reproduction(),
        "peat_vs_lora_fp32": table_peat_vs_lora_fp32(),
        "delta_sign": table_delta_sign(),
        "bounds": table_bounds(),
        "surrogates": table_surrogates(),
        "utility": table_utility(),
        "loss_dynamics": table_loss_dynamics(),
        "diagnostics": table_diag(),
    }
    out["multilingual"], out["multilingual_tests"] = table_multilingual_extrinsic()
    out["placement"], out["hp"] = table_placement_hp()
    out["cb"], out["cb_tests"] = table_cb()
    for k, v in out.items():
        v.to_csv(TAB / f"{k}.csv", index=False)
        print(f"{k}: {len(v)} rows")
    figures()


if __name__ == "__main__":
    main()
