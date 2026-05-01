"""
PEAT — Publication Figures.

Generates:
  - fig1_ss_vs_compute.pdf:       SS vs compute scatter plot
  - fig2_per_category_heatmap.pdf: Per-category SS heatmap
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from peat.data import CROWS_BIAS_TYPES
from peat.utils import AGG_DIR, FIG_DIR, ensure_dirs, setup_logger

logger = setup_logger("peat.figures")

# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def build_fig1_ss_vs_compute():
    """Figure 1: Stereotype Score vs parameter count, colored by method."""
    ensure_dirs()
    table1_path = AGG_DIR / "table1_headline.csv"
    if not table1_path.exists():
        logger.warning("Table 1 not found — skipping Figure 1")
        return

    df = pd.read_csv(table1_path)
    if df.empty:
        logger.warning("Table 1 is empty — skipping Figure 1")
        return

    # Approximate parameter counts (millions)
    param_counts = {
        "bert-base": 110,
        "modernbert-base": 150,
        "nomicbert": 137,
        "qwen2.5-1.5b": 1500,
        "gemma-3-4b": 4000,
        "llama-3.1-8b": 8000,
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    methods = df["Method"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]

    for i, method in enumerate(methods):
        subset = df[df["Method"] == method]
        x = [param_counts.get(m, 100) for m in subset["Model"]]
        y = subset["Stereotype Score"].values
        ax.scatter(x, y, c=[colors[i]], marker=markers[i % len(markers)],
                   s=80, label=method, edgecolors="black", linewidths=0.5, alpha=0.85)

    # Optimal line at SS=50
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Optimal (50%)")

    ax.set_xlabel("Model Parameters (Millions)")
    ax.set_xscale("log")
    ax.set_ylabel("Stereotype Score (%)")
    ax.set_title("Stereotype Score vs. Model Scale")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    out = FIG_DIR / "fig1_ss_vs_compute.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Figure 1 saved: {out}")


def build_fig2_per_category_heatmap():
    """Figure 2: Per-category SS heatmap for PEAT across models."""
    ensure_dirs()
    table2_path = AGG_DIR / "table2_per_category.csv"
    if not table2_path.exists():
        logger.warning("Table 2 not found — skipping Figure 2")
        return

    df = pd.read_csv(table2_path)
    if df.empty:
        logger.warning("Table 2 is empty — skipping Figure 2")
        return

    pivot = df.pivot_table(index="Bias Type", columns="Model",
                           values="Stereotype Score", aggfunc="mean")

    # Reorder rows to match standard order
    ordered_types = [bt for bt in CROWS_BIAS_TYPES if bt in pivot.index]
    pivot = pivot.loc[ordered_types]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Center colormap at 50 (optimal)
    vmin = max(30, pivot.min().min() - 5)
    vmax = min(70, pivot.max().max() + 5)

    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
        center=50, vmin=vmin, vmax=vmax,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Stereotype Score (%)"},
    )

    ax.set_title("PEAT Per-Category Stereotype Score")
    ax.set_xlabel("Model")
    ax.set_ylabel("Bias Category")

    out = FIG_DIR / "fig2_per_category_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Figure 2 saved: {out}")


def build_all_figures():
    """Build all publication figures."""
    ensure_dirs()
    build_fig1_ss_vs_compute()
    build_fig2_per_category_heatmap()
