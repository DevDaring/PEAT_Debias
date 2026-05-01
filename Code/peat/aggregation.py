"""
PEAT — Results Aggregation.

Builds the five result tables from raw CSVs:
  - table1_headline.csv:     method × model headline SS
  - table2_per_category.csv: PEAT per-category SS breakdown
  - table3_ablations.csv:    PEAT vs LoRA-SFT ablation
  - table4_selector.csv:     SHA vs grid vs random config selection
  - table5_scaling.csv:      PEAT on scaling models (Gemma + Llama)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from peat.models import CORE_MODELS, MODEL_REGISTRY, SCALING_MODELS
from peat.data import CROWS_BIAS_TYPES
from peat.utils import AGG_DIR, RAW_DIR, STATE_DIR, ensure_dirs, setup_logger

logger = setup_logger("peat.aggregation")


def _load_cell_metrics(state: dict) -> list[dict]:
    """Extract all completed cell metrics from run_state."""
    rows = []
    for key, cell in state.get("cells", {}).items():
        if cell.get("status") == "completed" and cell.get("metrics"):
            m = cell["metrics"].copy()
            parts = key.split("|")
            if len(parts) >= 3:
                m["stage"] = parts[0]
                m["model"] = parts[1]
                m["seed"] = int(parts[2])
            rows.append(m)
    return rows


def build_table1_headline(state: dict) -> pd.DataFrame:
    """Table 1: Method × Model headline Stereotype Score."""
    rows = _load_cell_metrics(state)
    if not rows:
        logger.warning("No metrics found for Table 1")
        return pd.DataFrame()

    records = []
    for r in rows:
        method = r.get("method", "unknown")
        model = r.get("model", "unknown")
        ss = r.get("Stereotype Score", r.get("ss_mean", np.nan))
        ci = r.get("ss_ci", "")
        records.append({
            "Method": method,
            "Model": model,
            "Stereotype Score": ss,
            "95% CI": ci,
        })

    df = pd.DataFrame(records)
    out = AGG_DIR / "table1_headline.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Table 1 saved: {out} ({len(df)} rows)")
    return df


def build_table2_per_category(state: dict) -> pd.DataFrame:
    """Table 2: PEAT per-category SS breakdown across all 9 bias types."""
    rows = _load_cell_metrics(state)
    peat_rows = [r for r in rows if r.get("method") == "PEAT"]

    records = []
    for r in peat_rows:
        model = r.get("model", "unknown")
        per_cat = r.get("ss_per_category", {})
        for bt in CROWS_BIAS_TYPES:
            records.append({
                "Model": model,
                "Bias Type": bt,
                "Stereotype Score": per_cat.get(bt, np.nan),
            })

    df = pd.DataFrame(records)
    out = AGG_DIR / "table2_per_category.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Table 2 saved: {out} ({len(df)} rows)")
    return df


def build_table3_ablations(state: dict) -> pd.DataFrame:
    """Table 3: PEAT vs LoRA-Vanilla-SFT ablation."""
    rows = _load_cell_metrics(state)
    ablation_methods = {"PEAT", "LoRA-Vanilla-SFT"}
    ablation_rows = [r for r in rows if r.get("method") in ablation_methods]

    records = []
    for r in ablation_rows:
        records.append({
            "Method": r.get("method"),
            "Model": r.get("model"),
            "Stereotype Score": r.get("Stereotype Score", np.nan),
            "GLUE Average": r.get("GLUE Average", np.nan),
            "WikiText-103 Perplexity": r.get("WikiText-103 Perplexity", np.nan),
        })

    df = pd.DataFrame(records)
    out = AGG_DIR / "table3_ablations.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Table 3 saved: {out} ({len(df)} rows)")
    return df


def build_table4_selector(state: dict) -> pd.DataFrame:
    """Table 4: SHA vs grid vs random config selection comparison."""
    # This table compares the selected config quality across selection methods
    records = []
    for key, cell in state.get("cells", {}).items():
        if "peat_training" in key and cell.get("metrics"):
            m = cell["metrics"]
            records.append({
                "Model": m.get("model", ""),
                "Selection Method": "Successive Halving + Bootstrap",
                "Best Config": m.get("best_config", ""),
                "Stereotype Score": m.get("Stereotype Score", np.nan),
            })

    df = pd.DataFrame(records)
    out = AGG_DIR / "table4_selector.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Table 4 saved: {out} ({len(df)} rows)")
    return df


def build_table5_scaling(state: dict) -> pd.DataFrame:
    """Table 5: PEAT on scaling models (Gemma + Llama)."""
    rows = _load_cell_metrics(state)
    scaling_rows = [
        r for r in rows
        if r.get("model") in SCALING_MODELS and r.get("method") == "PEAT"
    ]

    records = []
    for r in scaling_rows:
        records.append({
            "Model": r.get("model"),
            "Stereotype Score": r.get("Stereotype Score", np.nan),
            "95% CI": r.get("ss_ci", ""),
            "WikiText-103 Perplexity": r.get("WikiText-103 Perplexity", np.nan),
            "Config Source": "qwen2.5-1.5b (transferred)",
        })

    df = pd.DataFrame(records)
    out = AGG_DIR / "table5_scaling.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Table 5 saved: {out} ({len(df)} rows)")
    return df


def build_all_tables(state: dict) -> dict[str, pd.DataFrame]:
    """Build all five aggregated tables."""
    ensure_dirs()
    return {
        "table1": build_table1_headline(state),
        "table2": build_table2_per_category(state),
        "table3": build_table3_ablations(state),
        "table4": build_table4_selector(state),
        "table5": build_table5_scaling(state),
    }
