#!/usr/bin/env python3
"""
PEAT — Single Entry Point Launcher.

Usage:
    python3 run_all.py

Orchestrates the full research pipeline:
  Stage 1: PEAT training on core models (successive halving → bootstrap → final)
  Stage 2: PEAT scaling on larger models (reuse c_best from qwen2.5-1.5b)
  Stage 3: Baselines (9 methods × core models × 3 seeds)
  Stage 4: Aggregation (5 tables)
  Stage 5: Figures (2 PDFs)

Resumes cleanly from interruption via state/run_state.json.

# UNIFORM PRECISION POLICY
# All models in this study load in bfloat16 if the GPU supports it (compute capability >= 8.0),
# else float16. We do NOT use 4-bit or 8-bit quantization for any model in any baseline or PEAT
# run, because mixed precision regimes across baselines would invalidate compute-vs-accuracy
# comparisons. LoRA adapters are also in bfloat16/float16 matching the base model. Flash-Attention 2
# is enabled for every causal model and for ModernBERT/NeoBERT where supported. BERT-base uses
# eager attention because it predates SDPA-flash compatibility for masked-LM heads.
"""

import copy
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peat.utils import (
    LOG_DIR,
    RAW_DIR,
    STATE_DIR,
    cell_key,
    cleanup,
    ensure_dirs,
    is_cell_complete,
    load_run_state,
    mark_cell_complete,
    mark_cell_failed,
    mark_cell_skipped,
    save_run_state,
    send_sms,
    set_seed,
    setup_logger,
)

logger = setup_logger("peat.launcher", str(LOG_DIR / "training.log"))


def main():
    ensure_dirs()
    logger.info("=" * 70)
    logger.info(f"PEAT LAUNCHER — {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 70)

    state = load_run_state()
    if "start_time" not in state:
        state["start_time"] = datetime.now(timezone.utc).isoformat()
        save_run_state(state)

    # ── Stage 0: Dry Run ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 0: DRY RUN")
    logger.info("=" * 70)

    dryrun_file = STATE_DIR / "dryrun_passed"
    if not is_cell_complete(state, "dryrun"):
        from peat.dryrun import run_dryrun
        ok = run_dryrun(skip_if_recent=True)
        if not ok:
            logger.error("Dry run FAILED. Fix issues and re-run.")
            sys.exit(1)
        mark_cell_complete(state, "dryrun")
    else:
        logger.info("Dry run already passed — skipping")

    # Verify dryrun_passed file exists
    if not dryrun_file.exists():
        logger.error("state/dryrun_passed not found. Run dry run first.")
        sys.exit(1)

    # ── Stage 0b: Dataset Preflight ────────────────────────────────────────
    if not is_cell_complete(state, "dataset_preflight"):
        from peat.data import validate_dataset_structure
        logger.info("\nDataset preflight check...")
        if not validate_dataset_structure():
            logger.error("Dataset preflight FAILED")
            sys.exit(1)
        mark_cell_complete(state, "dataset_preflight")

    # ── Stage 1: PEAT Training (Core Models) ──────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: PEAT TRAINING (CORE MODELS)")
    logger.info("=" * 70)

    from peat.models import CORE_MODELS, SCALING_MODELS, get_spec, load_model
    from peat.peat import attach_lora, run_peat_full, run_peat_scaling
    from peat.eval import evaluate_full

    best_configs = {}
    SEEDS = [42, 123, 456]

    for model_tag in CORE_MODELS:
        stage_key = cell_key("peat_training", model_tag, 0, "full_pipeline")

        if is_cell_complete(state, stage_key):
            logger.info(f"PEAT training for {model_tag} already complete — skipping")
            # Recover best_config from state
            cell_data = state.get("cells", {}).get(stage_key, {})
            bc = cell_data.get("metrics", {}).get("best_config", {})
            if bc:
                best_configs[model_tag] = bc
            continue

        try:
            logger.info(f"\n--- PEAT Training: {model_tag} ---")
            best_config, checkpoints = run_peat_full(model_tag, device="cuda")
            best_configs[model_tag] = best_config

            # Load model ONCE, hot-swap weights for each seed (avoids 2 extra
            # 16GB load/unload cycles per model on the 80GB A100)
            model, tokenizer, _ = load_model(model_tag, device="cuda")
            model = attach_lora(model, model_tag)
            model.eval()

            # Evaluate with each seed's checkpoint
            for seed in SEEDS:
                eval_key = cell_key("peat_eval", model_tag, seed)
                if is_cell_complete(state, eval_key):
                    continue

                if seed in checkpoints:
                    model.load_state_dict(checkpoints[seed]["model_state"])
                    model.eval()

                csv_dir = RAW_DIR / "peat" / model_tag / f"seed_{seed}"
                csv_dir.mkdir(parents=True, exist_ok=True)
                metrics = evaluate_full(model, tokenizer, model_tag,
                                        seeds=[seed], device="cuda", csv_dir=csv_dir)
                metrics["method"] = "PEAT"
                metrics["seed"] = seed
                metrics["best_config"] = best_config.get("id", "")
                mark_cell_complete(state, eval_key, metrics)

            cleanup(model)  # single unload after all 3 seeds

            mark_cell_complete(state, stage_key, {
                "model": model_tag,
                "best_config": best_config,
                "method": "PEAT",
            })

        except Exception as e:
            logger.error(f"PEAT training failed for {model_tag}: {e}")
            logger.error(traceback.format_exc())
            mark_cell_failed(state, stage_key, str(e))

    # ── Stage 2: PEAT Scaling ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: PEAT SCALING")
    logger.info("=" * 70)

    # Reuse c_best from qwen2.5-1.5b
    # Justified: "smallest causal model's optimal config transfers; we verify scaling, not re-search"
    scaling_config = best_configs.get("qwen2.5-1.5b")
    if scaling_config is None:
        logger.warning("No c_best from qwen2.5-1.5b — using default config for scaling")
        scaling_config = {"lambda_1": 0.1, "lambda_2": 0.1, "id": "l1=0.1_l2=0.1"}

    for model_tag in SCALING_MODELS:
        stage_key = cell_key("peat_scaling", model_tag, 0, "full_pipeline")

        if is_cell_complete(state, stage_key):
            logger.info(f"PEAT scaling for {model_tag} already complete — skipping")
            continue

        try:
            logger.info(f"\n--- PEAT Scaling: {model_tag} ---")
            checkpoints = run_peat_scaling(model_tag, scaling_config, device="cuda")

            # Load model ONCE, hot-swap weights for each seed
            model, tokenizer, _ = load_model(model_tag, device="cuda")
            model = attach_lora(model, model_tag)
            model.eval()

            for seed in SEEDS:
                eval_key = cell_key("peat_scaling_eval", model_tag, seed)
                if is_cell_complete(state, eval_key):
                    continue

                if seed in checkpoints:
                    model.load_state_dict(checkpoints[seed]["model_state"])
                    model.eval()

                csv_dir = RAW_DIR / "peat" / model_tag / f"seed_{seed}"
                csv_dir.mkdir(parents=True, exist_ok=True)
                metrics = evaluate_full(model, tokenizer, model_tag,
                                        seeds=[seed], device="cuda", csv_dir=csv_dir)
                metrics["method"] = "PEAT"
                metrics["seed"] = seed
                metrics["config_source"] = "qwen2.5-1.5b (transferred)"
                mark_cell_complete(state, eval_key, metrics)

            cleanup(model)  # single unload after all 3 seeds

            mark_cell_complete(state, stage_key, {
                "model": model_tag,
                "config_source": "qwen2.5-1.5b",
                "method": "PEAT",
            })

        except Exception as e:
            logger.error(f"PEAT scaling failed for {model_tag}: {e}")
            logger.error(traceback.format_exc())
            mark_cell_failed(state, stage_key, str(e))

    # ── Stage 3: Baselines ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: BASELINES")
    logger.info("=" * 70)

    from peat.baselines import BASELINE_REGISTRY

    for baseline_name, baseline_fn in BASELINE_REGISTRY.items():
        for model_tag in CORE_MODELS:
            for seed in SEEDS:
                key = cell_key("baseline", f"{baseline_name}_{model_tag}", seed)

                if is_cell_complete(state, key):
                    logger.info(f"  {baseline_name}/{model_tag}/seed{seed} — already done")
                    continue

                try:
                    logger.info(f"\n--- Baseline: {baseline_name} | {model_tag} | seed={seed} ---")
                    metrics = baseline_fn(model_tag, seed=seed, device="cuda")

                    if "status" in metrics and "skipped" in str(metrics.get("status", "")):
                        mark_cell_skipped(state, key, metrics["status"])
                        logger.warning(f"  {baseline_name}/{model_tag}: {metrics['status']}")
                    else:
                        mark_cell_complete(state, key, metrics)

                except Exception as e:
                    logger.error(f"  {baseline_name}/{model_tag} FAILED: {e}")
                    logger.error(traceback.format_exc())
                    mark_cell_failed(state, key, str(e))

    # ── Stage 4: Aggregation ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 4: AGGREGATION")
    logger.info("=" * 70)

    try:
        from peat.aggregation import build_all_tables
        state = load_run_state()  # reload latest
        tables = build_all_tables(state)
        mark_cell_complete(state, "aggregation")
        logger.info("Aggregation complete")
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        logger.error(traceback.format_exc())

    # ── Stage 5: Figures ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 5: FIGURES")
    logger.info("=" * 70)

    try:
        from peat.figures import build_all_figures
        build_all_figures()
        mark_cell_complete(state, "figures")
        logger.info("Figures complete")
    except Exception as e:
        logger.error(f"Figures failed: {e}")
        logger.error(traceback.format_exc())

    # ── Stage 6: Push Results to GitHub ──────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 6: PUSH RESULTS TO GITHUB")
    logger.info("=" * 70)

    try:
        import subprocess
        _repo_root = PROJECT_ROOT.parent  # D:\PhD\PEAT_Debias
        _res_dir = _repo_root / "results"
        if _res_dir.exists() and any(_res_dir.rglob("*")):
            subprocess.run(["git", "add", "results/"], check=True, cwd=_repo_root)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m",
                 f"Add results [{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}]"],
                check=True, cwd=_repo_root,
            )
            subprocess.run(["git", "push"], check=True, cwd=_repo_root)
            logger.info("Results pushed to GitHub.")
        else:
            logger.info("No results directory found — skipping GitHub push.")
    except Exception as e:
        logger.warning(f"GitHub push failed (non-fatal): {e}")

    # ── Final Summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    state = load_run_state()
    for key, cell in sorted(state.get("cells", {}).items()):
        status = cell.get("status", "unknown")
        logger.info(f"  {key}: {status}")

    completed = sum(1 for c in state.get("cells", {}).values() if c.get("status") == "completed")
    failed    = sum(1 for c in state.get("cells", {}).values() if c.get("status") == "failed")
    total = len(state.get("cells", {}))
    logger.info(f"Completed: {completed}/{total} cells")
    logger.info(f"Pipeline finished at {datetime.now(timezone.utc).isoformat()}")

    send_sms(
        f"PEAT pipeline DONE. {completed}/{total} cells completed, {failed} failed. "
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as _exc:
        _msg = str(_exc)[:120]
        send_sms(f"PEAT pipeline CRASHED: {_msg}")
        raise
