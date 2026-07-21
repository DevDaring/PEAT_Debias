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
# is enabled for every causal model and for ModernBERT/NomicBERT where supported. BERT-base uses
# eager attention because it predates SDPA-flash compatibility for masked-LM heads.
"""

import copy
import json
import signal
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
    log_vram,
    mark_cell_complete,
    mark_cell_failed,
    mark_cell_skipped,
    save_run_state,
    send_sms,
    set_seed,
    set_smoke_test,
    setup_logger,
)

logger = setup_logger("peat.launcher", str(LOG_DIR / "training.log"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PEAT full pipeline")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run pipeline on 4 random rows — quick integration/sanity check",
    )
    args, _ = parser.parse_known_args()

    if args.smoke_test:
        set_smoke_test(True, size=2)

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
    from peft import set_peft_model_state_dict

    # ── Model loader ───────────────────────────────────────────────────────
    # Load each model directly to GPU (device_map="cuda:0" in load_causal;
    # .to(device) in load_encoder).  No CPU intermediate — avoids:
    #   • meta-tensor errors in multimodal models (e.g. Gemma-3 vision tower)
    #   • the slow CPU→GPU copy that leaves the GPU idle
    #   • background-thread CUDA-fork DataLoader deadlocks
    def _get_model(tag: str, device: str = "cuda"):
        """Load model directly to GPU and return (model, tokenizer)."""
        m, tok, _ = load_model(tag, device=device)
        return m, tok

    best_configs = {}
    from peat.utils import SMOKE_TEST as _SMOKE
    # WP-B (APIN revision): 5 seeds for the headline PEAT vs LoRA-Vanilla-SFT
    # comparison (inside Reviewer #3's requested 5-10 range); other trained
    # baselines keep 3 seeds (their per-seed SD <= 0.07, so extra seeds add
    # nothing). Instance-level permutation/McNemar over 1,508 pairs carry the
    # statistical power. See Submission/proposed_improvement.md, Section 5.
    SEEDS = [42] if _SMOKE else [42, 123, 456, 789, 1011]
    SCALING_SEEDS = [42] if _SMOKE else [42, 123, 456]          # scaling models: 3 seeds
    FIVE_SEED_BASELINES = {"lora_vanilla_sft"}
    # Base and Self-Debias involve no seed-dependent computation (no training,
    # no sampled data): re-running them per seed reproduces identical scores —
    # the published supplement reports SD 0.00 for both. One evaluation each.
    DETERMINISTIC_BASELINES = {"base", "self_debias"}

    def seeds_for(baseline_name: str) -> list:
        """5 seeds for LoRA-Vanilla-SFT (paired against PEAT); 1 for
        deterministic methods (Base, Self-Debias); 3 for the rest."""
        if _SMOKE:
            return [42]
        if baseline_name in DETERMINISTIC_BASELINES:
            return [42]
        return SEEDS if baseline_name in FIVE_SEED_BASELINES else SEEDS[:3]

    for i, model_tag in enumerate(CORE_MODELS):

        stage_key = cell_key("peat_training", model_tag, 0, "full_pipeline")

        if is_cell_complete(state, stage_key):
            logger.info(f"PEAT training for {model_tag} already complete — skipping")
            # Recover best_config from state
            cell_data = state.get("cells", {}).get(stage_key, {})
            bc = cell_data.get("metrics", {}).get("best_config", {})
            if bc:
                best_configs[model_tag] = bc
            # Resume any eval seeds that were interrupted after training completed
            incomplete_seeds = [
                s for s in SEEDS
                if not is_cell_complete(state, cell_key("peat_eval", model_tag, s))
            ]
            if incomplete_seeds:
                import torch
                logger.info(
                    f"  Resuming {len(incomplete_seeds)} incomplete eval seed(s) "
                    f"from disk checkpoint: {incomplete_seeds}"
                )
                model, tokenizer = _get_model(model_tag, device="cuda")
                model = attach_lora(model, model_tag)
                model.eval()
                for seed in incomplete_seeds:
                    eval_key = cell_key("peat_eval", model_tag, seed)
                    ckpt_path = (
                        STATE_DIR / "peat" / model_tag / f"seed_{seed}" / "epoch_5.checkpoint"
                    )
                    if not ckpt_path.exists():
                        logger.warning(
                            f"  No checkpoint at {ckpt_path} — cannot resume seed {seed}"
                        )
                        continue
                    ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
                    set_peft_model_state_dict(
                        model,
                        {k: v.to("cuda") for k, v in ckpt["lora_state"].items()},
                    )
                    model.eval()
                    csv_dir = RAW_DIR / "peat" / model_tag / f"seed_{seed}"
                    csv_dir.mkdir(parents=True, exist_ok=True)
                    metrics = evaluate_full(
                        model, tokenizer, model_tag,
                        seeds=[seed], device="cuda", csv_dir=csv_dir,
                    )
                    metrics["method"] = "PEAT"
                    metrics["seed"] = seed
                    metrics["best_config"] = bc.get("id", "") if isinstance(bc, dict) else str(bc)
                    mark_cell_complete(state, eval_key, metrics)
                cleanup(model)
            continue

        try:
            logger.info(f"\n--- PEAT Training: {model_tag} ---")
            # Load ONCE: model stays on GPU across SHA (41 configs), bootstrap,
            # final training (3 seeds), and eval (3 seeds) — no redundant reloads.
            model, tokenizer = _get_model(model_tag, device="cuda")
            model = attach_lora(model, model_tag)
            best_config, checkpoints = run_peat_full(
                model, tokenizer, model_tag, device="cuda", seeds=tuple(SEEDS))
            best_configs[model_tag] = best_config

            model.eval()
            for seed in SEEDS:
                eval_key = cell_key("peat_eval", model_tag, seed)
                if is_cell_complete(state, eval_key):
                    continue

                if seed in checkpoints:
                    set_peft_model_state_dict(
                        model,
                        {k: v.to("cuda") for k, v in checkpoints[seed]["lora_state"].items()},
                    )
                    model.eval()

                csv_dir = RAW_DIR / "peat" / model_tag / f"seed_{seed}"
                csv_dir.mkdir(parents=True, exist_ok=True)
                metrics = evaluate_full(model, tokenizer, model_tag,
                                        seeds=[seed], device="cuda", csv_dir=csv_dir)
                metrics["method"] = "PEAT"
                metrics["seed"] = seed
                metrics["best_config"] = best_config.get("id", "")
                mark_cell_complete(state, eval_key, metrics)

            log_vram(f"before cleanup ({model_tag})", logger)
            cleanup(model)  # single unload after full pipeline + eval
            log_vram(f"after cleanup ({model_tag})", logger)

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

    for i, model_tag in enumerate(SCALING_MODELS):
        stage_key = cell_key("peat_scaling", model_tag, 0, "full_pipeline")

        if is_cell_complete(state, stage_key):
            logger.info(f"PEAT scaling for {model_tag} already complete — skipping")
            continue

        try:
            logger.info(f"\n--- PEAT Scaling: {model_tag} ---")
            # Load ONCE: model stays on GPU across final training (3 seeds) + eval (3 seeds)
            model, tokenizer = _get_model(model_tag, device="cuda")
            model = attach_lora(model, model_tag)
            checkpoints = run_peat_scaling(model, tokenizer, model_tag, scaling_config,
                                           device="cuda", seeds=tuple(SCALING_SEEDS))
            model.eval()
            for seed in SCALING_SEEDS:
                eval_key = cell_key("peat_scaling_eval", model_tag, seed)
                if is_cell_complete(state, eval_key):
                    continue

                if seed in checkpoints:
                    set_peft_model_state_dict(
                        model,
                        {k: v.to("cuda") for k, v in checkpoints[seed]["lora_state"].items()},
                    )
                    model.eval()

                csv_dir = RAW_DIR / "peat" / model_tag / f"seed_{seed}"
                csv_dir.mkdir(parents=True, exist_ok=True)
                metrics = evaluate_full(model, tokenizer, model_tag,
                                        seeds=[seed], device="cuda", csv_dir=csv_dir)
                metrics["method"] = "PEAT"
                metrics["seed"] = seed
                metrics["config_source"] = "qwen2.5-1.5b (transferred)"
                mark_cell_complete(state, eval_key, metrics)

            log_vram(f"before cleanup ({model_tag})", logger)
            cleanup(model)  # single unload after full pipeline + eval
            log_vram(f"after cleanup ({model_tag})", logger)

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

    for i, model_tag in enumerate(CORE_MODELS):
        # Skip model load entirely if every baseline for this model_tag is done
        all_done = all(
            is_cell_complete(state, cell_key("baseline", f"{bn}_{model_tag}", s))
            for bn in BASELINE_REGISTRY
            for s in seeds_for(bn)
        )
        if all_done:
            logger.info(f"All baselines for {model_tag} already complete — skipping")
            continue

        # Load base model ONCE (no LoRA). Inference-only baselines share it
        # read-only; fine-tuning baselines deepcopy it internally.
        logger.info(f"\n--- Loading {model_tag} for all baselines ---")
        try:
            _base_model, _base_tokenizer = _get_model(model_tag, device="cuda")
        except Exception as e:
            logger.error(f"  Failed to load {model_tag} for baselines: {e}")
            logger.error(traceback.format_exc())
            for baseline_name in BASELINE_REGISTRY:
                for seed in SEEDS:
                    key = cell_key("baseline", f"{baseline_name}_{model_tag}", seed)
                    if not is_cell_complete(state, key):
                        mark_cell_failed(state, key, f"model load failed: {e}")
            continue
        try:
            for baseline_name, baseline_fn in BASELINE_REGISTRY.items():
                for seed in seeds_for(baseline_name):
                    key = cell_key("baseline", f"{baseline_name}_{model_tag}", seed)

                    if is_cell_complete(state, key):
                        logger.info(f"  {baseline_name}/{model_tag}/seed{seed} — already done")
                        continue

                    try:
                        logger.info(f"\n--- Baseline: {baseline_name} | {model_tag} | seed={seed} ---")
                        metrics = baseline_fn(model_tag, seed=seed, device="cuda",
                                              _model=_base_model, _tokenizer=_base_tokenizer)
                        if "status" in metrics and "skipped" in str(metrics.get("status", "")):
                            mark_cell_skipped(state, key, metrics["status"])
                            logger.warning(f"  {baseline_name}/{model_tag}: {metrics['status']}")
                        else:
                            mark_cell_complete(state, key, metrics)

                    except Exception as e:
                        logger.error(f"  {baseline_name}/{model_tag} FAILED: {e}")
                        logger.error(traceback.format_exc())
                        mark_cell_failed(state, key, str(e))
        finally:
            cleanup(_base_model)

    # ── Stage 3b: Revision ablations (WP-C) ───────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3b: ABLATIONS (loss terms + placement factorial)")
    logger.info("=" * 70)
    try:
        from peat.ablation import run_loss_ablation, run_placement_factorial
        run_loss_ablation(state, device="cuda")
        run_placement_factorial(state, device="cuda")
    except Exception as e:
        logger.error(f"Ablation stage failed: {e}")
        logger.error(traceback.format_exc())

    # ── Stage 3c: PEAT-CB (WP-G coverage-balanced training) ───────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3c: PEAT-CB")
    logger.info("=" * 70)
    try:
        from peat.ablation import run_peat_cb
        run_peat_cb(state, device="cuda")
    except Exception as e:
        logger.error(f"PEAT-CB stage failed: {e}")
        logger.error(traceback.format_exc())

    # ── Stage 3d: Extrinsic evaluation (WP-E) ─────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3d: EXTRINSIC (Bias-in-Bios / HONEST / StereoSet-heldout)")
    logger.info("=" * 70)
    try:
        import torch as _torch
        from peat.extrinsic import evaluate_extrinsic
        from peat.peat import _restore_lora_state

        def _latest_ckpt(dir_path):
            cks = sorted(dir_path.glob("epoch_*.checkpoint"),
                         key=lambda p: int(p.stem.split("_")[1]))
            return cks[-1] if cks else None

        for model_tag in CORE_MODELS:
            EX_SEEDS = SEEDS[:3]   # extrinsic at 3 seeds (convention used by scaling/ablation)
            jobs = [("Base", 0)]
            jobs += [("PEAT", s) for s in EX_SEEDS]
            jobs += [("LoRA-Vanilla-SFT", s) for s in EX_SEEDS]
            if all(is_cell_complete(state, cell_key("extrinsic", f"{m}_{model_tag}", s))
                   for m, s in jobs):
                logger.info(f"Extrinsic for {model_tag} already complete — skipping")
                continue

            model, tokenizer = _get_model(model_tag, device="cuda")
            try:
                # Base first (before LoRA wrapping)
                key = cell_key("extrinsic", f"Base_{model_tag}", 0)
                if not is_cell_complete(state, key):
                    m = evaluate_extrinsic(model, tokenizer, model_tag, device="cuda")
                    m.update({"method": "Base", "model": model_tag, "seed": 0})
                    mark_cell_complete(state, key, m)

                # Wrap once; hot-swap adapters (identical rank-4/last-2 structure)
                model = attach_lora(model, model_tag)
                for seed in EX_SEEDS:
                    key = cell_key("extrinsic", f"PEAT_{model_tag}", seed)
                    if not is_cell_complete(state, key):
                        ck = _latest_ckpt(STATE_DIR / "peat" / model_tag / f"seed_{seed}")
                        if ck is None:
                            logger.warning(f"  extrinsic: no PEAT ckpt for {model_tag}/s{seed}")
                        else:
                            ckpt = _torch.load(str(ck), map_location="cuda", weights_only=False)
                            _restore_lora_state(model, ckpt["lora_state"], "cuda")
                            model.eval()
                            m = evaluate_extrinsic(model, tokenizer, model_tag, device="cuda")
                            m.update({"method": "PEAT", "model": model_tag, "seed": seed})
                            mark_cell_complete(state, key, m)

                    key = cell_key("extrinsic", f"LoRA-Vanilla-SFT_{model_tag}", seed)
                    if not is_cell_complete(state, key):
                        lv = STATE_DIR / "lora_vanilla" / model_tag / f"seed_{seed}.pt"
                        if not lv.exists():
                            logger.warning(f"  extrinsic: no LoRA-Vanilla ckpt for {model_tag}/s{seed}")
                        else:
                            sd = _torch.load(str(lv), map_location="cuda", weights_only=False)
                            _restore_lora_state(model, sd, "cuda")
                            model.eval()
                            m = evaluate_extrinsic(model, tokenizer, model_tag, device="cuda")
                            m.update({"method": "LoRA-Vanilla-SFT", "model": model_tag,
                                      "seed": seed})
                            mark_cell_complete(state, key, m)
            except Exception as e:
                logger.error(f"Extrinsic failed for {model_tag}: {e}")
                logger.error(traceback.format_exc())
            finally:
                cleanup(model)
    except Exception as e:
        logger.error(f"Extrinsic stage failed: {e}")
        logger.error(traceback.format_exc())

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
    import atexit

    # Notify on normal exit or unexpected crash (catches sys.exit too)
    atexit.register(send_sms, "PEAT pipeline exited (completed or stopped).")

    # Notify on SIGTERM (kill / container stop) and SIGINT (Ctrl+C)
    def _on_signal(signum, frame):
        send_sms("PEAT pipeline was interrupted (SIGTERM/SIGINT).")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        main()
    except Exception as _exc:
        _msg = str(_exc)[:120]
        send_sms(f"PEAT pipeline CRASHED: {_msg}")
        raise
