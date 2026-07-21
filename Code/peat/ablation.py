"""
PEAT — Revision ablations (WP-C) and PEAT-CB training (WP-G).

Three resumable experiment runners, wired as Stages 3b/3c in run_all.py:

  run_loss_ablation        — per-term loss ablation (Reviewers #2-W4, #3-3):
                             A1 = L_neut only, A2 = +pair anchor, A3 = +KL;
                             full PEAT is the existing Stage-1 run.
  run_placement_factorial  — {PEAT, MLM} × {first2, all} adapter placement on
                             one encoder + one decoder (Reviewers #1-W2, #3-3,
                             #4-4); (PEAT, last2) and (MLM, last2) already
                             exist as PEAT and LoRA-Vanilla-SFT.
  run_peat_cb              — PEAT trained on the coverage-balanced corpus
                             (StereoSet + cb_pairs) on one encoder + one
                             decoder (Reviewer #4-5).

Budgets follow Submission/proposed_improvement.md Section 5: 2 seeds per
ablation cell (main-run per-seed SD <= 0.4 quoted alongside), 3 seeds for
PEAT-CB. Each cell records SS (+ per-category) and utility (GLUE or PPL).
"""
from __future__ import annotations

import traceback

import torch

from peat.cb_pairs import load_cb_augmented_pairs
from peat.eval import compute_ss_with_ci, evaluate_glue, evaluate_wikitext_perplexity
from peat.models import get_spec, load_model
from peat.peat import (
    KNOWN_BEST_CONFIGS,
    _lora_state_to_cpu,
    _restore_lora_state,
    attach_lora,
    run_final_training,
)
from peat.utils import (
    LOG_DIR,
    RAW_DIR,
    SMOKE_TEST,
    cell_key,
    cleanup,
    is_cell_complete,
    mark_cell_complete,
    mark_cell_failed,
    set_seed,
    setup_logger,
)

logger = setup_logger("peat.ablation", str(LOG_DIR / "training.log"))

ABLATION_MODELS_LOSS = ["bert-base", "modernbert-base", "nomicbert", "qwen2.5-1.5b"]
FACTORIAL_MODELS = ["bert-base", "qwen2.5-1.5b"]
CB_MODELS = ["bert-base", "qwen2.5-1.5b"]

# Ablation cells run at 1 seed — the budget fallback pre-agreed in
# Submission/proposed_improvement.md ("thin replication, not coverage"):
# coverage (all loss terms, 4 models, both placements/objectives) is unchanged,
# every cell still scores all 1,508 pairs (per-instance bootstrap CIs), and the
# main runs' per-seed SD (<= 0.45) is quoted alongside the ablation table.
# Reviewer seed requests target the PEAT-vs-baseline comparisons (5/3 seeds,
# untouched), not the ablation grid.
ABL_SEEDS = [42]
CB_SEEDS = [42] if SMOKE_TEST else [42, 123, 456]

# Loss-term variants: modify (λ1, λ2) of the model's published best config.
# A1 tests the collapse hypothesis (equaliser alone), A2/A3 isolate each anchor.
LOSS_VARIANTS = {
    "A1_neut_only": lambda cfg: {"lambda_1": 0.0, "lambda_2": 0.0,
                                 "id": "abl_A1_neut_only"},
    "A2_neut_pair": lambda cfg: {"lambda_1": cfg["lambda_1"], "lambda_2": 0.0,
                                 "id": "abl_A2_neut_pair"},
    "A3_neut_kl":   lambda cfg: {"lambda_1": 0.0, "lambda_2": cfg["lambda_2"],
                                 "id": "abl_A3_neut_kl"},
}


def _light_eval(model, tokenizer, model_tag: str, device: str, csv_dir) -> dict:
    """SS (with per-category + CI) and utility only — no BBQ/choice (cost)."""
    spec = get_spec(model_tag)
    model.eval()
    csv_dir.mkdir(parents=True, exist_ok=True)
    ss = compute_ss_with_ci(model, tokenizer, model_tag, device=device, csv_dir=csv_dir)
    metrics = {
        "Stereotype Score": ss["Stereotype Score"],
        "ss_ci": ss.get("formatted", ""),
        "ss_per_category": ss.get("ss_per_category", {}),
    }
    if spec.is_encoder:
        glue = evaluate_glue(model, tokenizer, model_tag, device)
        metrics["GLUE Average"] = glue["average"]
    else:
        metrics["WikiText-103 Perplexity"] = evaluate_wikitext_perplexity(
            model, tokenizer, model_tag, device)
    return metrics


def _train_and_eval_config(model, tokenizer, model_tag, config, seeds, device,
                           ckpt_subdir, csv_root, init_state, state, cell_prefix):
    """Reset adapter → run_final_training(config) → per-seed light eval + cells."""
    _restore_lora_state(model, init_state, device)
    checkpoints = run_final_training(
        model, tokenizer, model_tag, config, seeds=tuple(seeds), device=device,
        ckpt_subdir=ckpt_subdir,
    )
    for seed in seeds:
        key = cell_key(cell_prefix, model_tag, seed, config["id"])
        if is_cell_complete(state, key):
            continue
        if seed in checkpoints:
            _restore_lora_state(model, checkpoints[seed]["lora_state"], device)
        model.eval()
        csv_dir = RAW_DIR / csv_root / model_tag / f"seed_{seed}"
        metrics = _light_eval(model, tokenizer, model_tag, device, csv_dir)
        metrics.update({"method": config["id"], "seed": seed, "model": model_tag,
                        "tracking": checkpoints.get(seed, {}).get("tracking", [])})
        mark_cell_complete(state, key, metrics)


# ---------------------------------------------------------------------------
# Stage 3b-i — loss-term ablation
# ---------------------------------------------------------------------------
def run_loss_ablation(state, device: str = "cuda") -> None:
    for model_tag in ABLATION_MODELS_LOSS:
        base_cfg = KNOWN_BEST_CONFIGS.get(model_tag)
        if base_cfg is None:
            logger.warning(f"loss-ablation: no known config for {model_tag}; skipping")
            continue
        variants = {name: fn(base_cfg) for name, fn in LOSS_VARIANTS.items()}
        pending = [
            (name, cfg) for name, cfg in variants.items()
            if not all(is_cell_complete(state, cell_key("ablation_loss", model_tag, s, cfg["id"]))
                       for s in ABL_SEEDS)
        ]
        if not pending:
            logger.info(f"loss-ablation: {model_tag} complete — skipping")
            continue

        logger.info(f"\n--- Loss ablation: {model_tag} ({len(pending)} variants) ---")
        model = None
        try:
            model, tokenizer, _ = load_model(model_tag, device=device)
            model = attach_lora(model, model_tag)
            init_state = _lora_state_to_cpu(model)
            for name, cfg in pending:
                try:
                    _train_and_eval_config(
                        model, tokenizer, model_tag, cfg, ABL_SEEDS, device,
                        ckpt_subdir=f"ablation_loss/{name}",
                        csv_root=f"ablation_loss/{name}",
                        init_state=init_state, state=state,
                        cell_prefix="ablation_loss",
                    )
                except Exception as e:
                    logger.error(f"loss-ablation {name}/{model_tag} FAILED: {e}")
                    logger.error(traceback.format_exc())
                    for s in ABL_SEEDS:
                        k = cell_key("ablation_loss", model_tag, s, cfg["id"])
                        if not is_cell_complete(state, k):
                            mark_cell_failed(state, k, str(e))
        except Exception as e:
            logger.error(f"loss-ablation load failed for {model_tag}: {e}")
            logger.error(traceback.format_exc())
        finally:
            if model is not None:
                cleanup(model)


# ---------------------------------------------------------------------------
# Stage 3b-ii — placement × objective factorial
# ---------------------------------------------------------------------------
def run_placement_factorial(state, device: str = "cuda") -> None:
    from peat.baselines.lora_vanilla_sft import train_mlm_sft

    for model_tag in FACTORIAL_MODELS:
        base_cfg = KNOWN_BEST_CONFIGS.get(model_tag)
        for placement in ["first2", "all"]:
            for objective in ["peat", "mlm"]:
                variant_id = f"fact_{objective}_{placement}"
                done = all(
                    is_cell_complete(state, cell_key("ablation_place", model_tag, s, variant_id))
                    for s in ABL_SEEDS
                )
                if done:
                    continue

                logger.info(f"\n--- Factorial: {model_tag} {objective}×{placement} ---")
                model = None
                try:
                    # Fresh load per cell: placement changes adapter structure.
                    model, tokenizer, _ = load_model(model_tag, device=device)
                    model = attach_lora(model, model_tag, placement=placement)
                    init_state = _lora_state_to_cpu(model)

                    if objective == "peat":
                        cfg = dict(base_cfg); cfg["id"] = variant_id
                        _train_and_eval_config(
                            model, tokenizer, model_tag, cfg, ABL_SEEDS, device,
                            ckpt_subdir=f"ablation_place/{variant_id}",
                            csv_root=f"ablation_place/{variant_id}",
                            init_state=init_state, state=state,
                            cell_prefix="ablation_place",
                        )
                    else:
                        for seed in ABL_SEEDS:
                            key = cell_key("ablation_place", model_tag, seed, variant_id)
                            if is_cell_complete(state, key):
                                continue
                            set_seed(seed)
                            _restore_lora_state(model, init_state, device)
                            train_mlm_sft(model, tokenizer, model_tag, device=device)
                            csv_dir = RAW_DIR / f"ablation_place/{variant_id}" / model_tag / f"seed_{seed}"
                            metrics = _light_eval(model, tokenizer, model_tag, device, csv_dir)
                            metrics.update({"method": variant_id, "seed": seed,
                                            "model": model_tag})
                            mark_cell_complete(state, key, metrics)
                except Exception as e:
                    logger.error(f"factorial {variant_id}/{model_tag} FAILED: {e}")
                    logger.error(traceback.format_exc())
                    for s in ABL_SEEDS:
                        k = cell_key("ablation_place", model_tag, s, variant_id)
                        if not is_cell_complete(state, k):
                            mark_cell_failed(state, k, str(e))
                finally:
                    if model is not None:
                        cleanup(model)


# ---------------------------------------------------------------------------
# Stage 3c — PEAT-CB (coverage-balanced training corpus)
# ---------------------------------------------------------------------------
def run_peat_cb(state, device: str = "cuda") -> None:
    for model_tag in CB_MODELS:
        base_cfg = KNOWN_BEST_CONFIGS.get(model_tag)
        if base_cfg is None:
            continue
        cfg = dict(base_cfg); cfg["id"] = "peat_cb"
        done = all(
            is_cell_complete(state, cell_key("peat_cb", model_tag, s, "peat_cb"))
            for s in CB_SEEDS
        )
        if done:
            logger.info(f"PEAT-CB: {model_tag} complete — skipping")
            continue

        logger.info(f"\n--- PEAT-CB: {model_tag} ---")
        model = None
        try:
            train_df = load_cb_augmented_pairs(seed=42)
            logger.info(f"  CB corpus: {len(train_df)} pairs "
                        f"(StereoSet + coverage augmentation)")
            model, tokenizer, _ = load_model(model_tag, device=device)
            model = attach_lora(model, model_tag)
            init_state = _lora_state_to_cpu(model)
            _restore_lora_state(model, init_state, device)
            checkpoints = run_final_training(
                model, tokenizer, model_tag, cfg, seeds=tuple(CB_SEEDS),
                device=device, train_df=train_df, ckpt_subdir="peat_cb",
            )
            for seed in CB_SEEDS:
                key = cell_key("peat_cb", model_tag, seed, "peat_cb")
                if is_cell_complete(state, key):
                    continue
                if seed in checkpoints:
                    _restore_lora_state(model, checkpoints[seed]["lora_state"], device)
                model.eval()
                csv_dir = RAW_DIR / "peat_cb" / model_tag / f"seed_{seed}"
                metrics = _light_eval(model, tokenizer, model_tag, device, csv_dir)
                metrics.update({"method": "PEAT-CB", "seed": seed, "model": model_tag,
                                "n_train_pairs": len(train_df)})
                mark_cell_complete(state, key, metrics)
        except Exception as e:
            logger.error(f"PEAT-CB {model_tag} FAILED: {e}")
            logger.error(traceback.format_exc())
            for s in CB_SEEDS:
                k = cell_key("peat_cb", model_tag, s, "peat_cb")
                if not is_cell_complete(state, k):
                    mark_cell_failed(state, k, str(e))
        finally:
            if model is not None:
                cleanup(model)
