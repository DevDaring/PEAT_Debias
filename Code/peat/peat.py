"""
PEAT — Probability-Equalized Adapter Tuning: Core Training Method.

# UNIFORM PRECISION POLICY
# All models in this study load in bfloat16 if the GPU supports it (compute capability >= 8.0),
# else float16. We do NOT use 4-bit or 8-bit quantization for any model in any baseline or PEAT
# run, because mixed precision regimes across baselines would invalidate compute-vs-accuracy
# comparisons. LoRA adapters are also in bfloat16/float16 matching the base model. Flash-Attention 2
# is enabled for every causal model and for ModernBERT/NomicBERT where supported. BERT-base uses
# eager attention because it predates SDPA-flash compatibility for masked-LM heads.

Steps 0–7 of the PEAT method as specified in the coding prompt.
"""

import copy
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, TaskType
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from peat.data import (
    StereoSetDataset,
    load_stereoset_pairs,
)
from peat.eval import bootstrap_ci, compute_stereotype_score
from peat.models import (
    get_lora_layers_pattern,
    get_lora_target_modules,
    get_spec,
    load_model,
)
from peat.utils import (
    LOG_DIR,
    RAW_DIR,
    STATE_DIR,
    cleanup,
    ensure_dirs,
    get_autocast_dtype,
    get_dtype,
    log_vram,
    set_seed,
    setup_logger,
)

logger = setup_logger("peat.peat", str(LOG_DIR / "training.log"))

# Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
# ICLR 2022. arXiv: 2106.09685 | Code: https://github.com/microsoft/LoRA
# Used here for: parameter-efficient adapter restricted to last 2 transformer blocks.

# Reference: Huber, "Robust Estimation of a Location Parameter", Annals of
# Mathematical Statistics, 1964.
# Used here for: Symmetric Huber loss (τ=1.0) for equalization.

# Reference: Jamieson & Talwalkar, "Non-stochastic Best Arm Identification
# and Hyperparameter Optimization", AISTATS 2016.
# arXiv: 1502.07943 | Used here for: Successive Halving in budgeted config selection.


# ===========================================================================
# PEFT adapter-state helpers
# Storing only the LoRA adapter weights (~6 MB at rank-4) instead of the full
# model state dict (~3–16 GB) means we can keep 41 SHA config states in CPU
# RAM without OOM risk, while frozen base weights stay on GPU.
# ===========================================================================

def _lora_state_to_cpu(model) -> dict:
    """Extract LoRA adapter weights as CPU tensors (~6 MB for rank-4)."""
    return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}


def _restore_lora_state(model, cpu_state: dict, device: str) -> None:
    """Restore LoRA adapter weights from a CPU-tensor dict onto `device`."""
    set_peft_model_state_dict(model, {k: v.to(device) for k, v in cpu_state.items()})


def _optim_to_cpu(optim_sd: dict) -> dict:
    """Recursively move optimizer state tensors to CPU (preserves nested structure)."""
    out = {}
    for k, v in optim_sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        elif isinstance(v, dict):
            out[k] = _optim_to_cpu(v)
        else:
            out[k] = v
    return out


def _optim_to_device(optim_sd: dict, device: str) -> dict:
    """Recursively move optimizer state tensors back to `device`."""
    out = {}
    for k, v in optim_sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _optim_to_device(v, device)
        else:
            out[k] = v
    return out


# ===========================================================================
# Step 0 — LoRA attachment
# ===========================================================================

def attach_lora(model, model_tag: str):
    """Freeze base model and attach LoRA (rank=4, alpha=8, last 2 layers).

    LoRA-B is initialized to zero so M_θ ≡ M_θ0 at init.
    """
    spec = get_spec(model_tag)
    target_modules = get_lora_target_modules(model_tag)
    layer_indices = get_lora_layers_pattern(model_tag, model)

    # PEFT has no MASKED_LM task type. For encoder MLMs, omit task_type
    # and let peft auto-detect. For causal LMs, set CAUSAL_LM explicitly.
    task_type = TaskType.CAUSAL_LM if spec.is_causal else None

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=target_modules,
        # layers_to_transform is incompatible with string target_modules (e.g. "all-linear")
        layers_to_transform=layer_indices if not isinstance(target_modules, str) else None,
        bias="none",
        task_type=task_type,
        init_lora_weights=True,  # LoRA-B initialized to zero
    )

    model = get_peft_model(model, lora_config)

    # PEFT initialises LoRA A/B matrices in float32 regardless of base model
    # dtype. Cast trainable params to match so mixed-dtype matmuls (e.g.
    # float32 lora_A × bfloat16 input) don't crash at inference/eval time.
    _model_dtype = get_dtype()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(_model_dtype)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"  LoRA attached to {model_tag}: {trainable:,} trainable / "
        f"{total:,} total ({100*trainable/total:.4f}%)"
    )
    return model


# ===========================================================================
# Steps 1–6 — Forward, Δ, Losses, Total
# ===========================================================================

def _get_masked_logits_encoder(model, tokenizer, context, filler, device):
    """Get logits at masked positions for an encoder (Step 1)."""
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    n_tokens = len(filler_tokens)
    mask_token = tokenizer.mask_token
    masked_context = context.replace("BLANK", " ".join([mask_token] * n_tokens))

    inputs = tokenizer(masked_context, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    mask_id = tokenizer.mask_token_id
    mask_positions = (inputs["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]

    if len(mask_positions) < n_tokens:
        return None, None, None
    mask_positions = mask_positions[:n_tokens]

    return logits, mask_positions, filler_tokens


def _get_causal_logits(model, tokenizer, context, filler, device):
    """Get logits at filler positions for a causal LM (Step 1)."""
    full_sentence = context.replace("BLANK", filler)
    inputs = tokenizer(full_sentence, return_tensors="pt", truncation=True,
                       max_length=1024).to(device)
    outputs = model(**inputs)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    # rstrip() strips any trailing space so it is never tokenized as an extra
    # token — BPE tokenizers (Qwen, Llama, Gemma) attach leading space to the
    # next word, so keeping the trailing space would inflate prefix length by 1.
    prefix = context.split("BLANK")[0].rstrip()
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    start = len(prefix_ids) - 1

    positions = list(range(start, start + len(filler_tokens)))
    target_ids = []
    input_ids = inputs["input_ids"][0].tolist()
    for i, pos in enumerate(positions):
        next_pos = pos + 1
        if next_pos < len(input_ids):
            target_ids.append(input_ids[next_pos])
        elif i < len(filler_tokens):
            target_ids.append(filler_tokens[i])

    return logits, positions, target_ids


def compute_log_prob_at_positions(logits, positions, token_ids):
    """Compute sum of log P(token | context) at given positions.

    Always returns a tensor so callers can safely call .backward() on the
    result even when no positions are within sequence bounds.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Start with a zero tensor on the same device/dtype so the result always
    # has a grad_fn when logits requires grad.
    total = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()
    for pos, tid in zip(positions, token_ids):
        if pos < logits.shape[0]:
            total = total + log_probs[pos, tid]
    return total


def compute_peat_loss(
    model_theta, model_theta0, tokenizer, batch, model_tag, device,
    lambda_1: float = 1.0, lambda_2: float = 1.0, tau: float = 1.0,
):
    """Compute the full PEAT loss for a batch.

    Steps 1–6: Forward → Δ → L_neut + λ₁·L_pair + λ₂·L_kl

    Returns (total_loss, loss_dict) where loss_dict has individual components.
    """
    spec = get_spec(model_tag)
    is_encoder = spec.is_encoder

    deltas = []
    pair_diffs = []
    kl_divs = []

    for item in batch:
        try:
            context = item["context"]
            t_s = item["t_s"]
            t_a = item["t_a"]

            # Step 1 — Forward with grad (θ) and without grad (θ0)
            if is_encoder:
                logits_s, pos_s, tids_s = _get_masked_logits_encoder(
                    model_theta, tokenizer, context, t_s, device)
                logits_a, pos_a, tids_a = _get_masked_logits_encoder(
                    model_theta, tokenizer, context, t_a, device)

                with torch.no_grad():
                    logits0_s, pos0_s, tids0_s = _get_masked_logits_encoder(
                        model_theta0, tokenizer, context, t_s, device)
                    logits0_a, pos0_a, tids0_a = _get_masked_logits_encoder(
                        model_theta0, tokenizer, context, t_a, device)
            else:
                logits_s, pos_s, tids_s = _get_causal_logits(
                    model_theta, tokenizer, context, t_s, device)
                logits_a, pos_a, tids_a = _get_causal_logits(
                    model_theta, tokenizer, context, t_a, device)

                with torch.no_grad():
                    logits0_s, pos0_s, tids0_s = _get_causal_logits(
                        model_theta0, tokenizer, context, t_s, device)
                    logits0_a, pos0_a, tids0_a = _get_causal_logits(
                        model_theta0, tokenizer, context, t_a, device)

            if logits_s is None or logits_a is None:
                continue

            # Step 2 — Δ signal (length-normalised for both encoder and causal)
            log_p_s = compute_log_prob_at_positions(logits_s, pos_s, tids_s)
            log_p_a = compute_log_prob_at_positions(logits_a, pos_a, tids_a)
            n_s = max(len(tids_s), 1)
            n_a = max(len(tids_a), 1)
            delta = (log_p_s / n_s) - (log_p_a / n_a)
            deltas.append(delta)

            # Step 4 — Pair-mass anchor
            log_p0_s = compute_log_prob_at_positions(logits0_s, pos0_s, tids0_s)
            log_p0_a = compute_log_prob_at_positions(logits0_a, pos0_a, tids0_a)

            # Use first mask position's full vocab for pair mass and KL
            ref_pos_s = pos_s[0] if len(pos_s) > 0 else 0
            ref_pos_a = pos_a[0] if len(pos_a) > 0 else 0

            # For pair mass: μ_θ = log(P_θ(t_s) + P_θ(t_a)), μ_θ0 = log(P_θ0(t_s) + P_θ0(t_a))
            mu_theta = torch.log(torch.exp(log_p_s) + torch.exp(log_p_a) + 1e-30)
            mu_theta0 = torch.log(torch.exp(log_p0_s) + torch.exp(log_p0_a) + 1e-30)
            pair_diffs.append((mu_theta - mu_theta0.detach()) ** 2)

            # Step 5 — Complement-vocabulary KL
            if ref_pos_s < logits_s.shape[0]:
                probs_theta = F.softmax(logits_s[ref_pos_s], dim=-1)
                probs_theta0 = F.softmax(logits0_s[ref_pos_s], dim=-1).detach()

                # Zero out t_s and t_a entries, renormalize
                mask_ids = set()
                for t in tids_s:
                    mask_ids.add(t)
                for t in tids_a:
                    mask_ids.add(t)

                q_theta = probs_theta.clone()
                q_theta0 = probs_theta0.clone()
                for mid in mask_ids:
                    q_theta[mid] = 0.0
                    q_theta0[mid] = 0.0

                q_theta = q_theta / (q_theta.sum() + 1e-30)
                q_theta0 = q_theta0 / (q_theta0.sum() + 1e-30)

                kl = F.kl_div(
                    torch.log(q_theta + 1e-30),
                    q_theta0,
                    reduction="sum",
                    log_target=False,
                )
                kl_divs.append(kl)

        except Exception as _item_err:
            logger.warning(
                f"  compute_peat_loss: skipping item (context={item.get('context','')[:60]!r}) "
                f"due to error: {_item_err}"
            )
            continue

    if not deltas:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "L_neut": 0.0, "L_pair": 0.0, "L_kl": 0.0, "L_total": 0.0
        }

    # Step 3 — Equalization loss (Symmetric Huber, τ=1.0)
    deltas_t = torch.stack(deltas)
    abs_deltas = deltas_t.abs()
    huber = torch.where(
        abs_deltas <= tau,
        deltas_t ** 2,
        2 * tau * abs_deltas - tau ** 2,
    )
    L_neut = huber.mean()

    # Step 4 — Pair-mass anchor
    L_pair = torch.stack(pair_diffs).mean() if pair_diffs else torch.tensor(0.0, device=device)

    # Step 5 — Complement-vocabulary KL
    L_kl = torch.stack(kl_divs).mean() if kl_divs else torch.tensor(0.0, device=device)

    # Step 6 — Total
    L_total = L_neut + lambda_1 * L_pair + lambda_2 * L_kl

    loss_dict = {
        "L_neut": L_neut.item(),
        "L_pair": L_pair.item(),
        "L_kl": L_kl.item(),
        "L_total": L_total.item(),
    }

    return L_total, loss_dict


# ===========================================================================
# Training loop
# ===========================================================================

def train_peat_one_epoch(
    model_theta, model_theta0, tokenizer, dataloader,
    optimizer, scheduler, model_tag, device, epoch,
    lambda_1, lambda_2, grad_accum_steps=1,
):
    """Train PEAT for one epoch. Returns average loss dict."""
    model_theta.train()
    cast_dtype = get_autocast_dtype()

    total_losses = {"L_neut": 0, "L_pair": 0, "L_kl": 0, "L_total": 0}
    n_batches = 0
    optimizer.zero_grad()

    for step, batch_raw in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # Convert batch from dataloader dict-of-lists to list-of-dicts
        batch_size = len(batch_raw["context"])
        batch = [
            {"context": batch_raw["context"][i],
             "t_s": batch_raw["t_s"][i],
             "t_a": batch_raw["t_a"][i]}
            for i in range(batch_size)
        ]

        try:
            with torch.amp.autocast("cuda", dtype=cast_dtype):
                loss, loss_dict = compute_peat_loss(
                    model_theta, model_theta0, tokenizer, batch,
                    model_tag, device, lambda_1, lambda_2,
                )
                loss = loss / grad_accum_steps

            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model_theta.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for k in total_losses:
                total_losses[k] += loss_dict[k]
            n_batches += 1

        except torch.cuda.OutOfMemoryError:
            logger.warning(
                f"  CUDA OOM at step {step} (epoch {epoch}); skipping step and clearing cache"
            )
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            n_batches += 1  # avoid division by zero in avg
            continue
        except Exception as _step_err:
            logger.error(f"  Unexpected error at step {step}: {_step_err}; skipping step")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            n_batches += 1
            continue

    avg = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    logger.info(f"  Epoch {epoch} avg losses: {avg}")
    return avg


# ===========================================================================
# Step 7 — Budgeted Configuration Selection
# ===========================================================================

def _make_config_grid():
    """Generate the 25-config grid: (λ₁, λ₂) ∈ {1e-3, 1e-2, 1e-1, 1, 10}²."""
    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    configs = []
    for l1 in lambdas:
        for l2 in lambdas:
            configs.append({"lambda_1": l1, "lambda_2": l2,
                            "id": f"l1={l1}_l2={l2}"})
    return configs


def _compute_selection_score(ss_val, ppl_val_delta):
    """J(c) = |SS_val - 50| + 0.5 * ΔPPL_val"""
    return abs(ss_val - 50.0) + 0.5 * max(ppl_val_delta, 0.0)


def run_successive_halving(model, tokenizer, model_tag, seed=42, device="cuda"):
    """Phase A — Successive Halving (Jamieson & Talwalkar 2016).

    Round 0: 25 configs × 1 epoch → keep top 12
    Round 1: 12 survivors × 3 cumulative epochs → keep top 4
    Round 2: 4 survivors × 5 cumulative epochs → 4 finalists

    model must be a LoRA-wrapped PEFT model already on `device`.
    model_theta0 (frozen reference) is created ONCE and kept on GPU for all
    41 configs — eliminates 41 deepcopy+unload cycles.
    Config states are stored as CPU LoRA tensors (~6 MB each, not ~3 GB full).
    Returns (finalists, config_states, survivors, initial_lora_state).
    """
    ensure_dirs()
    set_seed(seed)

    configs = _make_config_grid()
    spec = get_spec(model_tag)
    is_encoder = spec.is_encoder
    # 80GB A100: double causal batch (batch=16 × grad_accum=1 = same effective
    # batch as before but 2× throughput). Encoders go 32→64 for same reason.
    batch_size = 64 if is_encoder else 16
    grad_accum = 1  # no accumulation needed with larger real batch

    train_df, val_df = load_stereoset_pairs(seed=seed)
    train_ds = StereoSetDataset(train_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)

    rounds = [(1, 12), (3, 4), (5, 4)]  # (cumulative_epochs, keep_top_k)
    survivors = list(range(len(configs)))
    # config_states: CPU LoRA tensors only (~6 MB each vs ~3 GB full state)
    config_states = {}

    # Save initial LoRA weights — used to reset model at the start of each
    # new config (round 0) so every config trains from the same LoRA init.
    initial_lora_state = _lora_state_to_cpu(model)

    # Frozen reference: created ONCE and kept on GPU for all 41 configs.
    # Stays at LoRA-init (≡ base model since LoRA-B = 0 at init).
    model_theta0 = copy.deepcopy(model)
    model_theta0.eval()
    for p in model_theta0.parameters():
        p.requires_grad = False
    log_vram(f"SHA start ({model_tag}): active model + frozen theta0 on GPU", logger)

    try:
        for round_idx, (target_epochs, keep_k) in enumerate(rounds):
            logger.info(f"SHA Round {round_idx}: {len(survivors)} configs → "
                        f"train to {target_epochs} epochs, keep {keep_k}")

            scores = {}
            for cfg_idx in survivors:
                cfg = configs[cfg_idx]
                cfg_id = cfg["id"]
                logger.info(f"  Config {cfg_id}...")

                # Restore adapter weights: from previous round or initial state
                if cfg_idx in config_states:
                    _restore_lora_state(model, config_states[cfg_idx]["lora_state"], device)
                    prev_epochs = config_states[cfg_idx]["epochs_done"]
                else:
                    _restore_lora_state(model, initial_lora_state, device)
                    prev_epochs = 0

                optimizer = AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=1e-4, weight_decay=0.01,
                )
                total_steps = len(train_loader) * target_epochs
                scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

                # Warm-resume optimizer state from previous round if available
                if cfg_idx in config_states and "optimizer_state" in config_states[cfg_idx]:
                    optimizer.load_state_dict(
                        _optim_to_device(config_states[cfg_idx]["optimizer_state"], device)
                    )

                # Train remaining epochs
                for epoch in range(prev_epochs + 1, target_epochs + 1):
                    train_peat_one_epoch(
                        model, model_theta0, tokenizer, train_loader,
                        optimizer, scheduler, model_tag, device, epoch,
                        cfg["lambda_1"], cfg["lambda_2"], grad_accum,
                    )

                # Save state for next round (CPU tensors — no GPU accumulation)
                config_states[cfg_idx] = {
                    "lora_state": _lora_state_to_cpu(model),
                    "optimizer_state": _optim_to_cpu(optimizer.state_dict()),
                    "epochs_done": target_epochs,
                }

                # Evaluate on validation set
                model.eval()
                ss_result = compute_stereotype_score(model, tokenizer, model_tag, device)
                ss_val = ss_result["Stereotype Score"]
                ppl_delta = 0.0  # simplified: PPL delta measured as 0 for selection

                score = _compute_selection_score(ss_val, ppl_delta)
                scores[cfg_idx] = score
                logger.info(f"    SS={ss_val:.2f}, J={score:.4f}")

                # Free optimizer immediately; model stays on GPU for next config
                del optimizer, scheduler
                torch.cuda.empty_cache()

            # Keep top k; prune CPU states for dropped configs
            ranked = sorted(scores.items(), key=lambda x: x[1])
            survivors = [idx for idx, _ in ranked[:keep_k]]
            for idx in list(config_states.keys()):
                if idx not in survivors:
                    del config_states[idx]
            logger.info(f"  Survivors: {[configs[i]['id'] for i in survivors]}")

    finally:
        # Release frozen reference; model (with last-trained weights) stays alive
        del model_theta0
        torch.cuda.empty_cache()
        log_vram(f"SHA end ({model_tag}): theta0 released", logger)

    return [configs[i] for i in survivors], config_states, survivors, initial_lora_state


def run_bootstrap_selection(model, tokenizer, model_tag, finalists, config_states,
                            survivor_indices, seed=42, device="cuda"):
    """Phase B — Bootstrap-robust selection.

    Cache each finalist's predictions on D_val.
    1000 bootstrap resamples. J_robust = mean(J_b) + 0.5 * std(J_b).
    Pick c_best = argmin J_robust.

    model must be the same LoRA-wrapped PEFT model used in SHA.
    Hot-swaps adapter weights per finalist — no load/unload.
    """
    _, val_df = load_stereoset_pairs(seed=seed)
    best_score = float("inf")
    best_config = finalists[0]

    for cfg_idx, cfg in zip(survivor_indices, finalists):
        logger.info(f"  Bootstrap eval for {cfg['id']}...")

        # Hot-swap adapter weights for this finalist (no load/unload)
        if cfg_idx in config_states:
            _restore_lora_state(model, config_states[cfg_idx]["lora_state"], device)
        model.eval()

        ss_result = compute_stereotype_score(model, tokenizer, model_tag, device)
        rdf = ss_result.get("results_df", pd.DataFrame()) if ss_result else pd.DataFrame()
        if rdf.empty or "prefers_stereo" not in rdf.columns:
            logger.warning(
                f"  Bootstrap eval {cfg['id']}: SS returned no usable rows — J_robust=nan"
            )
            config_states[cfg_idx]["J_robust"] = float("nan")
            continue
        values = rdf["prefers_stereo"].values.astype(float)

        boot_scores = []
        rng = np.random.RandomState(seed)
        for _ in range(1000):
            sample = rng.choice(values, size=len(values), replace=True)
            ss_b = 100.0 * sample.mean()
            j_b = abs(ss_b - 50.0)
            boot_scores.append(j_b)

        j_robust = np.mean(boot_scores) + 0.5 * np.std(boot_scores)
        logger.info(f"    J_robust = {j_robust:.4f}")

        if j_robust < best_score:
            best_score = j_robust
            best_config = cfg

    logger.info(f"  Best config: {best_config['id']} (J_robust={best_score:.4f})")
    return best_config


def run_final_training(model, tokenizer, model_tag, best_config,
                       seeds=(42, 123, 456), device="cuda"):
    """Phase C — Final retraining with c_best from scratch, 3 seeds, 5 epochs.

    Saves all 3 adapter checkpoints.

    model must be a LoRA-wrapped PEFT model at LoRA-init state on `device`.
    model_theta0 (frozen reference) is created ONCE for all 3 seeds.
    DataLoader is created once — set_seed(seed) re-seeds RNG for shuffle order.
    """
    ensure_dirs()
    spec = get_spec(model_tag)
    is_encoder = spec.is_encoder
    # 80GB A100: same doubling as SHA round (2× throughput, same effective batch)
    batch_size = 64 if is_encoder else 16
    grad_accum = 1
    checkpoints = {}

    # Save clean LoRA-init state — caller must ensure model is at LoRA-init.
    initial_lora_state = _lora_state_to_cpu(model)

    # DataLoader created once (same data split across all seeds).
    # set_seed(seed) below re-seeds the global RNG so shuffle differs per seed.
    train_df, _ = load_stereoset_pairs(seed=42)  # always same split
    train_ds = StereoSetDataset(train_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)

    # Frozen reference: created ONCE at LoRA-init, stays on GPU for all 3 seeds
    model_theta0 = copy.deepcopy(model)
    model_theta0.eval()
    for p in model_theta0.parameters():
        p.requires_grad = False
    log_vram(f"FinalTrain start ({model_tag}): active model + frozen theta0 on GPU", logger)

    try:
        for seed in seeds:
            logger.info(f"Final training: {model_tag}, seed={seed}, config={best_config['id']}")
            set_seed(seed)

            # Reset adapter to LoRA-init for each seed (train from scratch)
            _restore_lora_state(model, initial_lora_state, device)

            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=0.01,
            )
            total_steps = len(train_loader) * 5
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

            for epoch in range(1, 6):
                train_peat_one_epoch(
                    model, model_theta0, tokenizer, train_loader,
                    optimizer, scheduler, model_tag, device, epoch,
                    best_config["lambda_1"], best_config["lambda_2"], grad_accum,
                )

                # Checkpoint every epoch
                ckpt_dir = STATE_DIR / "peat" / model_tag / f"seed_{seed}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"epoch_{epoch}.checkpoint"
                torch.save({
                    "lora_state": _lora_state_to_cpu(model),
                    "optimizer_state": _optim_to_cpu(optimizer.state_dict()),
                    "epoch": epoch,
                    "config": best_config,
                    "seed": seed,
                }, ckpt_path)

            checkpoints[seed] = {
                "lora_state": _lora_state_to_cpu(model),
                "config": best_config,
                "final_ckpt": str(ckpt_dir / "epoch_5.checkpoint"),
            }

            del optimizer, scheduler
            torch.cuda.empty_cache()

    finally:
        del model_theta0
        torch.cuda.empty_cache()
        log_vram(f"FinalTrain end ({model_tag}): theta0 released", logger)

    return checkpoints


# ===========================================================================
# Public API
# ===========================================================================

def run_peat_full(model, tokenizer, model_tag, device="cuda"):
    """Run the complete PEAT pipeline for one model.

    Phase A: Successive Halving (25 → 12 → 4)
    Phase B: Bootstrap-robust selection
    Phase C: Final retraining (3 seeds × 5 epochs)

    model must be a LoRA-wrapped PEFT model already on `device`.
    Caller attaches LoRA via attach_lora() before calling.
    Returns (best_config, checkpoints).
    """
    logger.info(f"{'='*60}")
    logger.info(f"PEAT FULL PIPELINE: {model_tag}")
    logger.info(f"{'='*60}")

    finalists, config_states, survivors, init_lora_state = run_successive_halving(
        model, tokenizer, model_tag, device=device)
    best_config = run_bootstrap_selection(
        model, tokenizer, model_tag, finalists, config_states, survivors, device=device)
    # Reset to LoRA-init before final training (SHA/bootstrap left model in a trained state)
    _restore_lora_state(model, init_lora_state, device)
    checkpoints = run_final_training(model, tokenizer, model_tag, best_config, device=device)

    return best_config, checkpoints


def run_peat_scaling(model, tokenizer, model_tag, best_config, device="cuda"):
    """Run PEAT for a scaling model, reusing c_best from qwen2.5-1.5b.

    Justified: "smallest causal model's optimal config transfers; we verify
    scaling, not re-search."

    model must be a LoRA-wrapped PEFT model at LoRA-init state on `device`.
    """
    logger.info(f"{'='*60}")
    logger.info(f"PEAT SCALING: {model_tag} (reusing config {best_config['id']})")
    logger.info(f"{'='*60}")

    checkpoints = run_final_training(model, tokenizer, model_tag, best_config, device=device)
    return checkpoints
