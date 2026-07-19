"""
PEAT — Intervention layer for inference-time / edit baselines (WP-A).

Reviewer #2 (APIN-D-26-06244) observed that Self-Debias, FairSteer, KnowBias,
and BiasEdit reported a stereotype score identical to Base on every model. The
cause was that the intervention was computed but never connected to the
log-probability scoring path: `evaluate_full` scored the unmodified model. This
module supplies the missing connection so that each intervention is active
during SS scoring, and a sanity probe that makes a silent no-op impossible.

Two mechanisms:
  * Forward-hook context managers for activation-space methods
    (FairSteer steering, KnowBias neuron damping, BiasEdit editor). The hooks
    persist through every forward pass inside `evaluate_full`, so the scored
    logits reflect the intervention.
  * A sentence-scorer override for Self-Debias, whose Schick et al. (2021)
    reweighting needs an auxiliary forward pass with a self-diagnosis prefix and
    therefore cannot be expressed as a plain activation hook.

# References:
#   Schick et al., "Self-Diagnosis and Self-Debiasing", TACL 2021 (arXiv:2103.00453)
#   Li et al., "FairSteer", ACL 2025 Findings (arXiv:2504.14492)
#   Pan et al., "KnowBias", 2026 (arXiv:2601.21864)
#   Xu et al., "BiasEdit", TrustNLP@NAACL 2025 (arXiv:2503.08588)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peat.models import get_last_n_layer_indices
from peat.utils import LOG_DIR, setup_logger

logger = setup_logger("peat.interventions", str(LOG_DIR / "baselines.log"))


# ---------------------------------------------------------------------------
# Locating transformer blocks across all six architectures
# ---------------------------------------------------------------------------
def find_transformer_layers(model) -> Optional[nn.ModuleList]:
    """Return the nn.ModuleList of transformer blocks for any supported model.

    Strategy: prefer the ModuleList whose length equals config.num_hidden_layers
    (handles Gemma3's nested text_config); otherwise the longest ModuleList.
    Works for BERT (bert.encoder.layer), ModernBERT/NomicBERT, and the causal
    decoders (model.layers).
    """
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
    num_layers = None
    for c in (cfg, text_cfg):
        if c is not None and getattr(c, "num_hidden_layers", None):
            num_layers = c.num_hidden_layers
            break

    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if num_layers and len(mod) == num_layers:
                return mod
            if best is None or len(mod) > len(best):
                best = mod
    return best


def _modify_block_output(output, fn):
    """Apply `fn` to the hidden-state tensor of a block's output (tuple or tensor)."""
    if isinstance(output, tuple):
        return (fn(output[0]),) + tuple(output[1:])
    return fn(output)


# ---------------------------------------------------------------------------
# FairSteer — project the bias direction out of the last-n block residuals
# ---------------------------------------------------------------------------
@contextmanager
def steering_intervention(model, steering_vec: torch.Tensor, alpha: float = 1.0,
                          n_last: int = 2):
    """Subtract the projection onto the (unit) bias direction at the last n blocks.

    `steering_vec` is mean(h_stereo - h_anti); removing its component reduces the
    stereo/anti-stereo asymmetry, i.e. steers the model toward neutral output.
    """
    layers = find_transformer_layers(model)
    if layers is None:
        logger.warning("steering_intervention: no transformer layers found; no-op")
        yield
        return
    v = steering_vec.detach().flatten()
    vhat = v / (v.norm() + 1e-8)
    idxs = range(max(0, len(layers) - n_last), len(layers))

    def _hook(module, args, output):
        def fn(h):
            w = vhat.to(dtype=h.dtype, device=h.device)
            proj = (h * w).sum(dim=-1, keepdim=True) * w
            return h - alpha * proj
        return _modify_block_output(output, fn)

    handles = [layers[i].register_forward_hook(_hook) for i in idxs]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


# ---------------------------------------------------------------------------
# KnowBias — damp the identified bias neurons at the last block output
# ---------------------------------------------------------------------------
@contextmanager
def neuron_damping_intervention(model, neuron_idx: torch.Tensor, gamma: float = 0.5):
    """Scale the identified bias-neuron dimensions by `gamma` (<1) at the last block."""
    layers = find_transformer_layers(model)
    if layers is None:
        logger.warning("neuron_damping_intervention: no transformer layers found; no-op")
        yield
        return
    idx = neuron_idx.detach().long().flatten()

    def _hook(module, args, output):
        def fn(h):
            h = h.clone()
            j = idx.to(h.device)
            j = j[j < h.shape[-1]]
            h[..., j] = h[..., j] * gamma
            return h
        return _modify_block_output(output, fn)

    hd = layers[-1].register_forward_hook(_hook)
    try:
        yield
    finally:
        hd.remove()


# ---------------------------------------------------------------------------
# BiasEdit — apply the trained editor network at the last block output
# ---------------------------------------------------------------------------
@contextmanager
def editor_intervention(model, editor: nn.Module):
    """Apply the trained BiasEditor to the last block's hidden states during scoring."""
    layers = find_transformer_layers(model)
    if layers is None:
        logger.warning("editor_intervention: no transformer layers found; no-op")
        yield
        return
    ed_dtype = next(editor.parameters()).dtype

    def _hook(module, args, output):
        def fn(h):
            return editor(h.to(ed_dtype)).to(h.dtype)
        return _modify_block_output(output, fn)

    hd = layers[-1].register_forward_hook(_hook)
    try:
        yield
    finally:
        hd.remove()


# ---------------------------------------------------------------------------
# Self-Debias — Schick et al. (2021) reweighted sentence scorer
# ---------------------------------------------------------------------------
_SELF_DIAGNOSIS_PREFIX = (
    "The following text discriminates against people based on their "
    "identity, gender, race, religion, age, disability, or orientation: "
)


def make_self_debias_scorer(alpha: float = 50.0, prefix: str = _SELF_DIAGNOSIS_PREFIX):
    """Return a sentence_scorer implementing Self-Debiasing during SS scoring.

    At every scored position the token distribution is reweighted
    ``p'(t) ∝ p(t) · exp(-alpha · max(0, p_prefix(t) - p(t)))`` so that tokens
    the self-diagnosis prefix boosts (i.e. biased continuations) are suppressed.
    The scored sentence's log-probability is recomputed under p'. Signature
    matches the `sentence_scorer` contract used by `compute_stereotype_score`.
    """
    @torch.no_grad()
    def _debias_logits(logits_normal, logits_prefixed):
        p = F.softmax(logits_normal, dim=-1)
        pb = F.softmax(logits_prefixed, dim=-1)
        delta = (pb - p).clamp(min=0.0)
        p_deb = p * torch.exp(-alpha * delta)
        p_deb = p_deb / (p_deb.sum(dim=-1, keepdim=True) + 1e-12)
        return torch.log(p_deb + 1e-12)

    @torch.no_grad()
    def scorer(model, tokenizer, sentence, diff_positions, is_encoder, device):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

        if is_encoder:
            tokens = tokenizer.encode(sentence, add_special_tokens=True)
            special = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
            diff_positions = diff_positions or []
            shared = [i for i in range(len(tokens))
                      if i not in diff_positions and tokens[i] not in special]
            if not shared:
                return 0.0
            mask_id = tokenizer.mask_token_id
            total = 0.0
            for pos in shared:
                orig = tokens[pos]
                masked = tokens.copy(); masked[pos] = mask_id
                ids = torch.tensor([masked], device=device)
                lg = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, pos]
                # prefixed variant: insert the self-diagnosis prefix AFTER [CLS]
                # so the special-token structure stays valid; the mask position
                # shifts by len(prefix) for every original position >= 1.
                pmasked = [masked[0]] + prefix_ids + masked[1:]
                pids = torch.tensor([pmasked], device=device)
                plg = model(input_ids=pids,
                            attention_mask=torch.ones_like(pids)).logits[0, pos + len(prefix_ids)]
                total += _debias_logits(lg, plg)[orig].item()
            return total

        # causal path
        ids = tokenizer(sentence, return_tensors="pt", truncation=True,
                        max_length=1024).to(device)["input_ids"]
        pref = torch.tensor([prefix_ids], device=device)
        pids = torch.cat([pref, ids], dim=1)
        logits = model(input_ids=ids).logits[0]                 # (L, V)
        plogits = model(input_ids=pids).logits[0, pref.shape[1]:]  # align to sentence tokens
        total = 0.0
        for t in range(ids.shape[1] - 1):
            dl = _debias_logits(logits[t], plogits[t])
            total += dl[ids[0, t + 1]].item()
        return total

    return scorer


# ---------------------------------------------------------------------------
# Sanity probe — a silent no-op intervention is now impossible
# ---------------------------------------------------------------------------
def assert_intervention_active(base_df, method_df, method_name: str,
                               tol: float = 1e-6) -> None:
    """Raise if the intervention left the scored per-pair margins unchanged.

    Compares the continuous margin ``score_more - score_less`` on a fixed probe
    (first N pairs) with and without the intervention — continuous, so it detects
    a connected-but-weak intervention even when no pair's discrete preference
    flips. Directly prevents recurrence of the Reviewer #2 defect (a method that
    silently scores the unmodified model and reports Base-identical SS).
    """
    import numpy as np
    b = (base_df["score_more"] - base_df["score_less"]).to_numpy(dtype=float)
    m = (method_df["score_more"] - method_df["score_less"]).to_numpy(dtype=float)
    n = min(len(b), len(m))
    if n == 0:
        raise RuntimeError(f"{method_name}: empty probe — cannot verify intervention")
    drift = float(np.mean(np.abs(m[:n] - b[:n])))
    if drift < tol:
        raise RuntimeError(
            f"{method_name}: scored margins are identical with and without the "
            f"intervention (mean drift {drift:.2e} < {tol:.0e}) — intervention is "
            f"NOT active in the scoring path. Refusing to report a Base-equal "
            f"number (WP-A guard)."
        )
    logger.info(f"{method_name}: probe active (mean margin drift = {drift:.4f} over {n} pairs)")
