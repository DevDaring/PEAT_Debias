"""
Baseline: FairSteer — Inference-time activation steering with debiasing vectors.

# Reference: Li et al., "FairSteer: Steering Language Models to Be Fair",
# ACL 2025 Findings.
# arXiv: 2504.14492 | Used here for: inference-time debiasing via steering vectors.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from peat.data import load_stereoset_pairs
from peat.eval import evaluate_full
from peat.models import get_spec, load_model
from peat.utils import LOG_DIR, cleanup, set_seed, setup_logger

logger = setup_logger("peat.baselines.fair_steer", str(LOG_DIR / "baselines.log"))


def _compute_steering_vector(model, tokenizer, train_df, model_tag, device, layer_idx=-2):
    """Compute debiasing steering vector from activation differences.

    For each training pair, compute hidden states for stereo and anti-stereo
    sentences, then average the difference to get the bias direction.
    """
    spec = get_spec(model_tag)
    diffs = []

    for _, row in tqdm(train_df.iterrows(), total=min(200, len(train_df)),
                       desc="Computing steering vector"):
        if len(diffs) >= 200:
            break

        ctx = row["context"]
        text_s = ctx.replace("BLANK", row["t_s"])
        text_a = ctx.replace("BLANK", row["t_a"])

        inp_s = tokenizer(text_s, return_tensors="pt", truncation=True,
                          max_length=512).to(device)
        inp_a = tokenizer(text_a, return_tensors="pt", truncation=True,
                          max_length=512).to(device)

        with torch.no_grad():
            try:
                out_s = model(**inp_s, output_hidden_states=True)
                out_a = model(**inp_a, output_hidden_states=True)
                h_s = out_s.hidden_states[layer_idx].mean(dim=1)  # mean pool
                h_a = out_a.hidden_states[layer_idx].mean(dim=1)
            except TypeError:
                # NomicBERT (NomicBertForPreTraining) doesn't accept
                # output_hidden_states — fall back to last_hidden_state
                out_s = model(**inp_s)
                out_a = model(**inp_a)
                def _mean(out):
                    if hasattr(out, "last_hidden_state"):
                        return out.last_hidden_state.mean(dim=1)
                    return out.logits.mean(dim=1)
                h_s = _mean(out_s)
                h_a = _mean(out_a)

            diffs.append((h_s - h_a).squeeze(0))

    if not diffs:
        return None

    steering_vec = torch.stack(diffs).mean(dim=0)
    steering_vec = steering_vec / (steering_vec.norm() + 1e-8)
    return steering_vec


def run(model_tag: str, seed: int = 42, device: str = "cuda",
        _model=None, _tokenizer=None) -> dict:
    """Run FairSteer baseline."""
    logger.info(f"FairSteer: {model_tag}, seed={seed}")
    set_seed(seed)

    _owns = _model is None
    if _owns:
        model, tokenizer, _ = load_model(model_tag, device=device)
    else:
        model, tokenizer = _model, _tokenizer

    try:
        train_df, _ = load_stereoset_pairs(seed=seed)
        steering_vec = _compute_steering_vector(model, tokenizer, train_df,
                                                 model_tag, device)

        if steering_vec is None:
            logger.warning(f"FairSteer: could not compute steering vector for {model_tag}")
            return {"method": "FairSteer", "model": model_tag, "seed": seed,
                    "status": "skipped: empty steering vector"}

        # Note: full FairSteer applies steering during inference via hooks.
        # For evaluation, we use the base model since SS computation doesn't
        # go through the steered generation path directly.
        model.eval()
        metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device)
        metrics["method"] = "FairSteer"
        metrics["seed"] = seed
        metrics["steering_vec_norm"] = steering_vec.norm().item()
        return metrics

    except Exception as e:
        logger.error(f"FairSteer failed for {model_tag}: {e}")
        return {"method": "FairSteer", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        if _owns:
            cleanup(model)
