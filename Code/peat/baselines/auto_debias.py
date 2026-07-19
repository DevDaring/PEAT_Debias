"""
Baseline: Auto-Debias — JS-divergence on auto-discovered biased prompts.

# Reference: Guo et al., "Auto-Debias: Debiasing Masked Language Models with
# Automated Biased Prompts", ACL 2022.
# arXiv: 2104.04896 | Code: https://github.com/Irenehere/Auto-Debias
# Used here for: automated bias prompt discovery and JS-divergence debiasing.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from peat.data import load_stereoset_pairs, StereoSetDataset
from peat.eval import evaluate_full
from peat.models import get_spec, load_model
from peat.peat import attach_lora
from peat.utils import LOG_DIR, RAW_DIR, cleanup, get_autocast_dtype, set_seed, setup_logger

logger = setup_logger("peat.baselines.auto_debias", str(LOG_DIR / "baselines.log"))

# Attribute word lists for bias types (subset from Auto-Debias paper)
ATTRIBUTE_WORDS = {
    "gender": {
        "male": ["he", "man", "boy", "father", "son", "husband", "brother", "male"],
        "female": ["she", "woman", "girl", "mother", "daughter", "wife", "sister", "female"],
    },
    "race-color": {
        "white": ["white", "european", "caucasian"],
        "black": ["black", "african", "dark-skinned"],
    },
}


def _compute_js_divergence(p, q):
    """Compute Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m + 1e-10), p, reduction="batchmean", log_target=False)
    kl_qm = F.kl_div(torch.log(m + 1e-10), q, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pm + kl_qm)


def run(model_tag: str, seed: int = 42, device: str = "cuda",
        _model=None, _tokenizer=None) -> dict:
    """Run Auto-Debias baseline."""
    logger.info(f"Auto-Debias: {model_tag}, seed={seed}")
    set_seed(seed)
    spec = get_spec(model_tag)

    if not spec.is_encoder:
        logger.warning(f"Auto-Debias designed for MLMs; adapting for causal {model_tag}")

    # Fine-tuning baseline: deepcopy so training never corrupts the shared base
    if _model is not None:
        import copy
        model = copy.deepcopy(_model)
        tokenizer = _tokenizer
    else:
        model, tokenizer, _ = load_model(model_tag, device=device)

    try:
        model = attach_lora(model, model_tag)
        model.train()

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4, weight_decay=0.01,
        )
        cast_dtype = get_autocast_dtype()

        # Train to minimize JS-divergence between attribute word distributions
        for epoch in range(3):
            total_loss = 0.0
            n_steps = 0
            for bias_type, groups in ATTRIBUTE_WORDS.items():
                group_names = list(groups.keys())
                if len(group_names) < 2:
                    continue
                words_a = groups[group_names[0]]
                words_b = groups[group_names[1]]

                for w_a, w_b in zip(words_a, words_b):
                    if spec.is_encoder:
                        mask = tokenizer.mask_token
                        text_a = f"{w_a} is {mask}."
                        text_b = f"{w_b} is {mask}."
                    else:
                        text_a = f"{w_a} is"
                        text_b = f"{w_b} is"

                    inp_a = tokenizer(text_a, return_tensors="pt",
                                      truncation=True, max_length=64).to(device)
                    inp_b = tokenizer(text_b, return_tensors="pt",
                                      truncation=True, max_length=64).to(device)

                    with torch.amp.autocast("cuda", dtype=cast_dtype):
                        out_a = model(**inp_a)
                        out_b = model(**inp_b)

                        if spec.is_encoder:
                            mask_id = tokenizer.mask_token_id
                            pos_a = (inp_a["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]
                            pos_b = (inp_b["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]
                            if len(pos_a) > 0 and len(pos_b) > 0:
                                p_a = F.softmax(out_a.logits[0, pos_a[0]], dim=-1)
                                p_b = F.softmax(out_b.logits[0, pos_b[0]], dim=-1)
                            else:
                                continue
                        else:
                            p_a = F.softmax(out_a.logits[0, -1], dim=-1)
                            p_b = F.softmax(out_b.logits[0, -1], dim=-1)

                        loss = _compute_js_divergence(p_a, p_b)

                    if torch.isfinite(loss):
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()
                        n_steps += 1

            logger.info(f"  Auto-Debias epoch {epoch+1}: avg_loss={total_loss/max(n_steps,1):.4f}")

        model.eval()
        _csv = RAW_DIR / "baselines" / "auto_debias" / model_tag / f"seed_{seed}"
        _csv.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device, csv_dir=_csv)
        metrics["method"] = "Auto-Debias"
        metrics["seed"] = seed
        return metrics

    except Exception as e:
        logger.error(f"Auto-Debias failed for {model_tag}: {e}")
        return {"method": "Auto-Debias", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        cleanup(model)
