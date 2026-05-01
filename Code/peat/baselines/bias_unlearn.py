"""
Baseline: BiasUnlearn — Dual-pathway unlearning.

# Reference: Liu et al., "BiasUnlearn: Mitigating Social Bias in Language
# Models via Dual-Pathway Unlearning", EMNLP 2025.
# arXiv: 2509.25673 | Used here for: gradient ascent on biased knowledge +
# gradient descent on retention objective.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from peat.data import StereoSetDataset, load_stereoset_pairs
from peat.eval import evaluate_full
from peat.models import get_spec, load_model
from peat.peat import attach_lora
from peat.utils import LOG_DIR, cleanup, get_autocast_dtype, set_seed, setup_logger

logger = setup_logger("peat.baselines.bias_unlearn", str(LOG_DIR / "baselines.log"))


def run(model_tag: str, seed: int = 42, device: str = "cuda") -> dict:
    """Run BiasUnlearn baseline: dual-pathway gradient ascent/descent."""
    logger.info(f"BiasUnlearn: {model_tag}, seed={seed}")
    set_seed(seed)
    spec = get_spec(model_tag)

    model, tokenizer, _ = load_model(model_tag, device=device)

    try:
        model = attach_lora(model, model_tag)
        model.train()

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-5, weight_decay=0.01,
        )
        cast_dtype = get_autocast_dtype()

        train_df, _ = load_stereoset_pairs(seed=seed)
        train_ds = StereoSetDataset(train_df)
        batch_size = 16 if spec.is_encoder else 4
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        for epoch in range(3):
            total_loss = 0.0
            n = 0
            for batch in tqdm(loader, desc=f"BiasUnlearn epoch {epoch+1}"):
                for i in range(len(batch["context"])):
                    ctx = batch["context"][i]
                    t_s = batch["t_s"][i]
                    t_a = batch["t_a"][i]

                    # Pathway 1: Gradient ASCENT on biased (stereotype) predictions
                    text_s = ctx.replace("BLANK", t_s)
                    inp_s = tokenizer(text_s, return_tensors="pt", truncation=True,
                                      max_length=512).to(device)
                    inp_s["labels"] = inp_s["input_ids"].clone()

                    with torch.amp.autocast("cuda", dtype=cast_dtype):
                        out_s = model(**inp_s)
                        loss_forget = -out_s.loss  # gradient ascent = negative loss

                    # Pathway 2: Gradient DESCENT on retention (anti-stereotype)
                    text_a = ctx.replace("BLANK", t_a)
                    inp_a = tokenizer(text_a, return_tensors="pt", truncation=True,
                                      max_length=512).to(device)
                    inp_a["labels"] = inp_a["input_ids"].clone()

                    with torch.amp.autocast("cuda", dtype=cast_dtype):
                        out_a = model(**inp_a)
                        loss_retain = out_a.loss  # standard gradient descent

                    # Combined loss
                    if loss_retain is not None and loss_forget is not None:
                        loss = 0.5 * loss_forget + 0.5 * loss_retain
                        if torch.isfinite(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            total_loss += loss.item()
                            n += 1

            logger.info(f"  BiasUnlearn epoch {epoch+1}: avg_loss={total_loss/max(n,1):.6f}")

        model.eval()
        metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device)
        metrics["method"] = "BiasUnlearn"
        metrics["seed"] = seed
        return metrics

    except Exception as e:
        logger.error(f"BiasUnlearn failed for {model_tag}: {e}")
        return {"method": "BiasUnlearn", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        cleanup(model)
