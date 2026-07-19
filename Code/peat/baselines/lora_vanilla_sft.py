"""
Baseline: LoRA + Vanilla SFT — Internal ablation.

Same LoRA config as PEAT (rank=4, alpha=8, last 2 layers), but trained with
standard masked-LM cross-entropy on the same StereoSet pairs.
Isolates "the loss is doing the work."

# Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
# ICLR 2022. arXiv: 2106.09685 | Code: https://github.com/microsoft/LoRA
# Used here for: ablation — same adapter, different loss.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from peat.data import StereoSetDataset, load_stereoset_pairs
from peat.eval import evaluate_full
from peat.models import get_spec, load_model
from peat.peat import attach_lora
from peat.utils import (
    LOG_DIR, RAW_DIR, STATE_DIR, SMOKE_TEST, cleanup, get_autocast_dtype, set_seed,
    setup_logger,
)

logger = setup_logger("peat.baselines.lora_sft", str(LOG_DIR / "baselines.log"))


def train_mlm_sft(model, tokenizer, model_tag: str, device: str = "cuda",
                  n_epochs: int = 5) -> None:
    """Train an attached-LoRA model with the standard MLM/CLM objective.

    Extracted so the WP-C placement factorial can reuse the identical MLM
    training loop with a different adapter placement. `model` must already be
    LoRA-wrapped (attach_lora); trains in place.
    """
    spec = get_spec(model_tag)
    model.train()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01,
    )

    train_df, _ = load_stereoset_pairs(seed=42)
    train_ds = StereoSetDataset(train_df)
    batch_size = 32 if spec.is_encoder else 8
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if SMOKE_TEST:
        n_epochs = 1
    total_steps = len(loader) * n_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    cast_dtype = get_autocast_dtype()
    grad_accum = 1 if spec.is_encoder else 4

    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n = 0
        for step, batch in enumerate(tqdm(loader, desc=f"SFT epoch {epoch}")):
            batch_size_actual = len(batch["context"])
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for i in range(batch_size_actual):
                ctx = batch["context"][i]
                # Train on both stereo and anti-stereo with standard MLM/CLM loss
                for filler in [batch["t_s"][i], batch["t_a"][i]]:
                    text = ctx.replace("BLANK", filler)
                    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                       max_length=512).to(device)
                    inputs["labels"] = inputs["input_ids"].clone()

                    with torch.amp.autocast("cuda", dtype=cast_dtype):
                        outputs = model(**inputs)
                        if outputs.loss is not None and torch.isfinite(outputs.loss):
                            batch_loss = batch_loss + outputs.loss

            batch_loss = batch_loss / (batch_size_actual * 2 * grad_accum)
            batch_loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += batch_loss.item()
            n += 1

        logger.info(f"  SFT epoch {epoch}: avg_loss={total_loss/max(n,1):.6f}")


def run(model_tag: str, seed: int = 42, device: str = "cuda",
        _model=None, _tokenizer=None) -> dict:
    """Run LoRA + vanilla SFT ablation."""
    logger.info(f"LoRA-Vanilla-SFT: {model_tag}, seed={seed}")
    set_seed(seed)
    spec = get_spec(model_tag)

    # Fine-tuning baseline: deepcopy so training never corrupts the shared base
    if _model is not None:
        import copy
        model = copy.deepcopy(_model)
        tokenizer = _tokenizer
    else:
        model, tokenizer, _ = load_model(model_tag, device=device)

    try:
        model = attach_lora(model, model_tag)
        train_mlm_sft(model, tokenizer, model_tag, device=device)

        # WP-E: persist the adapter so the extrinsic stage can re-load this
        # trained LoRA-Vanilla model without retraining (~6 MB per seed).
        from peft import get_peft_model_state_dict
        ckpt_dir = STATE_DIR / "lora_vanilla" / model_tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
            ckpt_dir / f"seed_{seed}.pt",
        )

        model.eval()
        _csv = RAW_DIR / "baselines" / "lora_vanilla_sft" / model_tag / f"seed_{seed}"
        _csv.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device, csv_dir=_csv)
        metrics["method"] = "LoRA-Vanilla-SFT"
        metrics["seed"] = seed
        return metrics

    except Exception as e:
        logger.error(f"LoRA-Vanilla-SFT failed for {model_tag}: {e}")
        return {"method": "LoRA-Vanilla-SFT", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        cleanup(model)
