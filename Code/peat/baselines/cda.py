"""
Baseline: CDA — Counterfactual Data Augmentation.

# Reference: Zmigrod et al., "Counterfactual Data Augmentation for Mitigating
# Gender Stereotypes in Languages with Rich Morphology", ACL 2019.
# arXiv: 1906.04571 | Code: https://github.com/McGill-NLP/bias-bench
# Used here for: re-fine-tune base model on attribute-swapped corpus.
"""

import copy

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from peat.data import StereoSetDataset, load_stereoset_pairs
from peat.eval import evaluate_full
from peat.models import get_spec, load_model
from peat.utils import LOG_DIR, RAW_DIR, cleanup, get_autocast_dtype, set_seed, setup_logger

logger = setup_logger("peat.baselines.cda", str(LOG_DIR / "baselines.log"))


def _create_augmented_data(train_df):
    """Create counterfactual augmented data by swapping stereotype/anti-stereotype."""
    augmented = train_df.copy()
    swapped = train_df.copy()
    swapped["t_s"], swapped["t_a"] = train_df["t_a"].values, train_df["t_s"].values
    swapped["stereotype_sentence"], swapped["anti_stereotype_sentence"] = (
        train_df["anti_stereotype_sentence"].values,
        train_df["stereotype_sentence"].values,
    )
    return StereoSetDataset(augmented), StereoSetDataset(swapped)


def run(model_tag: str, seed: int = 42, device: str = "cuda",
        _model=None, _tokenizer=None) -> dict:
    """Run CDA baseline: fine-tune on attribute-swapped corpus."""
    logger.info(f"CDA: {model_tag}, seed={seed}")
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
        train_df, _ = load_stereoset_pairs(seed=seed)
        orig_ds, swapped_ds = _create_augmented_data(train_df)

        # Fine-tune on combined original + swapped data
        batch_size = 16 if spec.is_encoder else 4
        orig_loader = DataLoader(orig_ds, batch_size=batch_size, shuffle=True)
        swap_loader = DataLoader(swapped_ds, batch_size=batch_size, shuffle=True)

        # Only fine-tune last 2 layers to match PEAT's parameter budget
        from peat.peat import attach_lora
        model = attach_lora(model, model_tag)
        model.train()

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=2e-5, weight_decay=0.01,
        )
        cast_dtype = get_autocast_dtype()

        for epoch in range(3):
            total_loss = 0
            n = 0
            for batch in tqdm(orig_loader, desc=f"CDA epoch {epoch+1}"):
                contexts = batch["context"]
                fillers = batch["t_s"]  # train on stereotype sentences
                for ctx, filler in zip(contexts, fillers):
                    text = ctx.replace("BLANK", filler)
                    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                       max_length=512, padding="max_length").to(device)
                    inputs["labels"] = inputs["input_ids"].clone()
                    # Mask padding tokens: -100 is ignored by all HF loss functions
                    if tokenizer.pad_token_id is not None:
                        inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100

                    with torch.amp.autocast("cuda", dtype=cast_dtype):
                        outputs = model(**inputs)
                        loss = outputs.loss

                    if loss is not None and torch.isfinite(loss):
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()
                        n += 1

            # Also train on swapped data
            for batch in tqdm(swap_loader, desc=f"CDA-swap epoch {epoch+1}"):
                contexts = batch["context"]
                fillers = batch["t_a"]  # swapped: anti-stereo as target
                for ctx, filler in zip(contexts, fillers):
                    text = ctx.replace("BLANK", filler)
                    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                       max_length=512, padding="max_length").to(device)
                    inputs["labels"] = inputs["input_ids"].clone()
                    # Mask padding tokens: -100 is ignored by all HF loss functions
                    if tokenizer.pad_token_id is not None:
                        inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100

                    with torch.amp.autocast("cuda", dtype=cast_dtype):
                        outputs = model(**inputs)
                        loss = outputs.loss

                    if loss is not None and torch.isfinite(loss):
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()
                        n += 1

            avg = total_loss / max(n, 1)
            logger.info(f"  CDA epoch {epoch+1}: avg_loss={avg:.4f}")

        model.eval()
        _csv = RAW_DIR / "baselines" / "cda" / model_tag / f"seed_{seed}"
        _csv.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device, csv_dir=_csv)
        metrics["method"] = "CDA"
        metrics["seed"] = seed
        return metrics

    except Exception as e:
        logger.error(f"CDA failed for {model_tag}: {e}")
        return {"method": "CDA", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        cleanup(model)
