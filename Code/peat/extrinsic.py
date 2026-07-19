"""
PEAT — Extrinsic and cross-benchmark fairness evaluation (WP-E).

Reviewers #1 and #3 (APIN-D-26-06244) require at least one extrinsic
(downstream) fairness evaluation and evidence beyond the CrowS-Pairs intrinsic
benchmark. This module adds:

  * Bias-in-Bios occupation-classification probe (encoders): frozen-encoder
    embeddings → logistic-regression probe → accuracy + gender TPR gap.
    # Reference: De-Arteaga et al., "Bias in Bios", FAT* 2019 (arXiv:1901.09451).
  * HONEST hurtful-completion score (encoders and decoders).
    # Reference: Nozza et al., "HONEST: Measuring Hurtful Sentence Completion
    # in Language Models", NAACL 2021. Uses the `evaluate` implementation.
  * StereoSet selection-split SS (cross-benchmark, within-family
    generalisation: gradient-trained on the 90% split, scored on the held-out
    10% split; CrowS-Pairs remains the unseen cross-benchmark).

BBQ (extrinsic QA for decoders) is already part of evaluate_full (eval.py).
"""
from __future__ import annotations

import numpy as np
import torch

from peat.models import get_spec
from peat.utils import LOG_DIR, SMOKE_TEST, SMOKE_TEST_SIZE, setup_logger

logger = setup_logger("peat.extrinsic", str(LOG_DIR / "evaluation.log"))

BIOS_TRAIN_N = 8000
BIOS_TEST_N = 4000
HONEST_MAX_TEMPLATES = 800
HONEST_TOPK_ENCODER = 5


# ---------------------------------------------------------------------------
# Shared: mean-pooled sentence embedding for any supported architecture
# ---------------------------------------------------------------------------
@torch.no_grad()
def _embed(model, tokenizer, texts: list[str], device: str,
           batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings (frozen model)."""
    out_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True).to(device)
        try:
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1]
        except TypeError:
            # NomicBertForPreTraining: no output_hidden_states — use inner encoder
            _enc_mod = getattr(model, "bert", getattr(model, "encoder", None))
            kw = {k: v for k, v in enc.items()
                  if k in ("input_ids", "attention_mask", "token_type_ids")}
            out = _enc_mod(**kw) if _enc_mod is not None else model(**kw)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out_vecs.append(pooled.float().cpu().numpy())
    return np.concatenate(out_vecs, axis=0)


# ---------------------------------------------------------------------------
# Bias-in-Bios — occupation classification probe (encoders)
# ---------------------------------------------------------------------------
def evaluate_bias_in_bios(model, tokenizer, model_tag: str,
                          device: str = "cuda") -> dict:
    """Frozen-encoder probe on LabHC/bias_in_bios: accuracy + gender TPR gap.

    TPR gap = mean over occupations of |TPR_female - TPR_male| (occupations
    with >= 20 test examples per gender). Lower gap with preserved accuracy
    indicates the debiased encoder carries less gender-occupation leakage.
    """
    try:
        from datasets import load_dataset
        from sklearn.linear_model import LogisticRegression

        ds_tr = load_dataset("LabHC/bias_in_bios", split="train")
        ds_te = load_dataset("LabHC/bias_in_bios", split="test")

        n_tr = SMOKE_TEST_SIZE if SMOKE_TEST else BIOS_TRAIN_N
        n_te = SMOKE_TEST_SIZE if SMOKE_TEST else BIOS_TEST_N
        rng = np.random.default_rng(42)
        tr_idx = rng.choice(len(ds_tr), size=min(n_tr, len(ds_tr)), replace=False)
        te_idx = rng.choice(len(ds_te), size=min(n_te, len(ds_te)), replace=False)
        tr = ds_tr.select(tr_idx.tolist())
        te = ds_te.select(te_idx.tolist())

        model.eval()
        X_tr = _embed(model, tokenizer, [r["hard_text"] for r in tr], device)
        X_te = _embed(model, tokenizer, [r["hard_text"] for r in te], device)
        y_tr = np.array([r["profession"] for r in tr])
        y_te = np.array([r["profession"] for r in te])
        g_te = np.array([r["gender"] for r in te])

        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        acc = float((pred == y_te).mean())

        gaps = []
        for prof in np.unique(y_te):
            m = (y_te == prof) & (g_te == 0)
            f = (y_te == prof) & (g_te == 1)
            if (SMOKE_TEST or (m.sum() >= 20 and f.sum() >= 20)):
                if m.sum() > 0 and f.sum() > 0:
                    tpr_m = float((pred[m] == prof).mean())
                    tpr_f = float((pred[f] == prof).mean())
                    gaps.append(abs(tpr_f - tpr_m))
        tpr_gap = float(np.mean(gaps)) if gaps else float("nan")

        logger.info(f"Bias-in-Bios({model_tag}): acc={acc:.4f} tpr_gap={tpr_gap:.4f} "
                    f"({len(gaps)} occupations)")
        return {"bios_accuracy": acc, "bios_tpr_gap": tpr_gap,
                "bios_n_occupations": len(gaps)}
    except Exception as e:
        logger.warning(f"Bias-in-Bios({model_tag}) failed: {e}")
        return {"bios_accuracy": float("nan"), "bios_tpr_gap": float("nan"),
                "bios_error": str(e)[:200]}


# ---------------------------------------------------------------------------
# HONEST — hurtful sentence completion (encoders + decoders)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_honest(model, tokenizer, model_tag: str, device: str = "cuda") -> dict:
    """HONEST score: fraction of hurtful completions on MilaNLProc/honest (en).

    Encoders: top-k fills at the [M] position (k=5).
    Decoders: greedy continuation, first generated word (k=1).
    Scored with `evaluate.load('honest', 'en')` (HurtLex lexicon inside).
    """
    spec = get_spec(model_tag)
    try:
        import evaluate as hf_evaluate
        from datasets import load_dataset

        ds = load_dataset("MilaNLProc/honest", "en_binary", split="honest")
        templates = [r["template_masked"] for r in ds][:HONEST_MAX_TEMPLATES]
        if SMOKE_TEST:
            templates = templates[:SMOKE_TEST_SIZE]

        model.eval()
        completions: list[list[str]] = []

        if spec.is_encoder:
            k = HONEST_TOPK_ENCODER
            for tpl in templates:
                text = tpl.replace("[M]", tokenizer.mask_token)
                enc = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=128).to(device)
                logits = model(**enc).logits[0]
                mask_pos = (enc["input_ids"][0] == tokenizer.mask_token_id
                            ).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    completions.append([""] * k)
                    continue
                top = logits[mask_pos[0]].topk(k).indices.tolist()
                completions.append(
                    [tokenizer.decode([t]).strip().lower() for t in top])
        else:
            for tpl in templates:
                prompt = tpl.split("[M]")[0].strip()
                enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=128).to(device)
                gen = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id
                                     or tokenizer.eos_token_id)
                new = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
                first_word = new.strip().split()[0].lower() if new.strip() else ""
                completions.append([first_word])

        honest_metric = hf_evaluate.load("honest", "en")
        score = honest_metric.compute(predictions=completions)
        honest_score = float(score.get("honest_score", float("nan")))
        logger.info(f"HONEST({model_tag}): {honest_score:.4f} "
                    f"over {len(completions)} templates")
        return {"honest_score": honest_score, "honest_n_templates": len(completions)}
    except Exception as e:
        logger.warning(f"HONEST({model_tag}) failed: {e}")
        return {"honest_score": float("nan"), "honest_error": str(e)[:200]}


# ---------------------------------------------------------------------------
# StereoSet held-out split SS (cross-benchmark, within-family)
# ---------------------------------------------------------------------------
def evaluate_stereoset_heldout_ss(model, tokenizer, model_tag: str,
                                  device: str = "cuda") -> dict:
    """SS on the StereoSet 10% held-out split (never gradient-trained)."""
    try:
        from peat.data import load_stereoset_pairs
        from peat.peat import selection_split_ss
        _, val_df = load_stereoset_pairs(seed=42)
        ss = selection_split_ss(model, tokenizer, model_tag, val_df, device)
        model.eval()
        logger.info(f"StereoSet-heldout({model_tag}): SS={ss:.2f} (n={len(val_df)})")
        return {"stereoset_heldout_ss": float(ss), "stereoset_heldout_n": int(len(val_df))}
    except Exception as e:
        logger.warning(f"StereoSet-heldout({model_tag}) failed: {e}")
        return {"stereoset_heldout_ss": float("nan")}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def evaluate_extrinsic(model, tokenizer, model_tag: str, device: str = "cuda") -> dict:
    """Run the WP-E suite appropriate to the architecture.

    Encoders: Bias-in-Bios + HONEST + StereoSet-heldout.
    Decoders: HONEST + StereoSet-heldout (BBQ already in evaluate_full).
    """
    spec = get_spec(model_tag)
    metrics: dict = {}
    metrics.update(evaluate_stereoset_heldout_ss(model, tokenizer, model_tag, device))
    metrics.update(evaluate_honest(model, tokenizer, model_tag, device))
    if spec.is_encoder:
        metrics.update(evaluate_bias_in_bios(model, tokenizer, model_tag, device))
    return metrics
