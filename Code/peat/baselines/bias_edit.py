"""
Baseline: BiasEdit — Lightweight editor networks for debiasing.

# Reference: Xu et al., "BiasEdit: Debiasing Stereotyped Language Models via
# Model Editing", TrustNLP@NAACL 2025.
# arXiv: 2503.08588 | Code: https://github.com/zjunlp/BiasEdit
# Used here for: debiasing + retention loss via editor networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from peat.data import StereoSetDataset, load_stereoset_pairs
from peat.eval import compute_stereotype_score, evaluate_full
from peat.interventions import assert_intervention_active, editor_intervention
from peat.models import get_spec, load_model
from peat.utils import LOG_DIR, RAW_DIR, cleanup, get_autocast_dtype, get_dtype, set_seed, setup_logger
from torch.utils.data import DataLoader

PROBE_ROWS = 40

logger = setup_logger("peat.baselines.bias_edit", str(LOG_DIR / "baselines.log"))


class BiasEditor(nn.Module):
    """Small editor network that learns to modify hidden states for debiasing."""

    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        edit = self.up(self.act(self.down(hidden_states)))
        gate = torch.sigmoid(self.gate(hidden_states))
        return hidden_states + gate * edit


def run(model_tag: str, seed: int = 42, device: str = "cuda",
        _model=None, _tokenizer=None) -> dict:
    """Run BiasEdit baseline."""
    logger.info(f"BiasEdit: {model_tag}, seed={seed}")
    set_seed(seed)
    spec = get_spec(model_tag)

    # Fine-tuning baseline: deepcopy so freeze+train never corrupts shared base
    if _model is not None:
        import copy
        model = copy.deepcopy(_model)
        tokenizer = _tokenizer
        hidden_size = _model.config.hidden_size
    else:
        model, tokenizer, cfg = load_model(model_tag, device=device)
        hidden_size = cfg["hidden_size"]

    try:
        editor = BiasEditor(hidden_size).to(device).to(get_dtype())

        # Freeze base model, only train editor
        for p in model.parameters():
            p.requires_grad = False

        optimizer = AdamW(editor.parameters(), lr=1e-4, weight_decay=0.01)
        cast_dtype = get_autocast_dtype()

        train_df, _ = load_stereoset_pairs(seed=seed)
        train_ds = StereoSetDataset(train_df)
        batch_size = 16 if spec.is_encoder else 4
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Training: minimize difference in predictions for stereo vs anti-stereo
        for epoch in range(3):
            total_loss = 0.0
            n = 0
            for batch in tqdm(loader, desc=f"BiasEdit epoch {epoch+1}"):
                for i in range(len(batch["context"])):
                    ctx = batch["context"][i]
                    t_s = batch["t_s"][i]
                    t_a = batch["t_a"][i]

                    text_s = ctx.replace("BLANK", t_s)
                    text_a = ctx.replace("BLANK", t_a)

                    inp_s = tokenizer(text_s, return_tensors="pt", truncation=True,
                                      max_length=512).to(device)
                    inp_a = tokenizer(text_a, return_tensors="pt", truncation=True,
                                      max_length=512).to(device)

                    with torch.amp.autocast("cuda", dtype=cast_dtype):
                        try:
                            out_s = model(**inp_s, output_hidden_states=True)
                            out_a = model(**inp_a, output_hidden_states=True)
                            h_s = out_s.hidden_states[-1][:, 0, :]  # CLS or first token
                            h_a = out_a.hidden_states[-1][:, 0, :]
                        except TypeError:
                            # NomicBertForPreTraining doesn't accept output_hidden_states
                            # and its output has no last_hidden_state — access inner encoder.
                            _enc = getattr(model, "bert", getattr(model, "encoder", None))
                            _kw_s = {k: v for k, v in inp_s.items()
                                     if k in ("input_ids", "attention_mask", "token_type_ids")}
                            _kw_a = {k: v for k, v in inp_a.items()
                                     if k in ("input_ids", "attention_mask", "token_type_ids")}
                            if _enc is not None:
                                out_s = _enc(**_kw_s)
                                out_a = _enc(**_kw_a)
                            else:
                                out_s = model(**_kw_s)
                                out_a = model(**_kw_a)
                            def _cls(out):
                                if hasattr(out, "last_hidden_state"):
                                    return out.last_hidden_state[:, 0, :]
                                return out[0][:, 0, :]
                            h_s = _cls(out_s)
                            h_a = _cls(out_a)

                        # Edit hidden states
                        edited_s = editor(h_s)
                        edited_a = editor(h_a)

                        # Debiasing loss: edited representations should be similar
                        debias_loss = F.mse_loss(edited_s, edited_a)

                        # Retention loss: edited should stay close to original
                        retain_loss = 0.5 * (F.mse_loss(edited_s, h_s.detach()) +
                                             F.mse_loss(edited_a, h_a.detach()))

                        loss = debias_loss + 0.1 * retain_loss

                    if torch.isfinite(loss):
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()
                        n += 1

            logger.info(f"  BiasEdit epoch {epoch+1}: avg_loss={total_loss/max(n,1):.6f}")

        # WP-A fix: apply the trained editor to the last block's hidden states
        # via a forward hook so the edit is active during SS scoring. Previously
        # the un-edited model was scored, yielding Base-identical SS.
        model.eval()
        editor.eval()

        base_probe = compute_stereotype_score(
            model, tokenizer, model_tag, device, max_rows=PROBE_ROWS)["results_df"]
        with editor_intervention(model, editor):
            method_probe = compute_stereotype_score(
                model, tokenizer, model_tag, device, max_rows=PROBE_ROWS)["results_df"]
        assert_intervention_active(base_probe, method_probe, "BiasEdit")

        with editor_intervention(model, editor):
            _csv = RAW_DIR / "baselines" / "bias_edit" / model_tag / f"seed_{seed}"
            _csv.mkdir(parents=True, exist_ok=True)
            metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device, csv_dir=_csv, skip_utility=True)
        metrics["method"] = "BiasEdit"
        metrics["seed"] = seed
        return metrics

    except Exception as e:
        logger.error(f"BiasEdit failed for {model_tag}: {e}")
        return {"method": "BiasEdit", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        cleanup(model)
