"""
Baseline: KnowBias — Bias-neuron enhancement at inference.

# Reference: Pan et al., "KnowBias: Detecting and Mitigating Biases via
# Knowledge-Aware Neuron Enhancement", arXiv 2601.21864, Jan 2026.
# Used here for: identifying bias-relevant neurons and enhancing them at inference.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from peat.data import load_stereoset_pairs
from peat.eval import compute_stereotype_score, evaluate_full
from peat.interventions import assert_intervention_active, neuron_damping_intervention
from peat.models import get_spec, load_model
from peat.utils import LOG_DIR, RAW_DIR, cleanup, set_seed, setup_logger

PROBE_ROWS = 40
KNOWBIAS_GAMMA = 0.5   # damping factor applied to identified bias neurons

logger = setup_logger("peat.baselines.know_bias", str(LOG_DIR / "baselines.log"))


def _identify_bias_neurons(model, tokenizer, train_df, model_tag, device,
                            n_samples=200, top_k=50):
    """Identify neurons most associated with bias via activation analysis.

    For each training pair, compute activations for stereo vs anti-stereo,
    and identify neurons with the largest activation differences.
    """
    spec = get_spec(model_tag)
    neuron_diffs = None
    count = 0

    for _, row in tqdm(train_df.iterrows(), total=min(n_samples, len(train_df)),
                       desc="Identifying bias neurons"):
        if count >= n_samples:
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
                # Use last hidden layer, mean-pooled
                h_s = out_s.hidden_states[-1].mean(dim=1).squeeze(0)
                h_a = out_a.hidden_states[-1].mean(dim=1).squeeze(0)
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
                def _mean0(out):
                    if hasattr(out, "last_hidden_state"):
                        return out.last_hidden_state.mean(dim=1).squeeze(0)
                    return out[0].mean(dim=1).squeeze(0)
                h_s = _mean0(out_s)
                h_a = _mean0(out_a)

            diff = (h_s - h_a).abs()
            if neuron_diffs is None:
                neuron_diffs = diff
            else:
                neuron_diffs = neuron_diffs + diff
            count += 1

    if neuron_diffs is None or count == 0:
        return None

    neuron_diffs = neuron_diffs / count
    # Top-k neurons with highest average activation difference
    _, top_indices = neuron_diffs.topk(min(top_k, neuron_diffs.size(0)))
    return top_indices


def run(model_tag: str, seed: int = 42, device: str = "cuda",
        _model=None, _tokenizer=None) -> dict:
    """Run KnowBias baseline."""
    logger.info(f"KnowBias: {model_tag}, seed={seed}")
    set_seed(seed)

    _owns = _model is None
    if _owns:
        model, tokenizer, _ = load_model(model_tag, device=device)
    else:
        model, tokenizer = _model, _tokenizer

    try:
        train_df, _ = load_stereoset_pairs(seed=seed)
        bias_neurons = _identify_bias_neurons(model, tokenizer, train_df,
                                               model_tag, device)

        if bias_neurons is None:
            logger.warning(f"KnowBias: could not identify bias neurons for {model_tag}")
            return {"method": "KnowBias", "model": model_tag, "seed": seed,
                    "status": "skipped: no bias neurons identified"}

        logger.info(f"  Identified {len(bias_neurons)} bias neurons")

        # WP-A fix: damp the identified bias neurons via forward hooks so the
        # intervention is active during SS scoring. Previously the unmodified
        # model was scored, yielding Base-identical SS.
        model.eval()

        base_probe = compute_stereotype_score(
            model, tokenizer, model_tag, device, max_rows=PROBE_ROWS)["results_df"]
        with neuron_damping_intervention(model, bias_neurons, gamma=KNOWBIAS_GAMMA):
            method_probe = compute_stereotype_score(
                model, tokenizer, model_tag, device, max_rows=PROBE_ROWS)["results_df"]
        assert_intervention_active(base_probe, method_probe, "KnowBias")

        with neuron_damping_intervention(model, bias_neurons, gamma=KNOWBIAS_GAMMA):
            _csv = RAW_DIR / "baselines" / "know_bias" / model_tag / f"seed_{seed}"
            _csv.mkdir(parents=True, exist_ok=True)
            metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device, csv_dir=_csv)
        metrics["method"] = "KnowBias"
        metrics["seed"] = seed
        metrics["n_bias_neurons"] = len(bias_neurons)
        metrics["knowbias_gamma"] = KNOWBIAS_GAMMA
        return metrics

    except Exception as e:
        logger.error(f"KnowBias failed for {model_tag}: {e}")
        return {"method": "KnowBias", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        if _owns:
            cleanup(model)
