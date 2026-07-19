"""
Baseline: Self-Debias — Inference-time decoding modification.

# Reference: Schick et al., "Self-Diagnosis and Self-Debiasing of Large
# Language Models", TACL 2021.
# arXiv: 2103.00453 | Code: https://github.com/timoschick/self-debiasing
# Used here for: inference-time bias mitigation without retraining.
"""

import torch
import torch.nn.functional as F

from peat.eval import compute_stereotype_score, evaluate_full
from peat.interventions import assert_intervention_active, make_self_debias_scorer
from peat.models import get_spec, load_model
from peat.utils import LOG_DIR, RAW_DIR, cleanup, setup_logger

PROBE_ROWS = 40
SELF_DEBIAS_ALPHA = 50.0   # Schick et al. (2021) decay strength

logger = setup_logger("peat.baselines.self_debias", str(LOG_DIR / "baselines.log"))

BIAS_PROMPTS = {
    "race-color": "The following text discriminates against people based on their race or skin color: ",
    "gender": "The following text discriminates against people based on their gender: ",
    "religion": "The following text discriminates against people based on their religion: ",
    "age": "The following text discriminates against people based on their age: ",
    "nationality": "The following text discriminates against people based on their nationality: ",
    "disability": "The following text discriminates against people based on their disability: ",
    "physical-appearance": "The following text discriminates against people based on their physical appearance: ",
    "socioeconomic": "The following text discriminates against people based on their socioeconomic status: ",
    "sexual-orientation": "The following text discriminates against people based on their sexual orientation: ",
}


class SelfDebiasWrapper:
    """Wraps a model to apply self-debiasing at inference time.

    For each input, generates biased completions using bias-inducing prompts,
    then subtracts their logit contributions from the original output.
    """

    def __init__(self, model, tokenizer, model_tag, alpha=50.0, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_tag = model_tag
        self.alpha = alpha
        self.device = device
        self.spec = get_spec(model_tag)

    def get_debiased_logits(self, input_text):
        """Get debiased logits by subtracting bias-induced logit shifts."""
        inputs = self.tokenizer(input_text, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            base_outputs = self.model(**inputs)
            base_logits = base_outputs.logits[0, -1, :]  # last position

            # Compute bias-induced logits for each bias type
            bias_logits_list = []
            for bias_type, prompt in BIAS_PROMPTS.items():
                biased_input = prompt + input_text
                biased_inputs = self.tokenizer(biased_input, return_tensors="pt",
                                               truncation=True, max_length=512).to(self.device)
                biased_outputs = self.model(**biased_inputs)
                biased_logits = biased_outputs.logits[0, -1, :]
                bias_logits_list.append(biased_logits)

            # Self-debiasing: subtract max bias shift
            bias_shift = torch.stack(bias_logits_list, dim=0)
            max_bias = bias_shift.max(dim=0).values
            debiased = base_logits - self.alpha * F.softmax(max_bias - base_logits, dim=-1)

        return debiased


def run(model_tag: str, seed: int = 42, device: str = "cuda",
        _model=None, _tokenizer=None) -> dict:
    """Run Self-Debias baseline."""
    logger.info(f"Self-Debias: {model_tag}, seed={seed}")
    spec = get_spec(model_tag)

    if spec.is_encoder:
        logger.warning(f"Self-Debias: limited support for encoder {model_tag}")

    _owns = _model is None
    if _owns:
        model, tokenizer, _ = load_model(model_tag, device=device)
    else:
        model, tokenizer = _model, _tokenizer

    try:
        # WP-A fix: Self-Debias reshapes the scored token distribution via a
        # self-diagnosis prefix (Schick et al. 2021). This is applied through a
        # sentence_scorer override so the reweighting is active during SS
        # computation. Previously the unmodified model was scored (Base-identical).
        scorer = make_self_debias_scorer(alpha=SELF_DEBIAS_ALPHA)

        base_probe = compute_stereotype_score(
            model, tokenizer, model_tag, device, max_rows=PROBE_ROWS)["results_df"]
        method_probe = compute_stereotype_score(
            model, tokenizer, model_tag, device, max_rows=PROBE_ROWS,
            sentence_scorer=scorer)["results_df"]
        assert_intervention_active(base_probe, method_probe, "Self-Debias")

        _csv = RAW_DIR / "baselines" / "self_debias" / model_tag / f"seed_{seed}"
        _csv.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed],
                                device=device, sentence_scorer=scorer,
                                csv_dir=_csv)
        metrics["method"] = "Self-Debias"
        metrics["seed"] = seed
        metrics["self_debias_alpha"] = SELF_DEBIAS_ALPHA
        metrics["note"] = "Schick et al. reweighting active in SS scoring path"
        return metrics
    except Exception as e:
        logger.error(f"Self-Debias failed for {model_tag}: {e}")
        return {"method": "Self-Debias", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        if _owns:
            cleanup(model)
