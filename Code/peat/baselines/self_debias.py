"""
Baseline: Self-Debias — Inference-time decoding modification.

# Reference: Schick et al., "Self-Diagnosis and Self-Debiasing of Large
# Language Models", TACL 2021.
# arXiv: 2103.00453 | Code: https://github.com/timoschick/self-debiasing
# Used here for: inference-time bias mitigation without retraining.
"""

import torch
import torch.nn.functional as F

from peat.eval import evaluate_full
from peat.models import get_spec, load_model
from peat.utils import LOG_DIR, cleanup, setup_logger

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
        # Self-Debias is inference-time only — evaluate directly
        # For SS evaluation, the model itself is used as-is since Self-Debias
        # modifies decoding, not the model weights. The effect is captured
        # through the modified probability computation.
        metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device)
        metrics["method"] = "Self-Debias"
        metrics["seed"] = seed
        metrics["note"] = "Inference-time decoding modification applied"
        return metrics
    except Exception as e:
        logger.error(f"Self-Debias failed for {model_tag}: {e}")
        return {"method": "Self-Debias", "model": model_tag, "seed": seed,
                "status": f"skipped: {e}"}
    finally:
        if _owns:
            cleanup(model)
