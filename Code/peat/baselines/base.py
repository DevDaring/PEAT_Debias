"""
Baseline: Base (no mitigation) — floor measurement.

Simply evaluates the unmodified pre-trained model on all metrics.
This serves as the floor/reference for all comparisons.
"""

from peat.eval import evaluate_full
from peat.models import load_model
from peat.utils import LOG_DIR, cleanup, setup_logger

logger = setup_logger("peat.baselines.base", str(LOG_DIR / "baselines.log"))


def run(model_tag: str, seed: int = 42, device: str = "cuda",
        _model=None, _tokenizer=None) -> dict:
    """Run base (no mitigation) evaluation.

    Returns dict of metrics.
    """
    logger.info(f"Base (no mitigation): {model_tag}, seed={seed}")
    _owns = _model is None
    if _owns:
        model, tokenizer, _ = load_model(model_tag, device=device)
    else:
        model, tokenizer = _model, _tokenizer

    try:
        metrics = evaluate_full(model, tokenizer, model_tag, seeds=[seed], device=device)
        metrics["method"] = "Base"
        metrics["seed"] = seed
        return metrics
    finally:
        if _owns:
            cleanup(model)
