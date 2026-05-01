"""
PEAT — Baselines Registry.

Maps baseline method names to their run() functions.
"""

from peat.baselines.base import run as run_base
from peat.baselines.cda import run as run_cda
from peat.baselines.self_debias import run as run_self_debias
from peat.baselines.auto_debias import run as run_auto_debias
from peat.baselines.bias_edit import run as run_bias_edit
from peat.baselines.fair_steer import run as run_fair_steer
from peat.baselines.bias_unlearn import run as run_bias_unlearn
from peat.baselines.know_bias import run as run_know_bias
from peat.baselines.lora_vanilla_sft import run as run_lora_vanilla_sft

BASELINE_REGISTRY = {
    "base": run_base,
    "cda": run_cda,
    "self_debias": run_self_debias,
    "auto_debias": run_auto_debias,
    "bias_edit": run_bias_edit,
    "fair_steer": run_fair_steer,
    "bias_unlearn": run_bias_unlearn,
    "know_bias": run_know_bias,
    "lora_vanilla_sft": run_lora_vanilla_sft,
}

BASELINE_NAMES = list(BASELINE_REGISTRY.keys())
