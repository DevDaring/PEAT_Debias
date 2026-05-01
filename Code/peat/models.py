"""
PEAT — Model Registry and Loaders.

Hard-coded registry of the six models used in this study. Provides
load_encoder() and load_causal() with uniform precision policy.

# UNIFORM PRECISION POLICY
# All models in this study load in bfloat16 if the GPU supports it (compute capability >= 8.0),
# else float16. We do NOT use 4-bit or 8-bit quantization for any model in any baseline or PEAT
# run, because mixed precision regimes across baselines would invalidate compute-vs-accuracy
# comparisons. LoRA adapters are also in bfloat16/float16 matching the base model. Flash-Attention 2
# is enabled for every causal model and for ModernBERT/NomicBERT where supported. BERT-base uses
# eager attention because it predates SDPA-flash compatibility for masked-LM heads.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from peat.secrets import hf_token
from peat.utils import get_dtype, setup_logger

logger = setup_logger("peat.models")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single model in the study."""
    tag: str
    hf_id: str
    model_type: str  # "encoder" or "causal"
    attn_impl: str   # "eager" or "flash_attention_2"
    trust_remote_code: bool = False
    gated: bool = False  # requires HF token for download

    @property
    def is_encoder(self) -> bool:
        return self.model_type == "encoder"

    @property
    def is_causal(self) -> bool:
        return self.model_type == "causal"


# Reference: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers
# for Language Understanding", NAACL 2019.
# Used here for: encoder MLM baseline (110M params).
BERT_BASE = ModelSpec(
    tag="bert-base",
    hf_id="google-bert/bert-base-uncased",
    model_type="encoder",
    attn_impl="eager",
)

# Reference: Warner et al., "ModernBERT", 2024.
# Used here for: modern encoder MLM. Use sdpa to avoid transformers-5.x/flash_attn-2.8.x
# API conflict (TypeError: the first argument must be callable).
MODERNBERT_BASE = ModelSpec(
    tag="modernbert-base",
    hf_id="answerdotai/ModernBERT-base",
    model_type="encoder",
    attn_impl="sdpa",
)

# Reference: Nussbaum et al., "Nomic Embed", 2024 (NomicBERT backbone).
# Used here for: modern 137M encoder MLM with RoPE and 2048-token context.
# Requires trust_remote_code and einops. Ignores torch_dtype at load time;
# we manually cast to bfloat16 after loading. Use eager attention.
NOMICBERT = ModelSpec(
    tag="nomicbert",
    hf_id="nomic-ai/nomic-bert-2048",
    model_type="encoder",
    attn_impl="eager",
    trust_remote_code=True,
)

# Reference: Qwen Team, "Qwen2.5 Technical Report", 2024.
# Used here for: causal LM (1.5B); hyperparameter search base.
# Note: sdpa (PyTorch scaled dot product attention) uses FlashAttention kernel
# on CUDA automatically; avoids transformers-5.x/flash_attn-2.8.x API conflict.
QWEN_25_15B = ModelSpec(
    tag="qwen2.5-1.5b",
    hf_id="Qwen/Qwen2.5-1.5B-Instruct",
    model_type="causal",
    attn_impl="sdpa",
)

# Reference: Google DeepMind, "Gemma 3 Technical Report", 2025.
# Used here for: mid-size causal LM (4B); scaling evaluation.
GEMMA_3_4B = ModelSpec(
    tag="gemma-3-4b",
    hf_id="google/gemma-3-4b-it",
    model_type="causal",
    attn_impl="sdpa",
    gated=True,
)

# Reference: Meta AI, "Llama 3.1", 2024.
# Used here for: causal LM (8B); scaling evaluation. Upgraded from 3.2-3B to
# take advantage of the A100-SXM4-80GB VRAM headroom (peak ~34GB vs 80GB).
LLAMA_31_8B = ModelSpec(
    tag="llama-3.1-8b",
    hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_type="causal",
    attn_impl="sdpa",
    gated=True,
)


# Full registry
MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.tag: spec
    for spec in [BERT_BASE, MODERNBERT_BASE, NOMICBERT, QWEN_25_15B, GEMMA_3_4B, LLAMA_31_8B]
}

# Core models (hyperparameter search + full baselines)
CORE_MODELS = ["bert-base", "modernbert-base", "nomicbert", "qwen2.5-1.5b"]

# Scaling models (reuse c_best from qwen2.5-1.5b)
SCALING_MODELS = ["gemma-3-4b", "llama-3.1-8b"]

# All encoder tags
ENCODER_TAGS = [tag for tag, spec in MODEL_REGISTRY.items() if spec.is_encoder]

# All causal tags
CAUSAL_TAGS = [tag for tag, spec in MODEL_REGISTRY.items() if spec.is_causal]


def get_spec(tag: str) -> ModelSpec:
    """Look up a ModelSpec by tag. Raises KeyError if not found."""
    if tag not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model tag '{tag}'. Valid tags: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[tag]


# ---------------------------------------------------------------------------
# LoRA target modules per model type
# ---------------------------------------------------------------------------
def get_lora_target_modules(tag: str) -> list[str]:
    """Return the LoRA target module names for a given model.

    For encoders:  query, key, value, dense (attention) +
                   intermediate.dense, output.dense (FFN)
    For causal:    q_proj, k_proj, v_proj, o_proj (attention) +
                   gate_proj, up_proj, down_proj (FFN)
    """
    spec = get_spec(tag)
    if spec.is_encoder:
        if tag == "bert-base":
            return [
                "query", "key", "value", "dense",
                "intermediate.dense", "output.dense",
            ]
        elif tag == "modernbert-base":
            # ModernBERT uses fused QKV ("Wqkv"), output projection ("Wo"),
            # and FFN layers ("Wi", "Wo"). Deduplicate since "Wo" appears in
            # both attention and FFN but peft matches by name globally.
            return ["Wqkv", "Wo", "Wi"]
        elif tag == "nomicbert":
            # NomicBERT uses fused QKV ('Wqkv'), output projection ('out_proj'),
            # and gated FFN layers ('fc11', 'fc12', 'fc2').
            return ["Wqkv", "out_proj", "fc11", "fc12", "fc2"]
    else:
        # Causal LMs (Qwen, Gemma, Llama) all use the same projection names
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    return []


def get_last_n_layer_indices(model, n: int = 2) -> list[int]:
    """Return the indices of the last `n` transformer layers.

    Inspects model.config.num_hidden_layers to determine total layer count.
    Gemma3Config nests this under text_config; falls back gracefully.
    """
    _cfg = model.config
    _text_cfg = getattr(_cfg, "text_config", _cfg)
    num_layers = getattr(_cfg, "num_hidden_layers", getattr(_text_cfg, "num_hidden_layers", 0))
    return list(range(num_layers - n, num_layers))


def get_lora_layers_pattern(tag: str, model) -> Optional[list[str]]:
    """Return the layers_to_transform for LoRA (last 2 layers only).

    Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
    ICLR 2022. arXiv: 2106.09685 | Code: https://github.com/microsoft/LoRA
    Used here for: parameter-efficient adapter restricted to last 2 transformer blocks.
    """
    indices = get_last_n_layer_indices(model, n=2)
    _cfg = model.config
    _text_cfg = getattr(_cfg, "text_config", _cfg)
    num_layers = getattr(_cfg, "num_hidden_layers", getattr(_text_cfg, "num_hidden_layers", 0))
    logger.info(
        f"  LoRA target layers for {tag}: {indices} "
        f"(total layers: {num_layers})"
    )
    return indices


# ---------------------------------------------------------------------------
# Loader: Encoder MLM
# ---------------------------------------------------------------------------
def load_encoder(
    tag: str,
    device: str = "cuda",
) -> Tuple[AutoModelForMaskedLM, AutoTokenizer, dict]:
    """Load an encoder model for masked language modeling.

    Args:
        tag: Model tag from the registry (e.g., 'bert-base').
        device: Target device ('cuda' or 'cpu').

    Returns:
        (model, tokenizer, config_dict) tuple.
    """
    spec = get_spec(tag)
    if not spec.is_encoder:
        raise ValueError(f"Model '{tag}' is not an encoder. Use load_causal() instead.")

    dtype = get_dtype()
    token = hf_token() if spec.gated else None

    logger.info(f"Loading encoder: {spec.hf_id} (dtype={dtype}, attn={spec.attn_impl})")

    # Tokenizer — try local cache first; download only if missing.
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id,
        token=token,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=False,  # HF hub caches automatically
    )

    # Model
    model_kwargs = {
        "torch_dtype": dtype,
        "token": token,
        "trust_remote_code": spec.trust_remote_code,
    }

    # Only set attn_implementation if not eager (some older models don't support the kwarg)
    if spec.attn_impl != "eager":
        model_kwargs["attn_implementation"] = spec.attn_impl

    # Try loading from local cache first (no network), fall back to download.
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            spec.hf_id, local_files_only=True, **model_kwargs
        )
        logger.info(f"  Loaded {spec.tag} from local cache")
    except OSError:
        logger.info(f"  {spec.tag} not in local cache; downloading from HuggingFace Hub...")
        model = AutoModelForMaskedLM.from_pretrained(spec.hf_id, **model_kwargs)
    # Some custom loaders (e.g. NomicBERT) use torch.load internally and
    # ignore torch_dtype. Explicitly cast to the target dtype after loading.
    model = model.to(dtype=dtype, device=device)
    model.eval()

    config_dict = {
        "tag": spec.tag,
        "hf_id": spec.hf_id,
        "model_type": spec.model_type,
        "attn_impl": spec.attn_impl,
        "dtype": str(dtype),
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_params": sum(p.numel() for p in model.parameters()),
    }

    logger.info(
        f"  Loaded {spec.tag}: {config_dict['num_params']/1e6:.1f}M params, "
        f"{config_dict['num_hidden_layers']} layers"
    )
    return model, tokenizer, config_dict


# ---------------------------------------------------------------------------
# Loader: Causal LM
# ---------------------------------------------------------------------------
def load_causal(
    tag: str,
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """Load a causal language model.

    Args:
        tag: Model tag from the registry (e.g., 'qwen2.5-1.5b').
        device: Target device ('cuda' or 'cpu').

    Returns:
        (model, tokenizer, config_dict) tuple.
    """
    spec = get_spec(tag)
    if not spec.is_causal:
        raise ValueError(f"Model '{tag}' is not a causal LM. Use load_encoder() instead.")

    dtype = get_dtype()
    token = hf_token() if spec.gated else None

    logger.info(f"Loading causal: {spec.hf_id} (dtype={dtype}, attn={spec.attn_impl})")

    # Tokenizer — HF hub caches automatically; no extra local_files_only needed.
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id,
        token=token,
        trust_remote_code=spec.trust_remote_code,
    )
    # Ensure pad token exists for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs_c = {
        "torch_dtype": dtype,
        "attn_implementation": spec.attn_impl,
        "token": token,
        "trust_remote_code": spec.trust_remote_code,
    }
    # Try loading from local cache first; fall back to download.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, local_files_only=True, **model_kwargs_c
        )
        logger.info(f"  Loaded {spec.tag} from local cache")
    except OSError:
        logger.info(f"  {spec.tag} not in local cache; downloading from HuggingFace Hub...")
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs_c)
    model = model.to(device)
    model.eval()

    # Gemma3Config nests per-layer settings under text_config; fall back gracefully.
    _cfg = model.config
    _text_cfg = getattr(_cfg, "text_config", _cfg)
    config_dict = {
        "tag": spec.tag,
        "hf_id": spec.hf_id,
        "model_type": spec.model_type,
        "attn_impl": spec.attn_impl,
        "dtype": str(dtype),
        "num_hidden_layers": getattr(_cfg, "num_hidden_layers", getattr(_text_cfg, "num_hidden_layers", 0)),
        "hidden_size": getattr(_cfg, "hidden_size", getattr(_text_cfg, "hidden_size", 0)),
        "num_params": sum(p.numel() for p in model.parameters()),
    }

    logger.info(
        f"  Loaded {spec.tag}: {config_dict['num_params']/1e6:.1f}M params, "
        f"{config_dict['num_hidden_layers']} layers"
    )
    return model, tokenizer, config_dict


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------
def load_model(tag: str, device: str = "cuda"):
    """Load any model by tag, dispatching to the correct loader.

    Returns:
        (model, tokenizer, config_dict) tuple.
    """
    spec = get_spec(tag)
    if spec.is_encoder:
        return load_encoder(tag, device=device)
    else:
        return load_causal(tag, device=device)
