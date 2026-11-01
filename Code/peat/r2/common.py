"""
Shared helpers for the revision-2 experiments: paths, smoke-mode sizing,
extended LoRA attachment (rank / placement sweep), adapter persistence,
batched padding, and JSON/CSV writers.

# Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
# ICLR 2022. arXiv: 2106.09685 | Used here for: adapters with rank and
# placement varied for the sensitivity and placement experiments.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from peat.models import get_lora_target_modules, get_spec
from peat.utils import PROJECT_ROOT, get_dtype

R2_DIR = PROJECT_ROOT / "results" / "r2"
CELLS_DIR = R2_DIR / "cells"
ADAPTER_DIR = R2_DIR / "adapters"
CLAIMS_DIR = R2_DIR / "claims"
R2_DATA_DIR = PROJECT_ROOT / "data" / "r2"

# Smoke mode (run_r2.py --smoke): every dataset is cut to SMOKE_N rows and
# training to one short epoch, so the full code path runs in minutes.
SMOKE = os.environ.get("PEAT_R2_SMOKE", "0") == "1"
SMOKE_N = int(os.environ.get("PEAT_R2_SMOKE_N", "2"))


def cap(df: pd.DataFrame, n_full: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Return the smoke-sized or (optionally sub-sampled) full frame."""
    if SMOKE:
        return df.head(SMOKE_N).reset_index(drop=True)
    if n_full is not None and len(df) > n_full:
        return df.sample(n_full, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def ensure_r2_dirs() -> None:
    for d in (CELLS_DIR, ADAPTER_DIR, CLAIMS_DIR, R2_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def cell_dir(cell_id: str) -> Path:
    root = CELLS_DIR if not SMOKE else R2_DIR / "smoke" / "cells"
    d = root / cell_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def env_info() -> dict:
    info = {"host": platform.node(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__}
    try:
        import transformers, peft
        info["transformers"] = transformers.__version__
        info["peft"] = peft.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info


# ---------------------------------------------------------------------------
# LoRA attachment with rank and placement options
# ---------------------------------------------------------------------------

def num_layers(model) -> int:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    return getattr(cfg, "num_hidden_layers", getattr(text_cfg, "num_hidden_layers", 0))


def placement_layers(model, placement: str) -> Optional[list[int]]:
    n = num_layers(model)
    if placement == "last2":
        return [n - 2, n - 1]
    if placement == "first2":
        return [0, 1]
    if placement == "middle2":
        return [n // 2 - 1, n // 2]
    if placement == "all":
        return None
    raise ValueError(f"unknown placement {placement!r}")


def attach_lora_ext(model, model_tag: str, rank: int = 4, placement: str = "last2"):
    """Freeze the base model and attach LoRA, mirroring peat.peat.attach_lora.

    rank=4/placement=last2 reproduces the published configuration exactly
    (r=4, alpha=8, dropout 0, LoRA-B zero-initialised, adapters cast to the
    model dtype). For the rank sweep alpha is kept at 2*rank so that the
    effective scale alpha/r stays at the published value of 2.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    spec = get_spec(model_tag)
    target_modules = get_lora_target_modules(model_tag)
    layers = placement_layers(model, placement)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        layers_to_transform=layers,
        bias="none",
        task_type=TaskType.CAUSAL_LM if spec.is_causal else None,
        init_lora_weights=True,
    )
    model = get_peft_model(model, cfg)
    dtype = get_dtype()
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(dtype)
    return model


def count_params(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "pct": 100.0 * trainable / max(total, 1)}


def save_adapter(model, cell_id: str) -> Path:
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    root = ADAPTER_DIR if not SMOKE else R2_DIR / "smoke" / "adapters"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{cell_id}.safetensors"
    sd = {k: v.detach().to("cpu").contiguous() for k, v in get_peft_model_state_dict(model).items()}
    save_file(sd, str(path))
    return path


def load_adapter(model, path: Path, device: str) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(str(path))
    set_peft_model_state_dict(model, {k: v.to(device) for k, v in sd.items()})


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def pad_batch(seqs: list[list[int]], pad_id: int, device: str):
    """Right-pad token-id lists; returns (input_ids, attention_mask, lengths)."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        att[i, : len(s)] = 1
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return ids.to(device), att.to(device), lens


def pad_id_of(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def forward_logits(model, ids, att):
    out = model(input_ids=ids, attention_mask=att)
    return out.logits
