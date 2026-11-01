"""
Revision-2 experiment plan: every cell is one (method, model, seed, variant).

Reviewer mapping
  base_*, peat_*, lora_*  -> R4.3 (Hindi/Bengali/French, BBQ, BOLD), R4.5 and
                             R5.2 (perplexity, KL/TV drift, zero-shot tasks),
                             R4.8 (loss/gradient logs), R5.1 (Delta tables)
  surr_*                  -> R4.1, R5.1 (surrogate choice)
  diag_*, place_*         -> R4.4 (encoder bias geometry, adapter placement)
  hp_*                    -> R4.7 (rank and learning-rate sensitivity)
  cb_*                    -> R4.6 (coverage-balanced synthetic pairs)
"""

from __future__ import annotations

ENCODERS = ["bert-base", "modernbert-base", "nomicbert"]
DECODERS = ["qwen2.5-1.5b", "gemma-3-4b", "llama-3.1-8b"]
CORE = ["bert-base", "modernbert-base", "nomicbert", "qwen2.5-1.5b"]
SEEDS = [42, 123, 456]

# Published SHA + bootstrap selections (paper Table 4); the two scaling
# decoders reuse the Qwen2.5-1.5B configuration exactly as in the paper.
LAMBDAS = {
    "bert-base": (1.0, 0.001),
    "modernbert-base": (1.0, 10.0),
    "nomicbert": (10.0, 0.01),
    "qwen2.5-1.5b": (0.01, 10.0),
    "gemma-3-4b": (0.01, 10.0),
    "llama-3.1-8b": (0.01, 10.0),
}

VRAM_GB = {"bert-base": 6, "modernbert-base": 8, "nomicbert": 8, "qwen2.5-1.5b": 12,
           "gemma-3-4b": 20, "llama-3.1-8b": 22}
EST_MIN = {"bert-base": 6, "modernbert-base": 8, "nomicbert": 8, "qwen2.5-1.5b": 10,
           "gemma-3-4b": 20, "llama-3.1-8b": 35}


def _short(m: str) -> str:
    return {"bert-base": "bert", "modernbert-base": "modernbert", "nomicbert": "nomic",
            "qwen2.5-1.5b": "qwen", "gemma-3-4b": "gemma", "llama-3.1-8b": "llama"}[m]


def _evals(model: str, full: bool, zeroshot: bool) -> list[str]:
    ev = ["crows_en", "delta"]
    if full:
        ev.append("utility")
        if model in DECODERS:
            ev += ["multi", "bbq", "bold"]
            if zeroshot:
                ev.append("zeroshot")
    return ev


def _train_cfg(model: str, **over) -> dict:
    l1, l2 = LAMBDAS[model]
    cfg = {"lambda_1": l1, "lambda_2": l2, "surrogate": "huber", "tau": 1.0, "lr": 1e-4,
           "epochs": 5, "rank": 4, "placement": "last2", "data": "stereoset"}
    cfg.update(over)
    return cfg


def build_plan() -> list[dict]:
    cells: list[dict] = []

    def add(cid, kind, model, seed, priority, evals, cfg=None, minutes=None):
        cells.append({"id": cid, "kind": kind, "model": model, "seed": seed, "priority": priority,
                      "evals": evals, "cfg": cfg or {}, "vram": VRAM_GB[model],
                      "est_min": minutes or EST_MIN[model]})

    for m in ENCODERS + DECODERS:
        add(f"base_{_short(m)}", "base", m, 0, 0, _evals(m, True, True))
    for m in ENCODERS + DECODERS:
        for s in SEEDS:
            add(f"peat_{_short(m)}_s{s}", "peat", m, s, 1, _evals(m, True, s == 42), _train_cfg(m))
    for m in CORE:
        for s in SEEDS:
            add(f"lora_{_short(m)}_s{s}", "lora", m, s, 1, _evals(m, True, s == 42),
                minutes=EST_MIN[m] * 2)
    for m in ENCODERS + DECODERS:
        add(f"diag_{_short(m)}", "diag", m, 0, 2, [])
    for kind in ["l2", "l1", "logcosh", "logistic", "hinge"]:
        for m in ["bert-base", "qwen2.5-1.5b"]:
            for s in SEEDS:
                add(f"surr_{kind}_{_short(m)}_s{s}", "peat", m, s, 2, ["crows_en", "delta"],
                    _train_cfg(m, surrogate=kind))
    for m in ["bert-base", "modernbert-base", "nomicbert", "qwen2.5-1.5b", "gemma-3-4b"]:
        for s in SEEDS:
            add(f"cb_{_short(m)}_s{s}", "peat", m, s, 3, ["crows_en", "delta"], _train_cfg(m, data="cb"))
    for pl in ["first2", "middle2", "all"]:
        for m in CORE:
            for s in [42, 123]:
                add(f"place_{pl}_{_short(m)}_s{s}", "peat", m, s, 3, ["crows_en", "delta"],
                    _train_cfg(m, placement=pl))
    for name, over in [("r2", {"rank": 2}), ("r8", {"rank": 8}), ("r16", {"rank": 16}),
                       ("lr5e-5", {"lr": 5e-5}), ("lr2e-4", {"lr": 2e-4})]:
        for m in ["bert-base", "qwen2.5-1.5b"]:
            for s in [42, 123]:
                add(f"hp_{name}_{_short(m)}_s{s}", "peat", m, s, 4, ["crows_en", "delta"],
                    _train_cfg(m, **over))
    return cells


SMOKE_CELLS = [
    "base_bert", "base_qwen", "peat_bert_s42", "peat_qwen_s42", "lora_bert_s42", "lora_qwen_s42",
    "surr_l1_qwen_s42", "place_middle2_nomic_s42", "hp_r8_bert_s42", "cb_modernbert_s42",
    "diag_bert", "diag_qwen", "base_gemma", "peat_llama_s42",
]


def get_cell(cid: str) -> dict:
    for c in build_plan():
        if c["id"] == cid:
            return c
    raise KeyError(cid)
