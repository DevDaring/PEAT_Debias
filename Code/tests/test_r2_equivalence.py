"""
Equivalence check: batched revision-2 PEAT loss vs. the original per-item loss
(peat.peat.compute_peat_loss). Runs on CPU in float32 with a randomly
perturbed adapter so that all three terms are non-zero.

    python -m tests.test_r2_equivalence [encoder_hf_id] [decoder_hf_id]
"""

import copy
import sys

import torch

import peat.r2.common as common
from peat.r2 import fastpeat

common.get_dtype = lambda: torch.float32  # CPU test runs in float32


def _items():
    from peat.data import load_stereoset_pairs
    tr, _ = load_stereoset_pairs(seed=42)
    rows = tr.head(6)
    starts = tr[tr["context"].str.startswith("BLANK")].head(2)  # sentence-initial BLANK case
    import pandas as pd
    rows = pd.concat([rows, starts])
    print("sentence-initial BLANK contexts in train split:", int(tr["context"].str.startswith("BLANK").sum()))
    return [{"context": r.context, "t_s": r.t_s, "t_a": r.t_a} for r in rows.itertuples()]


def check(tag, hf_id, auto_cls):
    from transformers import AutoTokenizer
    from peat.peat import compute_peat_loss

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = auto_cls.from_pretrained(hf_id, torch_dtype=torch.float32)
    base.eval()
    theta0 = copy.deepcopy(base)
    model = common.attach_lora_ext(base, tag)
    torch.manual_seed(0)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02)
    model.eval()  # disable dropout in both paths for an exact comparison

    batch = _items()
    with torch.no_grad():
        _, ref = compute_peat_loss(model, theta0, tok, batch, tag, "cpu", 1.0, 1.0)
        items = [(b["context"], b["t_s"], b["t_a"]) for b in batch]
        L_neut, L_pair, L_kl, _ = fastpeat.peat_loss_terms(model, tok, tag, items, "cpu")
    new = {"L_neut": L_neut.item(), "L_pair": L_pair.item(), "L_kl": L_kl.item()}
    print(tag, "original:", {k: round(ref[k], 6) for k in new}, "batched:", {k: round(v, 6) for k, v in new.items()})
    for k in new:
        assert abs(new[k] - ref[k]) <= 1e-4 + 1e-3 * abs(ref[k]), (tag, k, new[k], ref[k])


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
    enc = sys.argv[1] if len(sys.argv) > 1 else "google-bert/bert-base-uncased"
    dec = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B-Instruct"
    check("bert-base", enc, AutoModelForMaskedLM)
    check("qwen2.5-1.5b", dec, AutoModelForCausalLM)
    print("EQUIVALENCE OK")
