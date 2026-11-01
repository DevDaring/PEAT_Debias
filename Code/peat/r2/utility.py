"""
Utility preservation beyond the Proposition-2 bound (Reviewer #4 c.5,
Reviewer #5 w.2).

Proposition 2 bounds the drift of the renormalised non-target distribution at
the training contexts only. The measurements here cover what it does not:
  * decoders: WikiText-103 test perplexity (sliding window, stride 512,
    window 1024, as peat.eval.evaluate_wikitext_perplexity) together with the
    full-vocabulary KL(P_theta0 || P_theta) and total-variation distance at
    every scored token of generic text;
  * encoders: pseudo-perplexity (Salazar et al. 2020) on WikiText-103 test
    sentences and the same KL/TV at every masked position;
  * decoders: zero-shot accuracy on six standard tasks, scored by
    log-likelihood exactly as in lm-evaluation-harness (acc / acc_norm).

# Reference: Merity et al., "Pointer Sentinel Mixture Models", 2016. arXiv: 1609.07843
# Reference: Salazar et al., "Masked Language Model Scoring", ACL 2020. arXiv: 1910.14659
# Reference: Gao et al., "A framework for few-shot language model evaluation"
#   (lm-evaluation-harness), Zenodo 2023. | Used here for: task formats and
#   acc/acc_norm definitions.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from peat.models import get_spec
from peat.r2.common import SMOKE, SMOKE_N, cap, pad_batch, pad_id_of, vocab_size


def _base_ctx(model):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else None


# ---------------------------------------------------------------------------
# Decoder: perplexity + KL/TV on WikiText-103
# ---------------------------------------------------------------------------

@torch.no_grad()
def decoder_ppl_kl(model, tokenizer, device="cuda", stride=512, max_length=1024,
                   max_tokens: int | None = None) -> dict:
    model.eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids_all = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    if SMOKE:
        ids_all = ids_all[: 2 * max_length]
    elif max_tokens:
        ids_all = ids_all[:max_tokens]
    n = ids_all.size(0)
    has_adapter = hasattr(model, "disable_adapter")
    nll = kl_sum = tv_sum = 0.0
    n_tok = 0
    prev_end = 0
    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        target_len = end - prev_end
        x = ids_all[begin:end].unsqueeze(0).to(device)
        logits = model(x).logits[:, :-1].float()
        actual = min(target_len, logits.size(1))
        lg = logits[0, -actual:]
        lab = x[0, 1:][-actual:]
        lp = torch.log_softmax(lg, -1)
        nll += -lp.gather(1, lab[:, None]).sum().item()
        if has_adapter:
            with model.disable_adapter():
                lg0 = model(x).logits[:, :-1].float()[0, -actual:]
            lp0 = torch.log_softmax(lg0, -1)
            p0 = lp0.exp()
            kl_sum += (p0 * (lp0 - lp)).sum().item()
            tv_sum += 0.5 * (p0 - lp.exp()).abs().sum().item()
        n_tok += actual
        prev_end = end
        if end >= n:
            break
    out = {"wikitext_ppl": math.exp(nll / max(n_tok, 1)), "wikitext_tokens": n_tok}
    if has_adapter:
        out.update({"wikitext_kl": kl_sum / n_tok, "wikitext_tv": tv_sum / n_tok})
    return out


# ---------------------------------------------------------------------------
# Encoder: pseudo-perplexity + KL/TV
# ---------------------------------------------------------------------------

def _wikitext_sentences(n: int = 400) -> list[str]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    out = []
    for x in ds:
        t = x["text"].strip()
        if not t or t.startswith("="):
            continue
        for s in re.split(r"(?<=[.!?])\s+", t):
            w = s.split()
            if 8 <= len(w) <= 40 and "@" not in s:
                out.append(s)
            if len(out) >= n:
                return out
    return out


@torch.no_grad()
def encoder_pppl_kl(model, tokenizer, device="cuda", n_sentences: int = 400, chunk: int = 256) -> dict:
    model.eval()
    sents = _wikitext_sentences(SMOKE_N if SMOKE else n_sentences)
    special = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    jobs = []
    for s in sents:
        toks = tokenizer.encode(s, add_special_tokens=True, truncation=True, max_length=128)
        for pos, t in enumerate(toks):
            if t in special:
                continue
            m = list(toks)
            m[pos] = tokenizer.mask_token_id
            jobs.append((m, pos, t))
    has_adapter = hasattr(model, "disable_adapter")
    nll = kl_sum = tv_sum = 0.0
    for start in range(0, len(jobs), chunk):
        part = jobs[start:start + chunk]
        ids, att, _ = pad_batch([j[0] for j in part], pad_id_of(tokenizer), device)
        r = torch.arange(len(part), device=device)
        pos = torch.tensor([j[1] for j in part], device=device)
        tgt = torch.tensor([j[2] for j in part], device=device)
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[r, pos].float(), -1)
        nll += -lp.gather(1, tgt[:, None]).sum().item()
        if has_adapter:
            with model.disable_adapter():
                lp0 = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[r, pos].float(), -1)
            p0 = lp0.exp()
            kl_sum += (p0 * (lp0 - lp)).sum().item()
            tv_sum += 0.5 * (p0 - lp.exp()).abs().sum().item()
    n = max(len(jobs), 1)
    out = {"wikitext_pppl": math.exp(nll / n), "wikitext_masked_tokens": len(jobs)}
    if has_adapter:
        out.update({"wikitext_kl": kl_sum / n, "wikitext_tv": tv_sum / n})
    return out


# ---------------------------------------------------------------------------
# Zero-shot multiple choice (decoders)
# ---------------------------------------------------------------------------

def _hs_pre(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _load_task(name: str):
    """Return list of (context, [continuations], gold, kind) with kind in
    {'mc','mc_norm','wino','lambada'}."""
    rows = []
    if name in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if name == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        for x in ds:
            labels = list(x["choices"]["label"])
            if x["answerKey"] not in labels:
                continue
            rows.append((f"Question: {x['question']}\nAnswer:",
                         [" " + c for c in x["choices"]["text"]], labels.index(x["answerKey"]), "mc_norm"))
    elif name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for x in ds:
            ctx = x["ctx_a"] + " " + x["ctx_b"].capitalize()
            q = _hs_pre(x["activity_label"] + ": " + ctx)
            rows.append((q, [" " + _hs_pre(e) for e in x["endings"]], int(x["label"]), "mc_norm"))
    elif name == "piqa":
        ds = load_dataset("baber/piqa", split="validation")
        for x in ds:
            rows.append((f"Question: {x['goal']}\nAnswer:", [" " + x["sol1"], " " + x["sol2"]],
                         int(x["label"]), "mc_norm"))
    elif name == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        for x in ds:
            i = x["sentence"].index("_")
            cont = " " + x["sentence"][i + 1:].strip()
            ctxs = [x["sentence"][:i] + x["option1"], x["sentence"][:i] + x["option2"]]
            rows.append((ctxs, [cont, cont], int(x["answer"]) - 1, "wino"))
    elif name == "boolq":
        ds = load_dataset("google/boolq", split="validation")
        for x in ds:
            rows.append((f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:", [" no", " yes"],
                         int(bool(x["answer"])), "mc"))
    elif name == "lambada_openai":
        ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        for x in ds:
            t = x["text"]
            ctx, last = t.rsplit(" ", 1)
            rows.append((ctx, [" " + last], 0, "lambada"))
    else:
        raise ValueError(name)
    return rows


def token_budget_batches(order, lengths, model, max_items=16, logits_bytes=1.5e9):
    """Group length-sorted indices so that the padded logits tensor
    (items x max_len x vocab, bf16) stays under logits_bytes."""
    vocab = vocab_size(model)
    budget = max(int(logits_bytes / (2 * vocab)), 64)  # padded tokens per batch
    batch, longest = [], 0
    for i in order:
        L = lengths[i]
        if batch and (len(batch) >= max_items or max(longest, L) * (len(batch) + 1) > budget):
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, L)
    if batch:
        yield batch


@torch.no_grad()
def _loglik(model, tokenizer, pairs: list[tuple[str, str]], device, chunk=16):
    """For (context, continuation) pairs: (sum log-prob, is_greedy, n_cont_tokens)."""
    enc = []
    for ctx, cont in pairs:
        c = tokenizer(ctx)["input_ids"]
        k = tokenizer(cont, add_special_tokens=False)["input_ids"]
        full = (c + k)[-1024:]
        enc.append((full, len(k)))
    res = [None] * len(enc)
    order = np.argsort([len(e[0]) for e in enc])
    for idx in token_budget_batches(order, [len(e[0]) for e in enc], model, chunk):
        ids, att, lens = pad_batch([enc[i][0] for i in idx], pad_id_of(tokenizer), device)
        logits = model(input_ids=ids, attention_mask=att).logits
        for k, i in enumerate(idx):
            L, nk = len(enc[i][0]), enc[i][1]
            tgt = ids[k, L - nk:L]
            rows = torch.log_softmax(logits[k, L - nk - 1:L - 1].float(), -1)
            s = rows.gather(1, tgt[:, None]).sum().item()
            greedy = bool((rows.argmax(-1) == tgt).all().item())
            res[i] = (s, greedy, nk)
    return res


def zero_shot_eval(model, tokenizer, device="cuda", tasks=None, n_max: int = 2000) -> dict:
    tasks = tasks or ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq",
                      "lambada_openai"]
    model.eval()
    out = {}
    for name in tasks:
        try:
            rows = _load_task(name)
        except Exception as e:  # dataset unavailable: report, never invent
            out[f"{name}_error"] = str(e)[:200]
            continue
        rng = np.random.RandomState(1234)
        if SMOKE:
            rows = rows[:SMOKE_N]
        elif len(rows) > n_max:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), n_max, replace=False))]
        pairs, owners = [], []
        for ri, (ctx, conts, gold, kind) in enumerate(rows):
            for ci, cont in enumerate(conts):
                pairs.append((ctx[ci] if kind == "wino" else ctx, cont))
                owners.append((ri, ci))
        scores = _loglik(model, tokenizer, pairs, device)
        per = {}
        for (ri, ci), sc in zip(owners, scores):
            per.setdefault(ri, {})[ci] = sc
        acc = acc_norm = 0
        for ri, (ctx, conts, gold, kind) in enumerate(rows):
            sc = per[ri]
            if kind == "lambada":
                acc += int(sc[0][1])
                continue
            ll = np.array([sc[c][0] for c in range(len(conts))])
            acc += int(ll.argmax() == gold)
            norm = ll / np.array([max(len(c), 1) for c in conts])
            acc_norm += int(norm.argmax() == gold)
        n = max(len(rows), 1)
        out[f"{name}_n"] = len(rows)
        out[f"{name}_acc"] = acc / n
        if rows and rows[0][3] == "mc_norm":
            out[f"{name}_acc_norm"] = acc_norm / n
    return out
