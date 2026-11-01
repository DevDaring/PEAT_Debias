"""
Batched PEAT objective and trainer (revision 2).

Mathematically identical to peat.peat.compute_peat_loss / run_final_training:
the same Delta (Eq. 1), pair-mass anchor (Eq. 6), complement-vocabulary KL
(Eq. 7), Huber equaliser (Eqs. 3-4), AdamW (lr 1e-4, wd 0.01), cosine
schedule, gradient clipping at 1.0, batch 64 (encoders) / 16 (decoders),
5 epochs. Differences are purely computational: the whole mini-batch is
scored in one padded forward pass instead of one forward per item, the frozen
reference theta_0 is the same network with the adapter disabled (LoRA-B = 0 at
init, so the two are identical), and log-softmax is taken in float32.
tests/test_r2_equivalence.py checks the batched loss against the original
per-item loss.

Revision-2 additions:
  * alternative surrogates for the equaliser (Reviewer #4, comment 1;
    Reviewer #5, weakness 1): squared, absolute, symmetric logistic
    (log-cosh form), one-sided logistic and one-sided hinge;
  * per-term loss and gradient-norm logging with pairwise gradient cosines
    (Reviewer #4, comment 8);
  * per-pair Delta dumps before and after training for the sign/magnitude
    analysis (Reviewer #5, weakness 1).

# Implements PEAT [Deb & Basu, APIN-D-26-06244, Sec. 3].
# Reference: Huber, "Robust Estimation of a Location Parameter", Ann. Math.
# Statist. 35(1), 1964. | Used here for: equaliser and its median/mean link.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from peat.models import get_spec
from peat.r2.common import SMOKE, pad_batch, pad_id_of
from peat.utils import get_autocast_dtype, set_seed

SURROGATES = ("huber", "l2", "l1", "logcosh", "logistic", "hinge")
LOG2 = math.log(2.0)


def surrogate(d: torch.Tensor, kind: str, tau: float = 1.0) -> torch.Tensor:
    """Per-pair equaliser penalty on Delta.

    huber    : PEAT (symmetric, quadratic near 0, linear tails)
    l2       : squared log-odds (Huber with tau -> inf)
    l1       : absolute log-odds (Huber with tau -> 0); its location minimiser
               is the median, the statistic that fixes SS = 50
    logcosh  : symmetric logistic, softplus(d) + softplus(-d) - 2 log 2
    logistic : one-sided logistic, softplus(d), the classical smooth surrogate
               of the indicator 1[d > 0]; pushes d toward -inf
    hinge    : one-sided hinge, max(0, d); pushes d to <= 0 and then stops
    """
    if kind == "huber":
        a = d.abs()
        return torch.where(a <= tau, d * d, 2 * tau * a - tau * tau)
    if kind == "l2":
        return d * d
    if kind == "l1":
        return d.abs()
    if kind == "logcosh":
        return F.softplus(d) + F.softplus(-d) - 2 * LOG2
    if kind == "logistic":
        return F.softplus(d)
    if kind == "hinge":
        return F.relu(d)
    raise ValueError(f"unknown surrogate {kind!r}")


# ---------------------------------------------------------------------------
# Tokenisation of one (context, filler) completion — identical rules to
# peat.peat._get_masked_logits_encoder / _get_causal_logits
# ---------------------------------------------------------------------------

def prepare_fill(tokenizer, is_encoder: bool, context: str, filler: str):
    """Return (input_ids, positions, target_ids) or None if unusable."""
    if is_encoder:
        fids = tokenizer.encode(filler, add_special_tokens=False)
        n = len(fids)
        if n == 0:
            return None
        text = context.replace("BLANK", " ".join([tokenizer.mask_token] * n))
        ids = tokenizer(text, truncation=True, max_length=512)["input_ids"]
        mpos = [i for i, t in enumerate(ids) if t == tokenizer.mask_token_id][:n]
        if len(mpos) < n:
            return None
        return ids, mpos, fids
    full = context.replace("BLANK", filler)
    ids = tokenizer(full, truncation=True, max_length=1024)["input_ids"]
    ftoks = tokenizer.encode(filler, add_special_tokens=False)
    if not ftoks:
        return None
    prefix = context.split("BLANK")[0].rstrip()
    start = len(tokenizer.encode(prefix, add_special_tokens=True)) - 1
    positions = list(range(start, start + len(ftoks)))
    targets = []
    for i, p in enumerate(positions):
        nxt = p + 1
        targets.append(ids[nxt] if nxt < len(ids) else ftoks[i])
    return ids, positions, targets


def _resolve(p: int, length: int) -> int:
    # The original indexed logits[p] on an unpadded row, so p = -1 (sentence-
    # initial BLANK on tokenisers without BOS) meant the last real position.
    return p if p >= 0 else p + length


def prepare_items(tokenizer, is_encoder: bool, items: list[tuple]):
    """items: list of (context, t_s, t_a). Returns per-item prepared fills."""
    prepared = []
    for ctx, t_s, t_a in items:
        s = prepare_fill(tokenizer, is_encoder, ctx, t_s)
        a = prepare_fill(tokenizer, is_encoder, ctx, t_a)
        prepared.append((s, a) if (s is not None and a is not None) else None)
    return prepared


def pair_forward(model, tokenizer, prepared, device, want_kl_logits: bool = True):
    """One padded forward over all stereo and anti fills of the valid items.

    Returns dict of tensors over the valid items (in order): lp_s, lp_a (summed
    log-probabilities), n_s, n_a, kl_logits (logits at the first stereo target
    position, or None), kl_ok (bool mask), mask_ids (list of lists).
    """
    valid = [p for p in prepared if p is not None]
    B = len(valid)
    seqs = [p[0][0] for p in valid] + [p[1][0] for p in valid]
    ids, att, lens = pad_batch(seqs, pad_id_of(tokenizer), device)
    logits = model(input_ids=ids, attention_mask=att).logits

    rows, cols, tgts, owner = [], [], [], []
    for j, (fill) in enumerate([p[0] for p in valid] + [p[1] for p in valid]):
        seq, pos, tg = fill
        L = len(seq)
        for p, t in zip(pos, tg):
            if p < L:
                rows.append(j)
                cols.append(_resolve(p, L))
                tgts.append(t)
    rows_t = torch.tensor(rows, device=device, dtype=torch.long)
    cols_t = torch.tensor(cols, device=device, dtype=torch.long)
    tg_t = torch.tensor(tgts, device=device, dtype=torch.long)
    sel = logits[rows_t, cols_t].float()
    lp = torch.log_softmax(sel, dim=-1).gather(1, tg_t[:, None]).squeeze(1)
    sums = torch.zeros(2 * B, device=device, dtype=torch.float32).index_add(0, rows_t, lp)

    n_s = torch.tensor([max(len(p[0][2]), 1) for p in valid], device=device, dtype=torch.float32)
    n_a = torch.tensor([max(len(p[1][2]), 1) for p in valid], device=device, dtype=torch.float32)
    out = {"lp_s": sums[:B], "lp_a": sums[B:], "n_s": n_s, "n_a": n_a}

    if want_kl_logits:
        ref_rows, ref_cols, ok = [], [], []
        for j, p in enumerate(valid):
            seq, pos, _ = p[0]
            r = pos[0] if len(pos) > 0 else 0
            ok.append(r < len(seq))
            ref_rows.append(j)
            ref_cols.append(_resolve(r, len(seq)) if r < len(seq) else 0)
        out["kl_logits"] = logits[torch.tensor(ref_rows, device=device),
                                  torch.tensor(ref_cols, device=device)]
        out["kl_ok"] = torch.tensor(ok, device=device)
        out["mask_ids"] = [sorted(set(p[0][2]) | set(p[1][2])) for p in valid]
    return out


def complement_kl(logits_theta, logits_theta0, mask_ids: list[list[int]]):
    """KL(P~_theta0 || P~_theta) over V \\ {t_s, t_a} for each item (Eq. 7)."""
    p = torch.softmax(logits_theta.float(), dim=-1)
    p0 = torch.softmax(logits_theta0.float(), dim=-1)
    keep = torch.ones_like(p)
    for i, mids in enumerate(mask_ids):
        keep[i, mids] = 0.0
    q = p * keep
    q = q / (q.sum(-1, keepdim=True) + 1e-30)
    q0 = p0 * keep
    q0 = q0 / (q0.sum(-1, keepdim=True) + 1e-30)
    return (torch.xlogy(q0, q0) - q0 * torch.log(q + 1e-30)).sum(-1)


def reference_forward(model, tokenizer, prepared, device):
    """Frozen theta_0 pass: same network, adapter disabled, eval mode."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    out0 = pair_forward(model, tokenizer, prepared, device)
            else:
                out0 = pair_forward(model, tokenizer, prepared, device)
    finally:
        if was_training:
            model.train()
    return out0


def peat_loss_terms(model, tokenizer, model_tag, items, device, surrogate_kind="huber", tau=1.0):
    """Return (L_neut, L_pair, L_kl, delta) for a mini-batch of (ctx, t_s, t_a)."""
    is_enc = get_spec(model_tag).is_encoder
    prepared = prepare_items(tokenizer, is_enc, items)
    if not any(p is not None for p in prepared):
        return None
    out = pair_forward(model, tokenizer, prepared, device)
    out0 = reference_forward(model, tokenizer, prepared, device)

    delta = out["lp_s"] / out["n_s"] - out["lp_a"] / out["n_a"]
    L_neut = surrogate(delta, surrogate_kind, tau).mean()

    # Same expression as the original implementation, log(e^a + e^b + 1e-30):
    # for very unlikely multi-token pairs both exponentials underflow and the
    # anchor saturates at log(1e-30), i.e. it is inactive on those pairs.
    mu = torch.log(torch.exp(out["lp_s"]) + torch.exp(out["lp_a"]) + 1e-30)
    mu0 = torch.log(torch.exp(out0["lp_s"]) + torch.exp(out0["lp_a"]) + 1e-30)
    L_pair = ((mu - mu0.detach()) ** 2).mean()

    kl = complement_kl(out["kl_logits"], out0["kl_logits"], out["mask_ids"])
    kl_ok = out["kl_ok"]
    L_kl = kl[kl_ok].mean() if bool(kl_ok.any()) else torch.zeros((), device=device)
    return L_neut, L_pair, L_kl, delta


# ---------------------------------------------------------------------------
# Delta tables (no grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def delta_table(model, tokenizer, model_tag, df: pd.DataFrame, device, batch_size=32,
                with_reference: bool = True) -> pd.DataFrame:
    """Per-pair length-normalised log-probabilities and Delta in the training
    (StereoSet cloze) format, under theta and (optionally) theta_0."""
    is_enc = get_spec(model_tag).is_encoder
    model.eval()
    recs = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        items = list(zip(chunk["context"], chunk["t_s"], chunk["t_a"]))
        prepared = prepare_items(tokenizer, is_enc, items)
        if not any(p is not None for p in prepared):
            continue
        out = pair_forward(model, tokenizer, prepared, device, want_kl_logits=False)
        if with_reference and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                out0 = pair_forward(model, tokenizer, prepared, device, want_kl_logits=False)
        else:
            out0 = out
        k = 0
        for (i, row), p in zip(chunk.iterrows(), prepared):
            if p is None:
                continue
            ls, la = out["lp_s"][k].item(), out["lp_a"][k].item()
            ns, na = out["n_s"][k].item(), out["n_a"][k].item()
            ls0, la0 = out0["lp_s"][k].item(), out0["lp_a"][k].item()
            recs.append({
                "row": i, "bias_type": row.get("bias_type", ""),
                "lp_s": ls / ns, "lp_a": la / na, "delta": ls / ns - la / na,
                "mu": float(np.logaddexp(ls, la)),
                "delta0": ls0 / ns - la0 / na, "mu0": float(np.logaddexp(ls0, la0)),
            })
            k += 1
    return pd.DataFrame(recs)


def summarise_delta(d: np.ndarray, tau: float = 1.0) -> dict:
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return {}
    a = np.abs(d)
    hub = np.where(a <= tau, d * d, 2 * tau * a - tau * tau)
    return {
        "n": int(d.size), "ss": 100.0 * float((d > 0).mean()), "mean": float(d.mean()),
        "median": float(np.median(d)), "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "mean_abs": float(a.mean()), "huber": float(hub.mean()),
        "frac_abs_lt_0.1": float((a < 0.1).mean()), "frac_abs_lt_0.5": float((a < 0.5).mean()),
        "frac_abs_lt_1": float((a < 1.0).mean()),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _grad_vector(grads, params):
    flat = [g.reshape(-1).float() if g is not None else torch.zeros(p.numel(), device=p.device)
            for g, p in zip(grads, params)]
    return torch.cat(flat)


def _cos(a, b):
    na, nb = a.norm(), b.norm()
    if na.item() == 0 or nb.item() == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def train_peat(model, tokenizer, model_tag: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
               cfg: dict, seed: int, device: str = "cuda", log_every: int = 10) -> dict:
    """Train a LoRA-wrapped model with the PEAT objective.

    cfg keys: lambda_1, lambda_2, surrogate, tau, lr, epochs, batch_size.
    Returns {"step_log": DataFrame, "epoch_log": DataFrame, "seconds": float}.
    """
    spec = get_spec(model_tag)
    set_seed(seed)
    lam1, lam2 = float(cfg["lambda_1"]), float(cfg["lambda_2"])
    kind, tau = cfg.get("surrogate", "huber"), float(cfg.get("tau", 1.0))
    epochs = 1 if SMOKE else int(cfg.get("epochs", 5))
    bs = int(cfg.get("batch_size", 64 if spec.is_encoder else 16))

    items = list(zip(train_df["context"], train_df["t_s"], train_df["t_a"]))
    loader = DataLoader(items, batch_size=bs, shuffle=True, collate_fn=lambda b: b)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=float(cfg.get("lr", 1e-4)), weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=max(len(loader) * epochs, 1))
    cast = get_autocast_dtype()

    step_rows, epoch_rows = [], []
    t0 = time.time()
    gstep = 0
    # The published configuration trains theta in train mode, so encoders
    # with dropout (BERT 0.1, NomicBERT 0.1) score theta with dropout noise
    # while the reference theta_0 is scored without it. cfg["no_dropout"]
    # removes that noise (diagnostic ablation for Reviewer #4, c.4 and c.8).
    no_dropout = bool(cfg.get("no_dropout", False))
    for epoch in range(1, epochs + 1):
        model.eval() if no_dropout else model.train()
        for batch in loader:
            with torch.amp.autocast("cuda", dtype=cast, enabled=torch.cuda.is_available()):
                terms = peat_loss_terms(model, tokenizer, model_tag, batch, device, kind, tau)
            if terms is None:
                continue
            L_neut, L_pair, L_kl, delta = terms
            total = L_neut + lam1 * L_pair + lam2 * L_kl
            if not torch.isfinite(total):
                opt.zero_grad(set_to_none=True)
                continue
            row = {"epoch": epoch, "step": gstep, "L_neut": L_neut.item(), "L_pair": L_pair.item(),
                   "L_kl": L_kl.item(), "L_total": total.item(),
                   "w_pair": lam1 * L_pair.item(), "w_kl": lam2 * L_kl.item(),
                   "mean_abs_delta": delta.detach().abs().mean().item()}
            if gstep % log_every == 0:
                # Separate gradients of the three weighted terms (Reviewer #4, c.8).
                g = []
                for term in (L_neut, lam1 * L_pair, lam2 * L_kl):
                    if term.requires_grad:
                        gr = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                    else:
                        gr = [None] * len(params)
                    g.append(_grad_vector(gr, params))
                gt = g[0] + g[1] + g[2]
                row.update({"g_neut": g[0].norm().item(), "g_pair": g[1].norm().item(),
                            "g_kl": g[2].norm().item(), "g_total": gt.norm().item(),
                            "cos_neut_pair": _cos(g[0], g[1]), "cos_neut_kl": _cos(g[0], g[2]),
                            "cos_pair_kl": _cos(g[1], g[2])})
                offset = 0
                for p in params:
                    n = p.numel()
                    p.grad = gt[offset:offset + n].view_as(p).to(p.dtype)
                    offset += n
            else:
                total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step_rows.append(row)
            gstep += 1

        vt = delta_table(model, tokenizer, model_tag, val_df, device, with_reference=False)
        ev = summarise_delta(vt["delta"].values, tau) if len(vt) else {}
        ep = {"epoch": epoch, "seconds": time.time() - t0}
        ep.update({f"val_{k}": v for k, v in ev.items()})
        last = [r for r in step_rows if r["epoch"] == epoch]
        for k in ("L_neut", "L_pair", "L_kl", "L_total", "w_pair", "w_kl"):
            ep[f"train_{k}"] = float(np.mean([r[k] for r in last])) if last else float("nan")
        epoch_rows.append(ep)
    return {"step_log": pd.DataFrame(step_rows), "epoch_log": pd.DataFrame(epoch_rows),
            "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# LoRA-Vanilla-SFT trainer (same objective as peat.baselines.lora_vanilla_sft)
# ---------------------------------------------------------------------------

def train_lora_vanilla(model, tokenizer, model_tag: str, train_df: pd.DataFrame, seed: int,
                       device: str = "cuda", n_epochs: int = 5) -> dict:
    """Language-modelling cross-entropy over both completed sentences.

    Mirrors peat.baselines.lora_vanilla_sft.train_mlm_sft: labels = input_ids
    on the unmasked sentence (causal LM loss for decoders; token-reconstruction
    loss for encoders), batch 32/8, gradient accumulation 1/4, lr 1e-4.
    """
    from tqdm import tqdm

    spec = get_spec(model_tag)
    set_seed(seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=1e-4, weight_decay=0.01)
    bs = 32 if spec.is_encoder else 8
    items = list(zip(train_df["context"], train_df["t_s"], train_df["t_a"]))
    loader = DataLoader(items, batch_size=bs, shuffle=True, collate_fn=lambda b: b)
    if SMOKE:
        n_epochs = 1
    sched = CosineAnnealingLR(opt, T_max=max(len(loader) * n_epochs, 1))
    cast = get_autocast_dtype()
    accum = 1 if spec.is_encoder else 4
    t0, rows = time.time(), []
    for epoch in range(1, n_epochs + 1):
        tot, n = 0.0, 0
        for step, batch in enumerate(tqdm(loader, desc=f"SFT epoch {epoch}", disable=True)):
            loss_sum = torch.zeros((), device=device)
            for ctx, t_s, t_a in batch:
                for filler in (t_s, t_a):
                    enc = tokenizer(ctx.replace("BLANK", filler), return_tensors="pt",
                                    truncation=True, max_length=512).to(device)
                    enc["labels"] = enc["input_ids"].clone()
                    with torch.amp.autocast("cuda", dtype=cast, enabled=torch.cuda.is_available()):
                        o = model(**enc)
                    if o.loss is not None and torch.isfinite(o.loss):
                        loss_sum = loss_sum + o.loss
            loss = loss_sum / (len(batch) * 2 * accum)
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item()
            n += 1
        rows.append({"epoch": epoch, "loss": tot / max(n, 1), "seconds": time.time() - t0})
    return {"epoch_log": pd.DataFrame(rows), "seconds": time.time() - t0}
