"""
Layer-wise diagnostics of bias geometry (Reviewer #4 c.4).

For each base model (no adapter) on the StereoSet pairs:
  1. equalisation-gradient profile: norm of d L_neut / d h_l for every hidden
     layer l (all models) and, for models up to 4B parameters, the parameter
     gradient norm per transformer block. Shows where the equaliser's signal
     enters the network and how much of it the last two blocks receive.
  2. linear stereotype direction: d_i = h_l(stereo fill) - h_l(anti fill)
     averaged over the filler tokens of pair i. A ridge classifier without
     intercept separates {d_i} from {-d_i}; 5-fold cross-validated accuracy
     above 0.5 means a single direction orders stereotypical above
     anti-stereotypical completions across held-out pairs.
  3. output-bias readout: cross-validated R^2 of a ridge regression from d_i
     to the base model's output Delta_i (Eq. 1) at every layer.

# Reference: Alain & Bengio, "Understanding intermediate layers using linear
#   classifier probes", ICLR Workshop 2017. arXiv: 1610.01644
# Reference: Belinkov, "Probing Classifiers: Promises, Shortcomings, and
#   Advances", Computational Linguistics 48(1), 2022.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch

from peat.models import get_spec
from peat.r2.common import SMOKE, cap, pad_batch, pad_id_of, word_embedding_module
from peat.r2.fastpeat import pair_forward, prepare_fill, prepare_items, surrogate

_LAYER_RE = re.compile(r"\.(?:layer|layers|h|blocks)\.(\d+)\.")


def gradient_profile(model, tokenizer, model_tag, df, device, n_batches=8, bs=16,
                     param_grads: bool = True) -> pd.DataFrame:
    is_enc = get_spec(model_tag).is_encoder
    model.eval()
    emb = word_embedding_module(model)
    hook = emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    for p in model.parameters():
        p.requires_grad_(param_grads)
    hs_norm, blk_norm = None, {}
    used = 0
    try:
        rows = cap(df, n_batches * bs, seed=7)
        for s in range(0, len(rows), bs):
            ch = rows.iloc[s:s + bs]
            prepared = prepare_items(tokenizer, is_enc, list(zip(ch["context"], ch["t_s"], ch["t_a"])))
            valid = [p for p in prepared if p is not None]
            if not valid:
                continue
            seqs = [p[0][0] for p in valid] + [p[1][0] for p in valid]
            ids, att, _ = pad_batch(seqs, pad_id_of(tokenizer), device)
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
            hidden = out.hidden_states
            # Reuse the training-format Delta on the same forward pass.
            logits = out.logits
            B = len(valid)
            r_, c_, t_ = [], [], []
            for j, fill in enumerate([p[0] for p in valid] + [p[1] for p in valid]):
                seq, pos, tg = fill
                for p_, t in zip(pos, tg):
                    if p_ < len(seq):
                        r_.append(j); c_.append(p_ if p_ >= 0 else p_ + len(seq)); t_.append(t)
            r_t = torch.tensor(r_, device=device)
            lp = torch.log_softmax(logits[r_t, torch.tensor(c_, device=device)].float(), -1)
            lp = lp.gather(1, torch.tensor(t_, device=device)[:, None]).squeeze(1)
            sums = torch.zeros(2 * B, device=device).index_add(0, r_t, lp)
            ns = torch.tensor([max(len(p[0][2]), 1) for p in valid], device=device, dtype=torch.float32)
            na = torch.tensor([max(len(p[1][2]), 1) for p in valid], device=device, dtype=torch.float32)
            L = surrogate(sums[:B] / ns - sums[B:] / na, "huber").mean()
            grads = torch.autograd.grad(L, list(hidden), retain_graph=param_grads, allow_unused=True)
            g = np.array([0.0 if x is None else x.float().norm().item() for x in grads])
            hs_norm = g if hs_norm is None else hs_norm + g
            if param_grads:
                model.zero_grad(set_to_none=True)
                L.backward()
                for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    m = _LAYER_RE.search(n)
                    key = int(m.group(1)) if m else -1
                    blk_norm[key] = blk_norm.get(key, 0.0) + p.grad.float().pow(2).sum().item()
                model.zero_grad(set_to_none=True)
            used += 1
    finally:
        hook.remove()
        for p in model.parameters():
            p.requires_grad_(False)
    rows_out = [{"kind": "hidden", "layer": l, "grad_norm": v / max(used, 1)}
                for l, v in enumerate(hs_norm if hs_norm is not None else [])]
    rows_out += [{"kind": "block_param", "layer": k, "grad_norm": (v / max(used, 1)) ** 0.5}
                 for k, v in sorted(blk_norm.items())]
    return pd.DataFrame(rows_out)


@torch.no_grad()
def filler_representations(model, tokenizer, model_tag, df, device, bs=16):
    """Per pair and layer: mean hidden state over the filler tokens of the
    *unmasked* filled sentence, for the stereotypical and the anti fill."""
    model.eval()
    H_s, H_a, keep = [], [], []
    for s in range(0, len(df), bs):
        ch = df.iloc[s:s + bs]
        seqs, spans = [], []
        for k, r in enumerate(ch.itertuples()):
            ts = tokenizer(r.context.replace("BLANK", r.t_s), truncation=True, max_length=512)["input_ids"]
            ta = tokenizer(r.context.replace("BLANK", r.t_a), truncation=True, max_length=512)["input_ids"]
            cp = 0
            while cp < min(len(ts), len(ta)) and ts[cp] == ta[cp]:
                cp += 1
            cs = 0
            while cs < min(len(ts), len(ta)) - cp and ts[-1 - cs] == ta[-1 - cs]:
                cs += 1
            if len(ts) - cs <= cp or len(ta) - cs <= cp:
                continue  # identical fills: no filler span
            seqs += [ts, ta]
            spans += [(cp, len(ts) - cs), (cp, len(ta) - cs)]
            keep.append(s + k)
        if not seqs:
            continue
        ids, att, _ = pad_batch(seqs, pad_id_of(tokenizer), device)
        hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
        reps = torch.stack([torch.stack([h[i, a:b].float().mean(0) for i, (a, b) in enumerate(spans)])
                            for h in hs])  # (layers, 2*len, d)
        reps = reps.cpu().half()
        H_s.append(reps[:, 0::2]); H_a.append(reps[:, 1::2])
    return torch.cat(H_s, 1), torch.cat(H_a, 1), keep


def _ridge_dual(Xtr, ytr, Xte, lam_rel=1e-2):
    K = Xtr @ Xtr.T
    lam = lam_rel * torch.trace(K) / K.shape[0]
    alpha = torch.linalg.solve(K + lam * torch.eye(K.shape[0], device=K.device), ytr)
    return (Xte @ Xtr.T) @ alpha


def probe_layers(H_s, H_a, delta0: np.ndarray, device, folds=5, seed=0) -> pd.DataFrame:
    n = H_s.shape[1]
    folds = max(2, min(folds, n))
    rng = np.random.RandomState(seed)
    fold_of = rng.permutation(n) % folds
    y_delta = torch.tensor(delta0, dtype=torch.float32, device=device)
    rows = []
    for l in range(H_s.shape[0]):
        D = (H_s[l] - H_a[l]).float().to(device)
        D = D / (D.norm(dim=1, keepdim=True).mean() + 1e-8)
        acc, ss_res, ss_tot = [], 0.0, 0.0
        for f in range(folds):
            tr = torch.tensor(np.where(fold_of != f)[0], device=device)
            te = torch.tensor(np.where(fold_of == f)[0], device=device)
            Xtr = torch.cat([D[tr], -D[tr]])
            ytr = torch.cat([torch.ones(len(tr), device=device), -torch.ones(len(tr), device=device)])
            pred = _ridge_dual(Xtr, ytr, D[te])
            acc.append((pred > 0).float().mean().item())
            yt = y_delta[tr]
            mu = yt.mean()
            pr = _ridge_dual(D[tr], yt - mu, D[te]) + mu
            ss_res += ((y_delta[te] - pr) ** 2).sum().item()
            ss_tot += ((y_delta[te] - mu) ** 2).sum().item()
        mean_dir = D.mean(0)
        cos = torch.nn.functional.cosine_similarity(D, mean_dir[None], dim=1).mean().item()
        rows.append({"layer": l, "direction_acc": float(np.mean(acc)),
                     "delta_r2": 1 - ss_res / max(ss_tot, 1e-12), "mean_cos_to_mean_dir": cos})
    return pd.DataFrame(rows)
