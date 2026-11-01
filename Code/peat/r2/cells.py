"""
Execute one revision-2 cell: load the model, train (PEAT / LoRA-Vanilla-SFT)
if the cell asks for it, run the requested evaluations, and write every
artefact incrementally under results/r2/cells/<cell_id>/. The cell is marked
complete only when meta.json carries status "done".
"""

from __future__ import annotations

import json
import time
import traceback

import pandas as pd
import torch

from peat.data import load_stereoset_pairs
from peat.models import get_spec, load_model
from peat.r2 import fastpeat
from peat.r2.common import (SMOKE, attach_lora_ext, cap, cell_dir, count_params, env_info,
                            save_adapter, write_json)


def _done(d, key):
    return (d / f"{key}.done").exists()


def _mark(d, key):
    (d / f"{key}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def _train_data(cfg):
    train_df, val_df = load_stereoset_pairs(seed=42)
    if cfg.get("data") == "cb":
        from peat.cb_pairs import load_cb_augmented_pairs
        train_df = load_cb_augmented_pairs(seed=42)
    return train_df, val_df


def run_cell(cell: dict, device: str = "cuda") -> dict:
    d = cell_dir(cell["id"])
    meta_path = d / "meta.json"
    meta = {"cell": cell, "status": "running", "env": env_info(), "started": time.time(),
            "smoke": SMOKE}
    write_json(meta_path, meta)
    tag = cell["model"]
    spec = get_spec(tag)
    summary = {}
    try:
        model, tok, mcfg = load_model(tag, device=device)
        meta["model_config"] = mcfg
        train_full, val_full = _train_data(cell["cfg"]) if cell["kind"] in ("peat", "lora") else \
            load_stereoset_pairs(seed=42)
        train_df, val_df = cap(train_full), cap(val_full)

        if cell["kind"] in ("peat", "lora"):
            cfg = cell["cfg"]
            model = attach_lora_ext(model, tag, rank=int(cfg.get("rank", 4)),
                                    placement=cfg.get("placement", "last2"))
            meta["params"] = count_params(model)
            if SMOKE and cell["kind"] == "peat":
                # Memory probe: one full-size batch, so OOM shows up in the smoke test.
                bs = 64 if spec.is_encoder else 16
                probe = list(zip(*[train_full.head(bs)[c] for c in ("context", "t_s", "t_a")]))
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    t = fastpeat.peat_loss_terms(model, tok, tag, probe, device, cfg["surrogate"])
                (t[0] + cfg["lambda_1"] * t[1] + cfg["lambda_2"] * t[2]).backward()
                model.zero_grad(set_to_none=True)
                meta["smoke_mem_probe_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 \
                    if torch.cuda.is_available() else 0.0
            if cell["kind"] == "peat":
                tr = fastpeat.train_peat(model, tok, tag, train_df, val_df, cfg, cell["seed"], device)
                tr["step_log"].to_csv(d / "train_steps.csv", index=False)
                tr["epoch_log"].to_csv(d / "train_epochs.csv", index=False)
            else:
                tr = fastpeat.train_lora_vanilla(model, tok, tag, train_df, cell["seed"], device)
                tr["epoch_log"].to_csv(d / "train_epochs.csv", index=False)
            meta["train_seconds"] = tr["seconds"]
            meta["adapter"] = str(save_adapter(model, cell["id"]).name)
            write_json(meta_path, meta)
        model.eval()

        if cell["kind"] == "diag":
            summary.update(_run_diag(model, tok, tag, train_full, val_full, d, device))

        for ev in cell["evals"]:
            if _done(d, ev):
                continue
            t0 = time.time()
            res = _run_eval(ev, model, tok, tag, train_df, val_df, d, device)
            summary.update(res)
            write_json(d / f"summary_{ev}.json", res)
            _mark(d, ev)
            meta.setdefault("eval_seconds", {})[ev] = time.time() - t0
            write_json(meta_path, meta)

        for f in d.glob("summary_*.json"):
            summary.update(json.loads(f.read_text()))
        meta.update({"status": "done", "finished": time.time(), "summary": summary,
                     "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0})
        write_json(meta_path, meta)
        return meta
    except Exception as e:
        meta.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        write_json(meta_path, meta)
        raise


def _run_eval(ev, model, tok, tag, train_df, val_df, d, device) -> dict:
    spec = get_spec(tag)
    if ev == "crows_en":
        from peat.r2.scoring import run_benchmark
        res, s = run_benchmark(model, tok, tag, "crows_en", device)
        res.to_csv(d / "crows_en.csv", index=False)
        return {f"crows_en_{k}": v for k, v in s.items()}
    if ev == "delta":
        out = {}
        for name, df in (("train", train_df), ("val", val_df)):
            t = fastpeat.delta_table(model, tok, tag, df, device)
            t.to_csv(d / f"delta_{name}.csv", index=False)
            for k, v in fastpeat.summarise_delta(t["delta"].values).items():
                out[f"delta_{name}_{k}"] = v
            for k, v in fastpeat.summarise_delta(t["delta0"].values).items():
                out[f"delta0_{name}_{k}"] = v
            out[f"pairmass_{name}_mean_abs_shift"] = float((t["mu"] - t["mu0"]).abs().mean())
        return out
    if ev == "cloze":
        # CrowS-derived content scored in the training format (Eq. 1): the
        # probability of the group word at the masked slot. Separates the
        # objective from the sentence-level CrowS measurement (encoders score
        # P(context | group word) there, the reverse direction).
        from peat.r2.scoring import load_multi_crows_cloze
        out = {}
        langs = ["english"] + (["hindi", "bengali"] if not spec.is_encoder else [])
        for lang in langs:
            df = cap(load_multi_crows_cloze(lang))
            t = fastpeat.delta_table(model, tok, tag, df, device)
            t = t.merge(df[["bias_type"]].rename_axis("row").reset_index().rename(
                columns={"bias_type": "bt"}), on="row", how="left")
            t.to_csv(d / f"cloze_{lang}.csv", index=False)
            for k, v in fastpeat.summarise_delta(t["delta"].values).items():
                out[f"cloze_{lang}_{k}"] = v
            for k, v in fastpeat.summarise_delta(t["delta0"].values).items():
                out[f"cloze0_{lang}_{k}"] = v
        return out
    if ev == "utility":
        from peat.r2 import utility
        return utility.encoder_pppl_kl(model, tok, device) if spec.is_encoder else \
            utility.decoder_ppl_kl(model, tok, device)
    if ev == "multi":
        from peat.r2.scoring import run_benchmark
        out = {}
        for name in ("crows_fr", "mcrows_hindi", "mcrows_bengali", "mcrows_english"):
            res, s = run_benchmark(model, tok, tag, name, device)
            res.to_csv(d / f"{name}.csv", index=False)
            out.update({f"{name}_{k}": v for k, v in s.items()})
        return out
    if ev == "bbq":
        from peat.r2.extrinsic2 import evaluate_bbq
        res, s = evaluate_bbq(model, tok, device)
        res.to_csv(d / "bbq.csv", index=False)
        return s
    if ev == "bold":
        from peat.r2.extrinsic2 import evaluate_bold
        res, s = evaluate_bold(model, tok, device)
        res.to_csv(d / "bold.csv", index=False)
        return s
    if ev == "zeroshot":
        from peat.r2.utility import zero_shot_eval
        return zero_shot_eval(model, tok, device)
    raise ValueError(ev)


def _run_diag(model, tok, tag, train_full, val_full, d, device) -> dict:
    from peat.r2 import diagnostics as dg
    pairs = pd.concat([train_full, val_full], ignore_index=True)
    pairs = cap(pairs)
    n_params = sum(p.numel() for p in model.parameters())
    gp = dg.gradient_profile(model, tok, tag, pairs, device, param_grads=n_params < 2e9)
    gp.to_csv(d / "grad_profile.csv", index=False)
    H_s, H_a, keep = dg.filler_representations(model, tok, tag, pairs, device)
    sub = pairs.iloc[keep].reset_index(drop=True)
    dt = fastpeat.delta_table(model, tok, tag, sub, device, with_reference=False)
    idx = torch.tensor(dt["row"].values, dtype=torch.long)
    n = len(idx)
    pr = dg.probe_layers(H_s[:, idx], H_a[:, idx], dt["delta"].values, device)
    pr.to_csv(d / "probe_layers.csv", index=False)
    return {"diag_layers": int(H_s.shape[0]), "diag_pairs": int(n)}
