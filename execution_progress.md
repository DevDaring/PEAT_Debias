# PEAT Pipeline — Execution Progress

**Last updated:** May 1, 2026 — pre-launch (full production run)
**VM:** provider.a100.dsm.val.akash.pub:31133 · A100 SXM4 80GB
**Log:** `/workspace/full_run.log`
**Commit:** `50c9c70` — TextBelt SMS + signal handlers + model doc

---

## Resource Baseline (pre-launch)
| Resource | Total | Free |
|---|---|---|
| Disk | 1.8 TB | 377 GB |
| RAM | 2.0 TB | 1.9 TB |
| GPU VRAM | 80 GB | ~80 GB (fully free at launch) |
| HF Model Cache | — | 29 GB (kept) |

---

## Pipeline Stages Overview

Full run total: **~136 cells** (vs 52 in smoke test)
Seeds: **42, 123, 456** (3 seeds per model/method)

| Stage | Description | Cells | Status |
|---|---|---|---|
| 0 | Dry run | 1 | ⬜ Not started |
| 0b | Dataset preflight | 1 | ⬜ Not started |
| 1 | PEAT training — bert-base | 1 train + 3 eval = 4 | ⬜ Not started |
| 1 | PEAT training — modernbert-base | 1 train + 3 eval = 4 | ⬜ Not started |
| 1 | PEAT training — nomicbert | 1 train + 3 eval = 4 | ⬜ Not started |
| 1 | PEAT training — qwen2.5-1.5b | 1 train + 3 eval = 4 | ⬜ Not started |
| 2 | PEAT scaling — gemma-3-4b | 1 train + 3 eval = 4 | ⬜ Not started |
| 2 | PEAT scaling — llama-3.1-8b | 1 train + 3 eval = 4 | ⬜ Not started |
| 3 | Baselines — bert-base (9 methods × 3 seeds) | 27 | ⬜ Not started |
| 3 | Baselines — modernbert-base (9 methods × 3 seeds) | 27 | ⬜ Not started |
| 3 | Baselines — nomicbert (9 methods × 3 seeds) | 27 | ⬜ Not started |
| 3 | Baselines — qwen2.5-1.5b (9 methods × 3 seeds) | 27 | ⬜ Not started |
| 4 | Aggregation (5 tables) | 1 | ⬜ Not started |
| 5 | Figures (2 PDFs) | 1 | ⬜ Not started |
| 6 | Push results to GitHub | — | ⬜ Not started |

**Total tracked cells: 136**

---

## Baseline Methods (Stage 3)
9 methods applied to all 4 core models × 3 seeds each:
1. `base` — No mitigation (floor)
2. `cda` — Counterfactual Data Augmentation
3. `self_debias` — Self-Debiasing
4. `auto_debias` — Auto-Debias
5. `bias_edit` — BiasEdit *(fixed: NomicBERT inner encoder)*
6. `fair_steer` — FairSteer *(fixed: NomicBERT inner encoder)*
7. `bias_unlearn` — Bias Unlearning
8. `know_bias` — KnowBias *(fixed: NomicBERT inner encoder)*
9. `lora_vanilla_sft` — LoRA Vanilla SFT

---

## Cell Progress Log
*(Updated each time user checks)*

| Checked At | Completed | Failed | Skipped | Active Stage |
|---|---|---|---|---|
| Pre-launch | 0 / 136 | 0 | 0 | — |

---

## SMS Notifications
TextBelt configured. You will receive SMS to PHONE_NO when:
- Pipeline **completes** normally: "PEAT pipeline DONE. X/136 cells..."
- Pipeline **interrupted** (kill/Ctrl+C): "PEAT pipeline was interrupted..."
- Pipeline **crashes**: "PEAT pipeline CRASHED: \<error\>"

---

## How to Monitor
```bash
# Last 100 lines of log (SSH from your machine):
ssh -p 31133 root@provider.a100.dsm.val.akash.pub "tail -100 /workspace/full_run.log"

# Cell completion count:
python akash/check_cells.py

# Live tail (Python script):
python akash/tail_full.py

# Check pipeline is still running:
ssh -p 31133 root@provider.a100.dsm.val.akash.pub "tmux list-sessions"
```

---

## Known Issues Resolved
| Issue | Fix | Commit |
|---|---|---|
| NomicBERT `output_hidden_states` TypeError in bias_edit | `model.bert` inner encoder fallback | `c296c99` |
| NomicBERT `output_hidden_states` TypeError in fair_steer | `model.bert` inner encoder fallback | `c296c99` |
| NomicBERT `output_hidden_states` TypeError in know_bias | `model.bert` inner encoder fallback | `c296c99` |
| accelerate version incompatible with transformers 5.x | Bumped to `>=1.1.0` | `5968cdc` |
| Gemma-3 meta-tensor crash on `.to(device)` | `device_map="cuda:0"` in loader | `c1d15f1` |
| CrowS-Pairs `sample(4)` crash with 2-row smoke test | `sample(min(3, len(df)))` | `9f3065d` |

---

## Notes
- Full run uses **3 seeds** (42, 123, 456) vs smoke test's 1 seed
- SHA grid: **41 configs** × full rounds vs smoke's 2 configs × 1 round
- Expected runtime: **~12–24 hours** on A100 depending on model load times
- Results will be pushed to GitHub automatically at Stage 6
