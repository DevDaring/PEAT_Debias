# PEAT — Probability-Equalized Adapter Tuning for Bias Mitigation in Language Models

PEAT trains a parameter-efficient LoRA adapter that equalizes the probability of stereotypical and anti-stereotypical token completions at masked positions, without degrading language-modeling utility. It is evaluated against nine comparative methods (including five recent SOTA bias mitigation approaches) on six language models (three encoders, three causal decoders), under a single fixed compute and precision configuration.

## Latest Results (2026-07 major-revision run)

The full revision pipeline completed on a single A100-80GB: **198 cells, 0 failures**, all six PEAT models reproducing published stereotype scores (1,508 CrowS-Pairs pairs each, deduplicated by `idx`). Every baseline intervention is now kept **active during scoring** (the corrected evaluation), and significance is measured across the 1,508 paired outcomes rather than across seeds.

Headline findings (honest scope):

- **Qwen2.5-1.5B:** PEAT reaches SS = 53.42 (5 seeds), **2.91 points below parameter-matched LoRA fine-tuning**; a McNemar test over pairs confirms the reduction (Holm-adjusted *p* = 0.037), and it transfers to a held-out StereoSet split (63.98 → 59.24).
- **Utility:** PEAT preserves GLUE better than vanilla LoRA on the encoders (BERT 0.604 vs 0.567).
- **Scope limits (reported honestly):** on the three encoders the PEAT-vs-LoRA advantage is not statistically reliable; the inference-time Self-Debias reaches lower intrinsic scores on every model, at a per-query cost; coverage-balanced training (PEAT-CB) lowers disability bias on the decoder (75.0 → 70.0) but not on the encoder.

Paper-ready tables are written to `Code/results/`: `CLEAN_SUMMARY.md`, `clean_ss_summary.csv`, `table_wpB_significance.csv` (PEAT vs Base), and `table_wpB_peat_vs_lora.csv` (PEAT vs LoRA, McNemar + Holm).

## Hardware Requirements

- **GPU**: Single NVIDIA A100-40GB or A6000-48GB
- **Disk**: ~150 GB free space
- **RAM**: 32+ GB system RAM
- **Estimated runtime**: ~200 GPU-hours total

## OS Requirements

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.12 (required for Flash-Attention 2.8.3 wheel)
- **CUDA**: 12.4

> ⚠️ **Flash-Attention 2.8.3** wheel only works for `cu12/torch2.5/cp312` on **Linux x86_64**. It will not install on Windows or macOS.

## Setup

### Step 1: Configure secrets
```bash
cp .env.template .env
# Edit .env and fill in your API keys
```

### Step 2: Install dependencies
```bash
bash install.sh
```

### Step 3: Run the pipeline
```bash
python3 run_all.py
```

That's it. The pipeline runs end-to-end without human intervention.

## What Runs

### Models

| Tag | HuggingFace ID | Type | Parameters |
|-----|----------------|------|------------|
| `bert-base` | `google-bert/bert-base-uncased` | Encoder MLM | 110M |
| `modernbert-base` | `answerdotai/ModernBERT-base` | Encoder MLM | 150M |
| `nomicbert` | `nomic-ai/nomic-bert-2048` | Encoder MLM | 137M |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | Causal LM | 1.5B |
| `gemma-3-4b` | `google/gemma-3-4b-it` | Causal LM | 4B |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | Causal LM | 3B |

### Methods

1. **Base** — No mitigation (floor)
2. **CDA** — Counterfactual Data Augmentation (Zmigrod et al., ACL 2019)
3. **Self-Debias** — Inference-time decoding modification (Schick et al., TACL 2021)
4. **Auto-Debias** — JS-divergence debiasing (Guo et al., ACL 2022)
5. **BiasEdit** — Lightweight editor networks (Xu et al., TrustNLP@NAACL 2025)
6. **FairSteer** — Activation steering (Li et al., ACL 2025 Findings)
7. **BiasUnlearn** — Dual-pathway unlearning (Liu et al., EMNLP 2025)
8. **KnowBias** — Bias-neuron enhancement (Pan et al., arXiv 2601.21864, 2026)
9. **LoRA + Vanilla SFT** — Internal ablation (same adapter, standard loss)
10. **PEAT** — Ours

### Datasets

- **StereoSet** (Nadeem et al., 2021) — Training source
- **CrowS-Pairs** (Nangia et al., 2020) — Primary test set (1,508 pairs, 9 bias categories)
- **BBQ** (Parrish et al., 2022) — Extrinsic sanity check (causal LMs only)
- **GLUE** (Wang et al., 2018) — Utility evaluation for encoders (8 tasks)
- **WikiText-103** (Merity et al., 2016) — Perplexity evaluation for decoders

## Output Layout

```
results/
  raw/                         # per-row CSVs flushed every 50 rows
    peat/<model>/<seed>/<config>.csv
    baseline_<name>/<model>/<seed>.csv
  aggregated/
    table1_headline.csv        # method × model headline
    table2_per_category.csv    # PEAT per-category SS
    table3_ablations.csv
    table4_selector.csv        # SHA vs grid vs random
    table5_scaling.csv         # PEAT on Gemma + Llama
  figures/
    fig1_ss_vs_compute.pdf
    fig2_per_category_heatmap.pdf
logs/
  install.log
  dataset_preflight.log
  dryrun.log
  training.log
  baselines.log
  evaluation.log
state/
  run_state.json
  dryrun_passed
```

## Resuming

If the run is interrupted, simply re-run `python3 run_all.py`. The launcher reads `state/run_state.json` and resumes at the first incomplete cell.

## Troubleshooting

- **Flash-Attention import fails** → Check Python is 3.12, CUDA is 12.4, OS is Linux x86_64.
- **Gated model 403** → Request access on HuggingFace for `google/gemma-3-4b-it` and `meta-llama/Llama-3.2-3B-Instruct` and ensure `HF_KEY` belongs to that account.
- **OOM on Llama-3.2-3B** → Reduce per-device batch size; never enable 8-bit/4-bit because the uniform precision policy forbids it.
- **Gemini returns truncated JSON** → Already handled by `thinking_budget=0` and `max_output_tokens=4096`; if it persists, advance round-robin key.

## License & Contact

This code is provided for research purposes. Please cite our paper if you use PEAT in your work.
