# Master Coding Prompt — PEAT (Probability-Equalized Adapter Tuning)



## PROMPT BEGINS HERE

You are an expert ML research engineer. Generate a complete, production-quality Python codebase for the research project described below. Output every file in full. Do not abbreviate. Do not use placeholders like `# TODO` or `# similar to above`. Every file must be complete and immediately runnable.

---

### 0. PROJECT IDENTITY

**Project name:** PEAT — Probability-Equalized Adapter Tuning for Bias Mitigation in Language Models.

**Goal:** Train a parameter-efficient adapter that equalizes the probability of stereotypical and anti-stereotypical token completions at masked positions, without degrading language-modeling utility. Compare against five recent state-of-the-art bias mitigation methods plus historical baselines, on six language models (three encoders, three causal decoders), under a single fixed compute and precision configuration.

**Core property:** The pipeline runs on a single GCP VM via one launch command. It must execute the full research and comparative study end-to-end without human intervention. It must resume cleanly if interrupted. All six comparative methods must run successfully or be reported as "skipped: <reason>" — the pipeline never silently fails.

---

### 1. ENVIRONMENT AND DEPENDENCIES

**Operating system:** Ubuntu 22.04 LTS (Flash-Attention 2.8.3 wheel for cu12/torch2.5/cp312 requires Python 3.12 and Linux x86_64 — verify in code on startup).

**Installation strategy:** Global Python environment, no venv. Provide a single shell script `install.sh` that runs:

```
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install "numpy<2.0" transformers==4.46.0 accelerate==0.34.0 datasets==2.16.0 \
    bitsandbytes==0.46.1 pandas==2.2.2 tqdm==4.65.0 python-dotenv==1.0.0 requests==2.31.0 \
    sentencepiece==0.2.0 protobuf==4.25.0 peft==0.13.0 scipy==1.13.0 scikit-learn==1.5.0 \
    matplotlib==3.9.0 seaborn==0.13.2 evaluate==0.4.2 google-generativeai==0.8.3 \
    xformers==0.0.28.post3
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/flash_attn.whl
python3 -m pip install --no-deps /tmp/flash_attn.whl
```

After install, run a verification block that imports torch, bitsandbytes, flash_attn, transformers, peft and prints versions. Exit with non-zero if any import fails.

**Uniform precision policy (write this verbatim as a top-level comment in every model-loading file):**
```
# UNIFORM PRECISION POLICY
# All models in this study load in bfloat16 if the GPU supports it (compute capability >= 8.0),
# else float16. We do NOT use 4-bit or 8-bit quantization for any model in any baseline or PEAT
# run, because mixed precision regimes across baselines would invalidate compute-vs-accuracy
# comparisons. LoRA adapters are also in bfloat16/float16 matching the base model. Flash-Attention 2
# is enabled for every causal model and for ModernBERT/NeoBERT where supported. BERT-base uses
# eager attention because it predates SDPA-flash compatibility for masked-LM heads.
```

---

### 2. SECRETS AND ENVIRONMENT FILE

Create `.env.template` containing exactly these keys (no values):
```
HF_KEY=
GCP_KEY_1=
GCP_KEY_2=
GCP_KEY_3=
GCP_KEY_4=
DEEPSEEK_KEY=
```

Create a module `peat/secrets.py` that loads `.env` via `python-dotenv`, exposes typed accessors, and **raises a clear error if any required key is missing**. No secret may ever appear in any source file, test file, log, or CSV. The module must include a function `mask(key: str) -> str` that returns `"sk-***" + key[-4:]` for safe logging.

---

### 3. MODELS

Implement exactly these six models. Hard-code them in `peat/models.py` as a registry.

| Tag | HF ID | Type | Attention impl |
|---|---|---|---|
| `bert-base` | `google-bert/bert-base-uncased` | encoder MLM | eager |
| `modernbert-base` | `answerdotai/ModernBERT-base` | encoder MLM | flash_attention_2 |
| `neobert` | `chandar-lab/NeoBERT` | encoder MLM | flash_attention_2 (requires `trust_remote_code=True`) |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | causal LM | flash_attention_2 |
| `gemma-3-4b` | `google/gemma-3-4b-it` | causal LM | flash_attention_2 (gated; needs `HF_KEY`) |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | causal LM | flash_attention_2 (gated; needs `HF_KEY`) |

Provide loader functions `load_encoder(tag)` and `load_causal(tag)` that:
- Accept the HF token from `secrets.py`.
- Set `torch_dtype` per the uniform precision policy.
- Set `attn_implementation` per the table.
- Set `trust_remote_code=True` only for `neobert`.
- Move to `cuda` and call `model.eval()` initially.
- Return `(model, tokenizer, model_config_dict)`.

---

### 4. DATASETS

Implement `peat/data.py`. **Inspect dataset structure before any processing — multi-MASK rows exist and must be handled correctly.** The cardinal rule: every training/eval example must produce a well-defined `Δ = log P(t_s | context, masked_positions) − log P(t_a | context, masked_positions)`. If that cannot be computed, the row is excluded with the reason logged.

#### 4.1 StereoSet — training source

- Source: HuggingFace `McGill-NLP/stereoset`, config `intrasentence`, split `validation`.
- Each example has fields: `id, target, bias_type, context, sentences[3]` where each sentence has `gold_label ∈ {stereotype, anti-stereotype, unrelated}`.
- The `context` field contains a `BLANK` token.
- For PEAT training, build pairs as: `(context_with_BLANK, t_s = filler_word_in_stereotype_sentence, t_a = filler_word_in_anti-stereotype_sentence)`. Drop the `unrelated` sentence.
- **Multi-token-target handling.** After tokenizing each filler with the model's tokenizer, the result may be 1, 2, or many subword tokens. For encoders, replace `BLANK` with `[MASK] × len(filler)` separately for the stereo and anti-stereo versions, and compute `log P(filler | context)` as the sum of per-position masked log-probabilities (Salazar-style pseudo-log-likelihood, citation in code). For causal LMs, build the full sentence with each filler substituted in, compute the sum of `log P(token_k | left context)` over the filler tokens only, and length-normalize by dividing by `len(filler)` to make `Δ` length-invariant. Document this clearly.
- After build, 90/10 split with seed 42 → `D_train`, `D_val`. Cache as Parquet under `data/stereoset_pairs/`.

#### 4.2 CrowS-Pairs — primary test set

- Source: HuggingFace `nyu-mll/crows_pairs`, full split (1,508 rows).
- Columns: `sent_more, sent_less, stereo_antistereo, bias_type`.
- For each row, build a minimal-edit alignment between `sent_more` and `sent_less` to identify the differing tokens. Use the official Nangia metric: for each sentence, compute the pseudo-log-likelihood of the *unmodified* tokens conditioned on the *modified* tokens (encoder) or simply the sentence log-probability (causal LM). Implement both code paths verbatim from `nyu-mll/crows-pairs/metric.py`, citing it in comments. Score = % of pairs where the model prefers the more-stereotyping sentence. Optimal = 50%.
- **Per-category breakdown is mandatory.** Report SS over all 9 `bias_type` values: `race-color, gender, religion, age, nationality, disability, physical-appearance, socioeconomic, sexual-orientation`.

#### 4.3 BBQ — extrinsic sanity

- Source: HuggingFace `heegyu/bbq`, ambiguous-context subset only.
- Causal LMs only. Encoders skip BBQ.
- Use Parrish et al. official bias score on the ambiguous subset.

#### 4.4 GLUE — utility for encoders

- Source: HuggingFace `nyu-mll/glue`. Tasks: `cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte`.
- Evaluate the **base model + frozen LoRA adapter** on each task by training a fresh classifier head per task for 3 epochs. Report 8-task average. Cite Wang et al. 2018.

#### 4.5 WikiText-103 — utility for decoders

- Source: HuggingFace `Salesforce/wikitext`, config `wikitext-103-raw-v1`, split `test`.
- Compute sliding-window perplexity, stride 512, max length 1024. Cite Merity et al. 2016.

#### 4.6 Dataset structural pre-flight check

Implement a function `validate_dataset_structure()` that runs *before any training* and:
1. Loads each dataset.
2. Prints column names, dtypes, row count.
3. For StereoSet: prints distribution of label counts per row, prints 3 random examples with parsed `(context, t_s, t_a)`, and asserts every kept row has a non-empty `t_s` and `t_a`.
4. For CrowS-Pairs: prints 3 random examples, asserts `sent_more` and `sent_less` have a non-empty token diff, asserts all 9 bias categories are present.
5. For BBQ: prints column structure and 3 random ambiguous-context examples.
6. Logs everything to `logs/dataset_preflight.log` and exits with non-zero if any assertion fails.

---

### 5. PEAT METHOD

Implement in `peat/peat.py`. The method has six components defined exactly as below. Cite each in code comments. Do not modify the formulation.

**Setup (Step 0).** Freeze base model. Attach LoRA (rank=4, alpha=8, dropout=0.0) on `q_proj, k_proj, v_proj, o_proj` and FFN `gate_proj/up_proj/down_proj` (causal) or `query/key/value/dense` and FFN `intermediate.dense/output.dense` (encoder), restricted to the **last 2 transformer blocks only**. LoRA-B initialized to zero so M_θ ≡ M_θ0 at init. For BERT-base that means layers 10–11; for ModernBERT 20–21; for NeoBERT 26–27 (verify by inspecting `model.config.num_hidden_layers`); for Qwen/Gemma/Llama, the last two decoder layers respectively.

**Step 1 — Forward.** For each batch element, run two forwards (M_θ with grad, M_θ0 with no_grad) and obtain `P_θ(·|x,m)` and `P_θ0(·|x,m)` at the masked position(s).

**Step 2 — Δ signal.**
```
Δ(x) = log P_θ(t_s | x, m) − log P_θ(t_a | x, m)
```
Length-normalized per §4.1.

**Step 3 — Equalization loss (Symmetric Huber, τ = 1.0).**
```
ρ(Δ) = Δ²            if |Δ| ≤ τ
       2τ|Δ| − τ²    otherwise
L_neut = mean over batch of ρ(Δ_i)
```
Cite Huber 1964 in comments.

**Step 4 — Pair-mass anchor.**
```
μ_θ  = log( P_θ(t_s|x,m)  + P_θ(t_a|x,m)  )
μ_θ0 = log( P_θ0(t_s|x,m) + P_θ0(t_a|x,m) )
L_pair = mean over batch of (μ_θ − μ_θ0)²
```

**Step 5 — Complement-vocabulary KL.** Build `Q_θ` by zeroing entries at `T = {t_s, t_a}` and renormalizing over `V \ T`; same for `Q_θ0`.
```
L_kl = mean over batch of KL(Q_θ0 || Q_θ)
```

**Step 6 — Total.**
```
L = L_neut + λ_1 · L_pair + λ_2 · L_kl
```
Optimizer: AdamW, lr=1e-4, weight_decay=0.01, warmup 10% of steps, cosine decay. Batch size 32 (encoders) / 8 with grad-accum 4 (causal). Max 5 epochs. Mixed precision via `torch.amp.autocast(dtype=bf16)`.

**Step 7 — Budgeted Configuration Selection.**

Phase A — Successive Halving (cite Jamieson & Talwalkar 2016 in comments):
- Grid `G = {(λ_1, λ_2) : λ_1, λ_2 ∈ {1e-3, 1e-2, 1e-1, 1, 10}}` → 25 configs.
- Round 0: train each config for 1 epoch; keep top 12 by validation `J(c) = |SS_val − 50| + 0.5 · ΔPPL_val`.
- Round 1: continue survivors to 3 cumulative epochs; keep top 4.
- Round 2: continue survivors to 5 cumulative epochs; output 4 finalists.

Phase B — Bootstrap-robust selection:
- Cache each finalist's predictions on `D_val`.
- 1,000 bootstrap resamples. `J_robust = mean(J_b) + 0.5 · std(J_b)`.
- Pick `c_best = argmin J_robust`.

Phase C — Final retraining:
- Train PEAT with `c_best` from scratch on full `D_train` for 5 epochs, seeds {42, 123, 456}.
- Save all 3 adapter checkpoints.

---

### 6. COMPARATIVE STUDY — METHODS

Implement each as a separate module under `peat/baselines/`. Each module exposes one function `run(model_tag, seed) -> dict_of_metrics`. Each module has a top-of-file comment block citing the paper and listing the URL of the official code if used.

1. **Base** (no mitigation) — floor.
2. **CDA** (Zmigrod et al., ACL 2019). Counterfactual data augmentation; re-fine-tune base model on attribute-swapped corpus. Use bias-bench reference: `https://github.com/McGill-NLP/bias-bench`.
3. **Self-Debias** (Schick et al., TACL 2021). Inference-time decoding modification, no retraining. Reference: `https://github.com/timoschick/self-debiasing`.
4. **Auto-Debias** (Guo et al., ACL 2022). JS-divergence on automatically-discovered biased prompts. Reference: `https://github.com/Irenehere/Auto-Debias`.
5. **BiasEdit** (Xu et al., TrustNLP@NAACL 2025, arXiv 2503.08588). Lightweight editor networks; debiasing + retention loss. Reference: `https://github.com/zjunlp/BiasEdit`.
6. **FairSteer** (Li et al., ACL 2025 Findings, arXiv 2504.14492). Inference-time activation steering with debiasing steering vectors.
7. **BiasUnlearn** (Liu et al., EMNLP 2025, arXiv 2509.25673). Dual-pathway unlearning.
8. **KnowBias** (Pan et al., arXiv 2601.21864, Jan 2026). Bias-neuron enhancement at inference.
9. **LoRA + vanilla SFT** (internal ablation). Same rank=4, last-2-layer LoRA, but trained with standard masked-LM cross-entropy on the same StereoSet pairs. Isolates "the loss is doing the work."
10. **PEAT** (ours).

For methods 2–8: where official code does not run on a given backbone, fall back to the paper's reported numbers, mark the cell `†reported by authors`, and emit a structured warning to `logs/baselines.log`. Never silently substitute.

---

### 7. EVALUATION

Implement in `peat/eval.py`.

**SS computation.**
- Encoder: for each CrowS-Pairs row, mask the modified tokens and compute per-position log-probabilities under the unmodified context, sum them, compare `sent_more` vs `sent_less`. Cite Nangia et al. 2020 `metric.py`.
- Causal LM: compute `log P(sentence)` autoregressively for each side, conditioning only on left context. Cite Liang et al. 2022 for the adaptation.

**Aggregate metric.** SS = 100 × (# pairs where preferred = stereotyping) / total_pairs. Optimal = 50.

**Per-category SS.** Same metric, partitioned by the 9 `bias_type` values.

**Statistical reporting.** For each (method, model) cell: 3 seeds × 1,000 bootstrap resamples. Report mean and 95% CI as `mean [lo, hi]`.

**Utility.** GLUE 8-task average for encoders, WikiText-103 perplexity for decoders.

**BBQ.** Causal LMs only, ambiguous subset, official bias-score computation.

---

### 8. JSON OUTPUT FROM CAUSAL MODELS AND LLM-AS-JUDGE FALLBACK

When PEAT or a baseline needs structured output from a causal LM (BBQ answer extraction, baseline judgement, etc.), follow this protocol exactly:

**Native JSON mode for all causal LMs at sampling time.**
- For Qwen/Gemma/Llama outputs that need parsing, prompt with explicit JSON schema and a one-shot example.
- Always set `do_sample=False` (greedy) for reproducibility.
- Always include this example block in the prompt:

```
Return ONLY a JSON object exactly matching this schema. No markdown, no code fences,
no commentary. Example:
{"answer": "A", "confidence": 0.91}
```

**Robust JSON parsing.** Implement `parse_json_safely(text)` that:
1. Strips markdown fences ```json ... ``` and any text outside the outermost `{...}`.
2. Calls `json.loads(s, strict=False)` to allow literal newlines inside string values.
3. Falls back to LLM-as-judge if parsing fails.

**LLM-as-judge fallback.** Use Gemini 2.5 Flash Lite via `google-generativeai` with the four GCP keys in round-robin (no retry to a different provider — explicit constraint). Set:
```
generation_config = {
  "response_mime_type": "application/json",
  "thinking_config": {"thinking_budget": 0},
  "max_output_tokens": 4096,
  "temperature": 0.0
}
```
Cite in code: "Thinking-budget=0 because reasoning tokens consume the output budget and produce truncated JSON; cf. Gemini 2.5 docs." Round-robin the four keys; on rate-limit error from one key, advance to the next. **No DeepSeek fallback** — DeepSeek is wired up but not auto-invoked. The user can manually flip a flag.

---

### 9. RESUMABILITY AND CHECKPOINTING

Every long-running stage must be resumable. Use:
- `state/run_state.json` — top-level pipeline progress: stage, model_tag, seed, config_id, status.
- `state/<stage>/<model>/<seed>/<config>.checkpoint` — adapter weights every epoch.
- `results/raw/<stage>/<model>/<seed>/<config>.csv` — per-row predictions, **flushed every 50 rows**.
- On startup, the launcher reads `run_state.json`, identifies the first incomplete cell, and resumes there. Already-complete cells are skipped.

**Memory cleaning between stages.** After every (model, seed, config) cell:
```
del model
del optimizer
torch.cuda.empty_cache()
gc.collect()
```
Wrap this in a `cleanup()` utility called from a `finally` block.

---

### 10. RESULTS LAYOUT

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
  ...
```

All CSVs use UTF-8, comma-separated, with header row. Long-form names everywhere — write `English` not `en`, `BERT-base-uncased` not `bert`, `Stereotype Score` not `SS` in column headers (internal code can use short names).

---

### 11. DRY RUN

Implement `peat/dryrun.py`, invoked by the launcher *before* the real run. Dry run does the following sequentially and exits non-zero on any failure:

1. **Environment check.** Python version, OS, CUDA available, GPU compute capability, free GPU RAM, free disk. Print all to `logs/dryrun.log`.
2. **Library import check.** Import torch, transformers, peft, bitsandbytes, flash_attn, datasets, evaluate, google.generativeai, scipy. Print each version.
3. **Secrets check.** Load every key from `.env`. For each, log `loaded: <name>=sk-***xxxx`. If any required key missing, raise.
4. **API key live test.**
   - HuggingFace: call `huggingface_hub.HfApi().whoami(token=HF_KEY)`.
   - Gemini (×4 keys): for each `GCP_KEY_i`, call Gemini 2.5 Flash Lite with prompt `"Return the JSON {\"ok\": true}"` and `response_mime_type=application/json`. Verify `{"ok": true}` parses.
   - DeepSeek: call `chat/completions` once with a 1-token prompt; log status (do **not** auto-fall-back — this only verifies the key works).
5. **Model load check.** For each of the 6 models: `load_*()`, run a single forward on a 8-token dummy input, free. Log `loaded + forward OK: <model_tag>` or full traceback.
6. **Dataset structure check.** Call `validate_dataset_structure()` from §4.6.
7. **One-row training check.** For each model: run **one** PEAT training step on **one** StereoSet pair. Verify loss is finite and gradients flow into LoRA params (and only LoRA params).
8. **Path check.** Verify all output directories under `results/`, `state/`, `logs/` are writable.

If all eight pass, write `state/dryrun_passed` (empty file) and proceed. The real run refuses to start without this file.

---

### 12. LAUNCHER

Single entry point `run_all.py`. The user runs:

```
python3 run_all.py
```

`run_all.py` does in order:

1. Load environment, setup logging.
2. Run dry run (§11). If `state/dryrun_passed` exists from a prior run within the last 24 hours, skip — otherwise enforce.
3. Run dataset preflight (§4.6).
4. **Stage 1 — PEAT training.** For each core model in `[bert-base, modernbert-base, neobert, qwen2.5-1.5b]`:
   a. Run successive halving Phase A (25→12→4) on seed 42.
   b. Run Phase B bootstrap-robust selection.
   c. Run Phase C: final retraining with seeds {42, 123, 456} for `c_best`.
   d. Evaluate on CrowS-Pairs full (per-category) + GLUE (encoders) or BBQ + WikiText-103 (causal).
5. **Stage 2 — PEAT scaling.** For each scaling model in `[gemma-3-4b,Llama-3.2-3B-Instruct]`: skip Phase A entirely, **reuse the `c_best` from `qwen2.5-1.5b`** (justified in code comments: "smallest causal model's optimal config transfers; we verify scaling, not re-search"). Run 3 seeds. Evaluate.
6. **Stage 3 — Baselines.** For each baseline in §6 (excluding PEAT) × each core model × 3 seeds, run and evaluate. Skipped cells emit a structured log entry.
7. **Stage 4 — Aggregation.** Build all five tables in `results/aggregated/`. Compute bootstrap CIs.
8. **Stage 5 — Figures.** Build the two figures in `results/figures/`.
9. Print a final summary to stdout listing every cell with status `(completed | skipped: reason)`.

Each stage writes its progress into `state/run_state.json` after every completed cell so resume is granular.

---

### 13. README.md

Write `README.md` to be the single source of truth for someone who has never seen this repo. Include in this order:

1. **What this is** — one paragraph.
2. **Hardware requirements** — single A100-40GB or A6000-48GB; ~150 GB disk; ~200 GPU-hours total.
3. **OS requirements** — Ubuntu 22.04, Python 3.12, CUDA 12.4. Explicit warning: Flash-Attention 2.8.3 wheel only works for cu12/torch2.5/cp312 on Linux x86_64.
4. **Setup**. Step 1: `cp .env.template .env` and fill in keys. Step 2: `bash install.sh`. Step 3: `python3 run_all.py`.
5. **What runs.** List every dataset, every model, every method with full citations.
6. **Output layout.** Reproduce §10 directory tree.
7. **Resuming.** "If the run is interrupted, simply re-run `python3 run_all.py`. The launcher reads `state/run_state.json` and resumes at the first incomplete cell."
8. **Troubleshooting.**
   - "Flash-Attention import fails" → check Python is 3.12, CUDA is 12.4, OS is Linux x86_64.
   - "Gated model 403" → request access on HuggingFace for `google/gemma-3-4b-it` and `meta-llama/Llama-3.1-8B-Instruct` and ensure `HF_KEY` belongs to that account.
   - "OOM onLlama-3.2-3B-Instruct" → reduce per-device batch size in `configs/llama.yaml`; never enable 8-bit/4-bit because the uniform precision policy forbids it.
   - "Gemini returns truncated JSON" → already handled by `thinking_budget=0` and `max_output_tokens=4096`; if it persists, advance round-robin key.
9. **Citations** — full bibtex for every paper referenced, in one block.
10. **License + contact.**

---

### 14. CITATIONS IN CODE

Every algorithmic component must carry an inline comment block of the form:

```
# Reference: <Author> et al., "<Title>", <Venue> <Year>.
# arXiv: <id>   |   Code: <url>   |   Used here for: <one-sentence rationale>
```

Do this for: Nangia et al. 2020 (CrowS-Pairs metric), Nadeem et al. 2021 (StereoSet), Salazar et al. 2020 (pseudo-log-likelihood), Liang et al. 2022 (causal LM bias eval adaptation), Wang et al. 2018 (GLUE), Merity et al. 2016 (WikiText-103), Parrish et al. 2022 (BBQ), Hu et al. 2022 (LoRA), Huber 1964 (Huber loss), Jamieson & Talwalkar 2016 (Successive Halving), Goldfarb-Tarrant et al. 2021 (intrinsic-extrinsic gap), and every method in §6.

---

### 15. NON-NEGOTIABLES

- One-file launch: `python3 run_all.py` does everything.
- Resumes cleanly from interruption.
- Flushes results every 50 rows.
- Every model loads with the same precision policy.
- Flash-Attention enabled where supported.
- Dry run runs before main run, including 1-row-per-model functional check.
- Long-form names in all human-readable output.
- All keys from `.env`; no hardcoded secrets anywhere in any file including tests.
- No 4-bit / 8-bit anywhere. No QLoRA. Plain LoRA in bf16/fp16 only.
- Memory cleaned between every cell.
- No retry from primary LLM-as-judge to secondary; round-robin within Gemini keys only.
- Every paper cited in code comments where used.

Generate every file in full. Begin with the directory tree, then output each file in its entirety, in this order: `install.sh`, `.env.template`, `README.md`, `run_all.py`, `peat/secrets.py`, `peat/models.py`, `peat/data.py`, `peat/peat.py`, `peat/eval.py`, `peat/dryrun.py`, `peat/utils.py`, `peat/baselines/__init__.py`, then one file per baseline in `peat/baselines/`, then the aggregation/figure scripts. End with the bibtex block for the README.

