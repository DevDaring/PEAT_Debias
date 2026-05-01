"""
PEAT — Dataset loading, preprocessing, and validation.

# UNIFORM PRECISION POLICY
# All models in this study load in bfloat16 if the GPU supports it (compute capability >= 8.0),
# else float16. We do NOT use 4-bit or 8-bit quantization for any model in any baseline or PEAT
# run, because mixed precision regimes across baselines would invalidate compute-vs-accuracy
# comparisons. LoRA adapters are also in bfloat16/float16 matching the base model. Flash-Attention 2
# is enabled for every causal model and for ModernBERT/NomicBERT where supported. BERT-base uses
# eager attention because it predates SDPA-flash compatibility for masked-LM heads.

Datasets:
  - StereoSet (training): McGill-NLP/stereoset, intrasentence, validation split
  - CrowS-Pairs (test):   nyu-mll/crows_pairs, full split (1508 rows)
  - BBQ (extrinsic):      heegyu/bbq, ambiguous subset, causal LMs only
  - GLUE (utility):       nyu-mll/glue, 8 tasks, encoders only
  - WikiText-103 (utility): Salesforce/wikitext, wikitext-103-raw-v1, decoders only
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from peat.utils import DATA_DIR, LOG_DIR, SMOKE_TEST, SMOKE_TEST_SIZE, ensure_dirs, setup_logger

logger = setup_logger("peat.data", str(LOG_DIR / "dataset_preflight.log"))

STEREOSET_CACHE = DATA_DIR / "stereoset_pairs"
CROWS_BIAS_TYPES = [
    "race-color", "gender", "religion", "age", "nationality",
    "disability", "physical-appearance", "socioeconomic", "sexual-orientation",
]


# ===========================================================================
# §4.1 — StereoSet
# ===========================================================================
# Reference: Nadeem et al., "StereoSet: Measuring stereotypical bias in
# pretrained language models", ACL 2021.
# arXiv: 2004.09456 | Code: https://github.com/moinnadeem/StereoSet
# Used here for: training source — build (context, t_s, t_a) pairs.

def _extract_stereoset_pair(example: dict) -> Optional[dict]:
    """Extract a (context, stereotype_filler, anti-stereotype_filler) triple."""
    context = example.get("context", "")
    if "BLANK" not in context:
        return None

    sentences = example.get("sentences", {})
    labels = sentences.get("gold_label", [])
    sentence_texts = sentences.get("sentence", [])

    if len(labels) != len(sentence_texts):
        return None

    stereo_text = None
    anti_text = None

    # gold_label values may be integers (ClassLabel) or strings.
    # HuggingFace ClassLabel mapping: 0=anti-stereotype, 1=stereotype, 2=unrelated
    _INT_TO_LABEL = {0: "anti-stereotype", 1: "stereotype", 2: "unrelated"}
    for lbl, sent in zip(labels, sentence_texts):
        if isinstance(lbl, int):
            lbl_str = _INT_TO_LABEL.get(lbl, "")
        else:
            lbl_str = str(lbl).lower().strip()
        if lbl_str == "stereotype":
            stereo_text = sent
        elif lbl_str in ("anti-stereotype", "anti_stereotype"):
            anti_text = sent
        # skip "unrelated"

    if stereo_text is None or anti_text is None:
        return None

    # Extract filler: the word(s) that replace BLANK
    # The context has BLANK; filler is the difference between the sentence and context.
    # E.g. context="The chess player was BLANK.", sentence="The chess player was asian."
    # → filler = "asian"
    def extract_filler(context_str: str, sentence: str) -> str:
        parts = context_str.split("BLANK")
        if len(parts) != 2:
            return ""
        prefix = parts[0]  # keep original whitespace for matching
        suffix = parts[1]

        s = sentence
        # Try to strip the prefix
        if prefix and s.startswith(prefix):
            s = s[len(prefix):]
        elif prefix.strip() and s.strip().startswith(prefix.strip()):
            # Fallback: strip then match
            s = s.strip()[len(prefix.strip()):]

        # Try to strip the suffix
        if suffix.strip() and s.strip().endswith(suffix.strip()):
            suf_stripped = suffix.strip()
            s_stripped = s.strip()
            s = s_stripped[: len(s_stripped) - len(suf_stripped)]

        result = s.strip()
        # Don't strip trailing period if filler itself contains periods
        if result.endswith(".") and not result.count(".") > 1:
            result = result.rstrip(".")
        return result

    t_s = extract_filler(context, stereo_text)
    t_a = extract_filler(context, anti_text)

    if not t_s or not t_a:
        return None

    return {
        "id": example.get("id", ""),
        "target": example.get("target", ""),
        "bias_type": example.get("bias_type", ""),
        "context": context,
        "stereotype_sentence": stereo_text,
        "anti_stereotype_sentence": anti_text,
        "t_s": t_s,
        "t_a": t_a,
    }


def load_stereoset_pairs(seed: int = 42, force_rebuild: bool = False):
    """Load StereoSet pairs, building and caching as Parquet if needed.

    Returns:
        (train_df, val_df) — 90/10 split with the given seed.
    """
    ensure_dirs()
    train_path = STEREOSET_CACHE / "train.parquet"
    val_path = STEREOSET_CACHE / "val.parquet"

    if train_path.exists() and val_path.exists() and not force_rebuild:
        logger.info("Loading cached StereoSet pairs from Parquet")
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        if SMOKE_TEST:
            train_df = train_df.head(SMOKE_TEST_SIZE).reset_index(drop=True)
            val_df = val_df.head(SMOKE_TEST_SIZE).reset_index(drop=True)
            logger.info(f"  [smoke-test] truncated to train={len(train_df)}, val={len(val_df)}")
        return train_df, val_df

    logger.info("Building StereoSet pairs from HuggingFace...")
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")

    pairs = []
    skipped = 0
    for ex in ds:
        pair = _extract_stereoset_pair(ex)
        if pair is not None:
            pairs.append(pair)
        else:
            skipped += 1

    logger.info(f"  Built {len(pairs)} pairs, skipped {skipped} rows")
    df = pd.DataFrame(pairs)

    # 90/10 split
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    logger.info(f"  Cached: train={len(train_df)}, val={len(val_df)}")

    if SMOKE_TEST:
        train_df = train_df.head(SMOKE_TEST_SIZE).reset_index(drop=True)
        val_df = val_df.head(SMOKE_TEST_SIZE).reset_index(drop=True)
        logger.info(f"  [smoke-test] truncated to train={len(train_df)}, val={len(val_df)}")

    return train_df, val_df


class StereoSetDataset(Dataset):
    """PyTorch Dataset wrapper for StereoSet pairs."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "context": row["context"],
            "t_s": row["t_s"],
            "t_a": row["t_a"],
            "bias_type": row["bias_type"],
        }


# ===========================================================================
# §4.1 — Multi-token target handling
# ===========================================================================
# Reference: Salazar et al., "Masked Language Model Scoring", ACL 2020.
# arXiv: 1910.14659 | Used here for: pseudo-log-likelihood computation for
# multi-token fillers in encoder models.

def compute_encoder_log_prob(model, tokenizer, context: str, filler: str,
                             device: str = "cuda") -> float:
    """Compute log P(filler | context) for an encoder MLM.

    For multi-token fillers, uses Salazar-style pseudo-log-likelihood:
    replace BLANK with [MASK] × len(filler_tokens), compute sum of
    per-position masked log-probabilities.
    """
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    n_tokens = len(filler_tokens)

    if n_tokens == 0:
        return float("-inf")

    mask_token = tokenizer.mask_token
    masked_context = context.replace("BLANK", " ".join([mask_token] * n_tokens))

    inputs = tokenizer(masked_context, return_tensors="pt", truncation=True,
                       max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Find mask positions
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0]

    if len(mask_positions) != n_tokens:
        # Fallback: use first n_tokens mask positions
        mask_positions = mask_positions[:n_tokens]
        if len(mask_positions) < n_tokens:
            return float("-inf")

    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    total_log_prob = 0.0
    for i, pos in enumerate(mask_positions):
        total_log_prob += log_probs[pos, filler_tokens[i]].item()

    return total_log_prob


def compute_causal_log_prob(model, tokenizer, sentence: str,
                            filler: str, context: str,
                            device: str = "cuda") -> float:
    """Compute length-normalized log P(filler | left context) for a causal LM.

    Reference: Liang et al., "Holistic Evaluation of Language Models", 2022.
    arXiv: 2211.09110 | Used here for: causal LM bias evaluation adaptation.

    Builds the full sentence with filler substituted in, computes sum of
    log P(token_k | left context) over filler tokens only, length-normalized.
    """
    full_sentence = context.replace("BLANK", filler)
    inputs = tokenizer(full_sentence, return_tensors="pt", truncation=True,
                       max_length=1024).to(device)

    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    n_filler = len(filler_tokens)

    if n_filler == 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # rstrip() strips any trailing space so it is never tokenized as a
    # separate token — BPE tokenizers attach the leading space to the next word,
    # so keeping the trailing space would make len(prefix_tokens) off by 1.
    prefix = context.split("BLANK")[0].rstrip()
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
    filler_start = len(prefix_tokens) - 1  # logits[filler_start] predicts filler_tokens[0]

    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    total_log_prob = 0.0
    for i in range(n_filler):
        pos = filler_start + i
        if pos < logits.shape[1] and (filler_start + i) < len(input_ids):
            target_id = input_ids[filler_start + i + 1] if (filler_start + i + 1) < len(input_ids) else filler_tokens[i]
            total_log_prob += log_probs[pos, target_id].item()

    # Length-normalize to make Δ length-invariant
    return total_log_prob / n_filler


# ===========================================================================
# §4.2 — CrowS-Pairs
# ===========================================================================
# Reference: Nangia et al., "CrowS-Pairs: A Challenge Dataset for Measuring
# Social Biases in Masked Language Models", EMNLP 2020.
# arXiv: 2010.00133 | Code: https://github.com/nyu-mll/crows-pairs
# Used here for: primary test set — stereotype score computation.

def load_crows_pairs():
    """Load CrowS-Pairs dataset (1508 rows).

    IMPORTANT: bias_type and stereo_antistereo are HF ClassLabel features
    (stored as integers). We convert them to human-readable strings here.
    """
    ds = load_dataset("nyu-mll/crows_pairs", split="test")

    # Get ClassLabel mappings before converting to pandas
    features = ds.features
    bias_type_names = features["bias_type"].names  # e.g. ["race-color", "socioeconomic", ...]
    stereo_names = features["stereo_antistereo"].names  # ["stereo", "antistereo"]

    df = pd.DataFrame(ds)

    # Convert integer ClassLabel columns to string labels
    df["bias_type"] = df["bias_type"].map(lambda x: bias_type_names[x] if 0 <= x < len(bias_type_names) else str(x))
    df["stereo_antistereo"] = df["stereo_antistereo"].map(lambda x: stereo_names[x] if 0 <= x < len(stereo_names) else str(x))

    if SMOKE_TEST:
        df = df.head(SMOKE_TEST_SIZE).reset_index(drop=True)
        logger.info(f"  [smoke-test] truncated CrowS-Pairs to {len(df)} rows")

    logger.info(f"Loaded CrowS-Pairs: {len(df)} rows, columns: {list(df.columns)}")
    logger.info(f"  bias_type values: {sorted(df['bias_type'].unique())}")
    logger.info(f"  stereo_antistereo values: {sorted(df['stereo_antistereo'].unique())}")
    return df


def get_crows_token_diff(sent_more: str, sent_less: str, tokenizer) -> dict:
    """Compute minimal-edit alignment between sent_more and sent_less.

    Returns dict with 'shared_tokens', 'diff_more', 'diff_less' indices.

    IMPORTANT: uses add_special_tokens=True so that diff position indices
    are in the same coordinate system as the encoding used in
    _encoder_sentence_score (which also calls tokenizer.encode with
    add_special_tokens=True).  CLS is always common prefix and SEP always
    common suffix so they never appear in the diff lists; the special-token
    filter in _encoder_sentence_score excludes them from scoring anyway.

    Reference: Nangia et al. 2020, metric.py — verbatim from
    https://github.com/nyu-mll/crows-pairs/blob/master/metric.py
    """
    tokens_more = tokenizer.encode(sent_more, add_special_tokens=True)
    tokens_less = tokenizer.encode(sent_less, add_special_tokens=True)

    # Find common prefix and suffix
    common_prefix = 0
    for a, b in zip(tokens_more, tokens_less):
        if a == b:
            common_prefix += 1
        else:
            break

    common_suffix = 0
    for a, b in zip(reversed(tokens_more), reversed(tokens_less)):
        if a == b:
            common_suffix += 1
        else:
            break

    # Indices of differing tokens
    end_more = len(tokens_more) - common_suffix if common_suffix > 0 else len(tokens_more)
    end_less = len(tokens_less) - common_suffix if common_suffix > 0 else len(tokens_less)

    diff_more = list(range(common_prefix, end_more))
    diff_less = list(range(common_prefix, end_less))

    # Shared = all positions NOT in the diff
    shared_more = [i for i in range(len(tokens_more)) if i not in diff_more]
    shared_less = [i for i in range(len(tokens_less)) if i not in diff_less]

    return {
        "tokens_more": tokens_more,
        "tokens_less": tokens_less,
        "diff_more": diff_more,
        "diff_less": diff_less,
        "shared_more": shared_more,
        "shared_less": shared_less,
    }


# ===========================================================================
# §4.3 — BBQ
# ===========================================================================
# Reference: Parrish et al., "BBQ: A Hand-Built Bias Benchmark for Question
# Answering", ACL Findings 2022.
# arXiv: 2110.08193 | Used here for: extrinsic sanity check on causal LMs.

# BBQ category configs in heegyu/bbq
BBQ_CONFIGS = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES",
    "Race_x_gender", "Religion", "SES", "Sexual_orientation",
]


def load_bbq_ambiguous():
    """Load BBQ ambiguous-context subset. Causal LMs only.

    heegyu/bbq has multiple configs (one per bias category).
    We load all and concatenate, then filter to ambiguous context.
    """
    all_dfs = []
    for config in BBQ_CONFIGS:
        try:
            ds = load_dataset("heegyu/bbq", config, split="test")
            sub_df = pd.DataFrame(ds)
            sub_df["bbq_category"] = config
            all_dfs.append(sub_df)
        except Exception as e:
            logger.warning(f"  BBQ config '{config}' failed to load: {e}")

    if not all_dfs:
        logger.error("No BBQ configs loaded successfully")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Filter to ambiguous context — check multiple possible column names
    if "context_condition" in df.columns:
        df = df[df["context_condition"] == "ambig"].reset_index(drop=True)
    elif "context_type" in df.columns:
        df = df[df["context_type"] == "ambig"].reset_index(drop=True)
    else:
        # If no explicit column, log warning and use all rows
        logger.warning("BBQ: no context_condition column found, using all rows")

    if SMOKE_TEST:
        df = df.head(SMOKE_TEST_SIZE).reset_index(drop=True)
        logger.info(f"  [smoke-test] truncated BBQ to {len(df)} rows")

    logger.info(f"Loaded BBQ ambiguous: {len(df)} rows from {len(all_dfs)} categories")
    return df


# ===========================================================================
# §4.4 — GLUE
# ===========================================================================
# Reference: Wang et al., "GLUE: A Multi-Task Benchmark and Analysis Platform
# for Natural Language Understanding", EMNLP 2018.
# arXiv: 1804.07461 | Used here for: utility evaluation for encoders.

GLUE_TASKS = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]


def load_glue_task(task_name: str, split: str = "validation"):
    """Load a single GLUE task dataset."""
    if task_name == "mnli" and split == "validation":
        split = "validation_matched"
    ds = load_dataset("nyu-mll/glue", task_name, split=split)
    return ds


# ===========================================================================
# §4.5 — WikiText-103
# ===========================================================================
# Reference: Merity et al., "Pointer Sentinel Mixture Models", 2016.
# arXiv: 1609.07843 | Used here for: perplexity evaluation for decoders.

def load_wikitext103_test():
    """Load WikiText-103 test split for perplexity evaluation."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    logger.info(f"Loaded WikiText-103 test: {len(ds)} examples")
    return ds


# ===========================================================================
# §4.6 — Dataset structural pre-flight check
# ===========================================================================

def validate_dataset_structure() -> bool:
    """Run structural validation on all datasets before training.

    Returns True if all assertions pass, False otherwise.
    Logs everything to logs/dataset_preflight.log.
    """
    ensure_dirs()
    all_ok = True

    # ── StereoSet ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PREFLIGHT: StereoSet")
    logger.info("=" * 60)
    try:
        ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
        logger.info(f"  Rows: {len(ds)}")
        logger.info(f"  Columns: {ds.column_names}")
        logger.info(f"  Features: {ds.features}")

        # Label distribution
        label_counts = {}
        for ex in ds:
            labels = ex.get("sentences", {}).get("gold_label", [])
            n = len(labels)
            label_counts[n] = label_counts.get(n, 0) + 1
        logger.info(f"  Label-count distribution: {label_counts}")

        # 3 random examples
        random.seed(42)
        sample_indices = random.sample(range(len(ds)), min(3, len(ds)))
        for idx in sample_indices:
            ex = ds[idx]
            pair = _extract_stereoset_pair(ex)
            if pair:
                logger.info(
                    f"  Example {idx}: context='{pair['context'][:80]}...', "
                    f"t_s='{pair['t_s']}', t_a='{pair['t_a']}'"
                )

        # Build pairs and validate
        train_df, val_df = load_stereoset_pairs()
        assert len(train_df) > 0, "StereoSet train set is empty"
        assert len(val_df) > 0, "StereoSet val set is empty"
        assert (train_df["t_s"].str.len() > 0).all(), "Empty t_s found in train"
        assert (train_df["t_a"].str.len() > 0).all(), "Empty t_a found in train"
        logger.info(f"  ✓ StereoSet OK: train={len(train_df)}, val={len(val_df)}")

    except Exception as e:
        logger.error(f"  ✗ StereoSet FAILED: {e}")
        all_ok = False

    # ── CrowS-Pairs ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PREFLIGHT: CrowS-Pairs")
    logger.info("=" * 60)
    try:
        df = load_crows_pairs()
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Dtypes:\n{df.dtypes}")

        # 3 random examples
        for idx in df.sample(3, random_state=42).index:
            row = df.loc[idx]
            logger.info(
                f"  Example {idx}: more='{str(row.get('sent_more', ''))[:60]}...', "
                f"less='{str(row.get('sent_less', ''))[:60]}...', "
                f"type={row.get('bias_type', 'N/A')}"
            )

        # Assert all 9 bias categories present (skip in smoke-test: tiny slice
        # won't cover every type and that is expected / not a bug)
        present_types = set(df["bias_type"].unique())
        if SMOKE_TEST:
            logger.info(f"  [smoke-test] bias_type coverage check skipped — present: {sorted(present_types)}")
        else:
            for bt in CROWS_BIAS_TYPES:
                assert bt in present_types, f"Missing bias type: {bt}"
        logger.info(f"  ✓ CrowS-Pairs OK")

    except Exception as e:
        logger.error(f"  ✗ CrowS-Pairs FAILED: {e}")
        all_ok = False

    # ── BBQ ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PREFLIGHT: BBQ")
    logger.info("=" * 60)
    try:
        bbq_df = load_bbq_ambiguous()
        logger.info(f"  Rows: {len(bbq_df)}")
        logger.info(f"  Columns: {list(bbq_df.columns)}")
        for idx in bbq_df.sample(min(3, len(bbq_df)), random_state=42).index:
            row = bbq_df.loc[idx]
            logger.info(f"  Example {idx}: {dict(row)}")
        logger.info("  ✓ BBQ OK")

    except Exception as e:
        logger.error(f"  ✗ BBQ FAILED: {e}")
        all_ok = False

    # ── Summary ────────────────────────────────────────────────────────────
    if all_ok:
        logger.info("=" * 60)
        logger.info("PREFLIGHT: ALL DATASETS PASSED")
        logger.info("=" * 60)
    else:
        logger.error("PREFLIGHT: SOME DATASETS FAILED — see above")

    return all_ok
