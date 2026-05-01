"""
PEAT — Shared utilities.

Central module for:
  - GPU precision detection (bf16 vs fp16)
  - Memory cleanup between stages
  - Logging setup
  - Run-state (checkpoint) management
  - CSV row flushing
  - JSON-safe parsing with LLM-as-judge fallback
"""

import gc
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

# ---------------------------------------------------------------------------
# Project paths (relative to CWD which should be Code/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
AGG_DIR = RESULTS_DIR / "aggregated"
FIG_DIR = RESULTS_DIR / "figures"

STATE_DIR = PROJECT_ROOT / "state"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

ALL_DIRS = [
    RAW_DIR, AGG_DIR, FIG_DIR,
    STATE_DIR, LOG_DIR, DATA_DIR,
    DATA_DIR / "stereoset_pairs",
]


def ensure_dirs() -> None:
    """Create all required output directories."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SMS Notifications (Fast2SMS)
# ---------------------------------------------------------------------------
def send_sms(message: str) -> None:
    """Send an SMS via Fast2SMS using credentials from .env.

    Reads PHONE_NO and FAST2SMS_API_KEY from the environment.
    Silently swallows all errors so it never interrupts the pipeline.
    Phone number: strips leading '+' and whitespace (Fast2SMS wants digits only).
    """
    try:
        import requests as _req
        from peat.secrets import _get_key  # type: ignore

        api_key  = _get_key("FAST2SMS_API_KEY", required=False)
        phone_no = _get_key("PHONE_NO", required=False)

        if not api_key or not phone_no:
            return  # not configured — skip silently

        # Fast2SMS expects digits only, no '+' or spaces
        phone_digits = phone_no.replace("+", "").replace(" ", "").strip()

        payload = {
            "sender_id": "FSTSMS",
            "message":   message[:160],   # SMS length limit
            "language":  "english",
            "route":     "q",
            "numbers":   phone_digits,
        }
        headers = {
            "authorization":  api_key,
            "Content-Type":   "application/x-www-form-urlencoded",
        }
        _req.post(
            "https://www.fast2sms.com/dev/bulkV2",
            data=payload,
            headers=headers,
            timeout=15,
        )
    except Exception:
        pass  # SMS failure must never crash the pipeline


# ---------------------------------------------------------------------------
# Uniform Precision Policy
# ---------------------------------------------------------------------------
# UNIFORM PRECISION POLICY
# All models in this study load in bfloat16 if the GPU supports it (compute capability >= 8.0),
# else float16. We do NOT use 4-bit or 8-bit quantization for any model in any baseline or PEAT
# run, because mixed precision regimes across baselines would invalidate compute-vs-accuracy
# comparisons. LoRA adapters are also in bfloat16/float16 matching the base model. Flash-Attention 2
# is enabled for every causal model and for ModernBERT/NomicBERT where supported. BERT-base uses
# eager attention because it predates SDPA-flash compatibility for masked-LM heads.


def get_dtype() -> torch.dtype:
    """Return bfloat16 if GPU compute capability >= 8.0, else float16."""
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 8:
            return torch.bfloat16
    return torch.float16


def get_autocast_dtype() -> torch.dtype:
    """Same as get_dtype — used for torch.amp.autocast."""
    return get_dtype()


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------
def cleanup(*objects: Any) -> None:
    """Delete objects and free GPU memory.

    Should be called from a `finally` block after every (model, seed, config) cell.
    """
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a named logger that writes to both console and file.

    Args:
        name: Logger name.
        log_file: Optional path to log file. If None, only console output.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        ensure_dirs()
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Run-state management (checkpointing)
# ---------------------------------------------------------------------------
RUN_STATE_FILE = STATE_DIR / "run_state.json"


def load_run_state() -> dict:
    """Load the top-level pipeline state from disk.

    Returns:
        dict with keys like 'stages', 'current_stage', 'cells', etc.
        Returns empty dict if no state file exists.
    """
    if RUN_STATE_FILE.exists():
        with open(RUN_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_run_state(state: dict) -> None:
    """Persist pipeline state to disk atomically."""
    ensure_dirs()
    tmp = RUN_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(RUN_STATE_FILE)


def cell_key(stage: str, model: str, seed: int, config: str = "default") -> str:
    """Generate a unique key for a (stage, model, seed, config) cell."""
    return f"{stage}|{model}|{seed}|{config}"


def is_cell_complete(state: dict, key: str) -> bool:
    """Check if a cell has already completed successfully."""
    cells = state.get("cells", {})
    return cells.get(key, {}).get("status") == "completed"


def mark_cell_complete(state: dict, key: str, metrics: Optional[dict] = None) -> None:
    """Mark a cell as completed and save state."""
    if "cells" not in state:
        state["cells"] = {}
    state["cells"][key] = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics or {},
    }
    save_run_state(state)


def mark_cell_skipped(state: dict, key: str, reason: str) -> None:
    """Mark a cell as skipped with a reason."""
    if "cells" not in state:
        state["cells"] = {}
    state["cells"][key] = {
        "status": f"skipped: {reason}",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_run_state(state)


def mark_cell_failed(state: dict, key: str, error: str) -> None:
    """Mark a cell as failed with an error message."""
    if "cells" not in state:
        state["cells"] = {}
    state["cells"][key] = {
        "status": f"failed: {error}",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_run_state(state)


# ---------------------------------------------------------------------------
# CSV flushing
# ---------------------------------------------------------------------------
class CSVFlusher:
    """Accumulate rows and flush to CSV every `flush_every` rows.

    Ensures partial results survive crashes.
    """

    def __init__(self, path: Path, columns: list[str], flush_every: int = 50):
        self.path = Path(path)
        self.columns = columns
        self.flush_every = flush_every
        self._buffer: list[dict] = []
        self._total_written = 0

        # Create parent dirs
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # If file already exists, count existing rows for resume
        if self.path.exists():
            import pandas as pd
            try:
                existing = pd.read_csv(self.path)
                self._total_written = len(existing)
            except Exception:
                self._total_written = 0

    @property
    def total_written(self) -> int:
        return self._total_written + len(self._buffer)

    def add_row(self, row: dict) -> None:
        """Add a single row. Auto-flushes when buffer reaches flush_every."""
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows to disk."""
        if not self._buffer:
            return
        import pandas as pd
        df = pd.DataFrame(self._buffer, columns=self.columns)
        write_header = not self.path.exists() or self._total_written == 0
        df.to_csv(self.path, mode="a", header=write_header, index=False,
                  encoding="utf-8")
        self._total_written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        """Flush any remaining rows."""
        self.flush()


# ---------------------------------------------------------------------------
# JSON-safe parsing
# ---------------------------------------------------------------------------
def parse_json_safely(text: str) -> Optional[dict]:
    """Parse JSON robustly from LLM output.

    1. Strips markdown fences ```json ... ``` and any text outside outermost {}.
    2. Calls json.loads with strict=False to allow literal newlines.
    3. Returns None if parsing fails (caller should invoke LLM-as-judge fallback).
    """
    if not text or not text.strip():
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)

    # Extract outermost {...}
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
        return None

    json_str = cleaned[brace_start : brace_end + 1]

    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# LLM-as-judge fallback chain: Gemini 2.5 Flash Lite → DeepSeek → Mistral
# ---------------------------------------------------------------------------
# Fallback rules (applies at every level):
#   • API exception (rate-limit / auth / network) → advance to next provider
#   • API responded but JSON is malformed          → stop chain, return None
#   • No retries within a provider (Gemini rotates its 4 keys, but that is
#     key rotation, not retrying the same key)
# Gemini safety filters are fully disabled so bias-related content is not
# blocked by the judge.
# ---------------------------------------------------------------------------
_gemini_key_index = 0

# Gemini safety settings — all categories set to BLOCK_NONE
_GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


def llm_judge_extract(
    raw_text: str,
    schema_description: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """Use Gemini 2.5 Flash Lite → DeepSeek → Mistral as fallback JSON extractor.

    Called when a local model's output passes through parse_json_safely() and
    returns None (malformed JSON from the *local* model).

    Fallback rules:
        - API exception (rate-limit, auth, network) → try next provider; no retry
        - Provider responds but JSON is malformed   → stop chain immediately; return None
        - All providers exhausted                   → return None

    Args:
        raw_text: The raw LLM output that failed JSON parsing.
        schema_description: Description of the expected JSON schema.
        logger: Optional logger for diagnostics.

    Returns:
        Parsed dict on success, or None if all judges fail / JSON is malformed.
    """
    global _gemini_key_index

    prompt = (
        f"Extract structured data from the following text and return ONLY a "
        f"JSON object matching this schema: {schema_description}\n\n"
        f"Text to extract from:\n{raw_text}\n\n"
        f"Return ONLY a JSON object exactly matching the schema. "
        f"No markdown, no code fences, no commentary."
    )

    # ── Level 1: Gemini 2.5 Flash Lite (round-robin 4 GCP keys) ──────────
    gemini_all_api_failed = False
    try:
        import google.generativeai as genai
        from peat.secrets import gcp_keys

        keys = gcp_keys()
        gemini_all_api_failed = True  # assume failure until a key succeeds or responds

        for attempt in range(len(keys)):
            idx = (_gemini_key_index + attempt) % len(keys)
            key = keys[idx]
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel("gemini-2.5-flash-lite")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        max_output_tokens=4096,
                        temperature=0.0,
                        thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
                    ),
                    safety_settings=_GEMINI_SAFETY_SETTINGS,
                )
                # API responded — advance key index
                _gemini_key_index = (idx + 1) % len(keys)
                gemini_all_api_failed = False

                result = parse_json_safely(response.text)
                if result is not None:
                    return result  # ✓ success
                # Gemini responded but JSON is malformed → stop chain
                if logger:
                    logger.warning(
                        "Gemini (key %d) returned malformed JSON; stopping fallback chain",
                        idx + 1,
                    )
                return None

            except Exception as e:
                error_str = str(e).lower()
                if logger:
                    if "rate" in error_str or "quota" in error_str or "429" in error_str:
                        logger.warning("Gemini key %d rate-limited; trying next key", idx + 1)
                    else:
                        logger.warning("Gemini key %d API error: %s; trying next key", idx + 1, e)
                continue  # try next key

        if gemini_all_api_failed and logger:
            logger.warning("All Gemini keys failed with API errors; falling back to DeepSeek")

    except ImportError as e:
        if logger:
            logger.error("google.generativeai not available: %s; falling back to DeepSeek", e)
    except Exception as e:
        if logger:
            logger.error("Gemini setup error: %s; falling back to DeepSeek", e)

    # ── Level 2: DeepSeek (single attempt) ────────────────────────────────
    try:
        import requests as _requests
        from peat.secrets import deepseek_key

        resp = _requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {deepseek_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

        result = parse_json_safely(raw)
        if result is not None:
            return result  # ✓ success
        # DeepSeek responded but JSON is malformed → stop chain
        if logger:
            logger.warning("DeepSeek returned malformed JSON; stopping fallback chain")
        return None

    except Exception as e:
        if logger:
            logger.warning("DeepSeek API error: %s; falling back to Mistral", e)

    # ── Level 3: Mistral mistral-small-latest (single attempt) ────────────
    try:
        import requests as _requests
        from peat.secrets import mistral_key

        resp = _requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {mistral_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

        result = parse_json_safely(raw)
        if result is not None:
            return result  # ✓ success
        # Mistral responded but JSON is malformed → return None
        if logger:
            logger.warning("Mistral returned malformed JSON; all judges exhausted")
        return None

    except Exception as e:
        if logger:
            logger.error("Mistral API error: %s; all judges exhausted, returning None", e)
        return None


# ---------------------------------------------------------------------------
# Smoke-test mode — limits all datasets to N rows for quick pipeline checks
# ---------------------------------------------------------------------------
SMOKE_TEST: bool = False
SMOKE_TEST_SIZE: int = 4


def set_smoke_test(enabled: bool, size: int = 4) -> None:
    """Enable or disable smoke-test mode globally.

    When enabled every dataset loader caps its output to ``size`` rows,
    evaluation loops exit after ``size`` rows, and training runs for a
    single epoch over ``size`` samples.  Use for rapid integration checks
    before committing to a full run.
    """
    global SMOKE_TEST, SMOKE_TEST_SIZE
    SMOKE_TEST = enabled
    SMOKE_TEST_SIZE = size
    if enabled:
        _logger = logging.getLogger("peat")
        _logger.info(f"SMOKE TEST MODE enabled: datasets capped to {size} rows per split")


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------
class Timer:
    """Simple context manager for timing blocks."""

    def __init__(self, label: str = "", logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.logger and self.label:
            self.logger.info(f"{self.label}: {self.elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Seed setting
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
