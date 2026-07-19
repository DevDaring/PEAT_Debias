"""
PEAT — Secrets Manager.

Loads API keys from .env via python-dotenv. Supports dual naming conventions
so the user's existing .env (with keys like GCP_Key1) works alongside the
canonical names (GCP_KEY_1) from the coding prompt.

No secret may ever appear in any source file, test file, log, or CSV.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Locate and load .env
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=True)
else:
    # Also try CWD
    _cwd_env = Path.cwd() / ".env"
    if _cwd_env.exists():
        load_dotenv(_cwd_env, override=True)
    else:
        print(
            f"WARNING: No .env file found at {_ENV_PATH} or {_cwd_env}. "
            "Secrets must be set as environment variables.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Helper: read with fallback names
# ---------------------------------------------------------------------------
def _get_key(*candidate_names: str, required: bool = True) -> str:
    """Return the first non-empty environment variable found among *candidate_names*.

    Args:
        *candidate_names: One or more env-var names to try, in priority order.
        required: If True, raise ValueError when none of the candidates is set.

    Returns:
        The value of the first matching env var (stripped of whitespace).
    """
    for name in candidate_names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    if required:
        tried = ", ".join(candidate_names)
        raise ValueError(
            f"Required secret not found. Tried environment variables: {tried}. "
            f"Please set at least one in your .env file at {_ENV_PATH}"
        )
    return ""


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------
def hf_token() -> str:
    """HuggingFace API token (needed for gated models like Gemma, Llama)."""
    return _get_key("HF_KEY", "HF_Classic_Token")


def gcp_key(index: int) -> str:
    """Gemini / GCP API key by 1-based index (1–4).

    Accepts both naming conventions:
      GCP_KEY_1  or  GCP_Key1
    """
    if index < 1 or index > 4:
        raise ValueError(f"GCP key index must be 1–4, got {index}")
    return _get_key(f"GCP_KEY_{index}", f"GCP_Key{index}")


def gcp_keys() -> list[str]:
    """Return all four GCP keys as a list (index 0 = key 1)."""
    return [gcp_key(i) for i in range(1, 5)]


def deepseek_key() -> str:
    """DeepSeek API key."""
    return _get_key("DEEPSEEK_KEY", "Deepseek_API_key")


def mistral_key() -> str:
    """Mistral API key."""
    return _get_key("MISTRAL_API_KEY", "Mistral_API_Key")


# ---------------------------------------------------------------------------
# Safe logging
# ---------------------------------------------------------------------------
def mask(key: str) -> str:
    """Mask a secret for safe logging. Returns 'sk-***' + last 4 chars."""
    if not key or len(key) < 4:
        return "sk-***"
    return f"sk-***{key[-4:]}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
# Only HuggingFace is truly required (gated model downloads). The cloud
# LLM-as-judge keys (GCP/DeepSeek/Mistral) drive the auxiliary generation-based
# CrowS choice score, which is explicitly NOT a main-paper result (see
# Supplement §6). The APIN revision's metrics — SS via log-probability, GLUE,
# WikiText PPL, BBQ, and the extrinsic suite — need none of them, so their
# absence must not block the run.
_REQUIRED_KEYS = [
    ("HF_KEY / HF_Classic_Token", hf_token),
]

_OPTIONAL_KEYS = [
    ("GCP_KEY_1 / GCP_Key1", lambda: gcp_key(1)),
    ("GCP_KEY_2 / GCP_Key2", lambda: gcp_key(2)),
    ("GCP_KEY_3 / GCP_Key3", lambda: gcp_key(3)),
    ("GCP_KEY_4 / GCP_Key4", lambda: gcp_key(4)),
    ("DEEPSEEK_KEY / Deepseek_API_key", deepseek_key),
    ("MISTRAL_API_KEY / Mistral_API_Key", mistral_key),
]


def validate_all_keys() -> dict[str, str]:
    """Validate that every REQUIRED key is present; report optional ones.

    Returns:
        dict mapping descriptive key name → masked value (present keys only).

    Raises:
        ValueError: if a truly-required key (HuggingFace) is missing.
    """
    results = {}
    missing = []
    for desc, accessor in _REQUIRED_KEYS:
        try:
            results[desc] = mask(accessor())
        except ValueError:
            missing.append(desc)

    if missing:
        raise ValueError(
            f"Missing required secrets: {', '.join(missing)}. "
            f"Please set them in {_ENV_PATH}"
        )

    for desc, accessor in _OPTIONAL_KEYS:
        try:
            results[desc] = mask(accessor())
        except ValueError:
            results[desc] = "(optional, absent)"
    return results


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Validating secrets...")
    try:
        masked = validate_all_keys()
        for name, mval in masked.items():
            print(f"  loaded: {name} = {mval}")
        print("All secrets validated successfully.")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
