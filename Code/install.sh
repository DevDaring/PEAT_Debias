#!/usr/bin/env bash
# ============================================================================
# PEAT — Probability-Equalized Adapter Tuning
# Installation script for Ubuntu 22.04 LTS, Python 3.10/3.12, CUDA 12.x
#
# Design:
#   - set -u  : fail on undefined variable references (catches typos)
#   - No set -e: critical packages check return codes explicitly; optional
#     packages (flash-attn, xformers) warn and continue on failure.
# ============================================================================
set -uo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " PEAT Install — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# ── Pre-flight checks ──────────────────────────────────────────────────────
echo "[1/5] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
echo "  Python: $PYTHON_VERSION"
if ! python3 -c "import sys; assert sys.version_info[:2] >= (3,10), f'Need 3.10+, got {sys.version_info[:2]}'"; then
    echo "ERROR: Python 3.10+ is required. Aborting."
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
echo "  Python minor tag: cp${PY_VER}"

echo "[1/5] Checking OS..."
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: Linux x86_64 is required. Aborting."
    exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: x86_64 architecture required. Aborting."
    exit 1
fi
echo "  OS: $(uname -s) $(uname -m) — OK"

# ── Step 1: Upgrade pip ────────────────────────────────────────────────────
echo ""
echo "[2/5] Upgrading pip, setuptools, wheel..."
if ! python3 -m pip --version &>/dev/null; then
    echo "  pip not found — bootstrapping..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi
python3 -m pip install --upgrade pip setuptools wheel || { echo "ERROR: pip upgrade failed"; exit 1; }

# ── Step 2: Install PyTorch 2.5.1 ─────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch 2.5.1 + CUDA 12.4..."
python3 -m pip install \
    torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    || { echo "ERROR: PyTorch install failed"; exit 1; }

# Verify CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available after torch install'" \
    || echo "  WARNING: CUDA not available (OK for CPU-only setup)"

# ── Step 3: Install all core dependencies ─────────────────────────────────
echo ""
echo "[4/5] Installing project dependencies..."
# Versions chosen for compatibility with PyTorch 2.5.1 and the
# transformers 5.x API used throughout the codebase.
python3 -m pip install \
    "numpy<2.0" \
    "transformers>=5.0.0" \
    "accelerate>=0.40.0" \
    "datasets>=2.16.0" \
    "bitsandbytes>=0.46.1" \
    "pandas>=2.2.0" \
    "tqdm>=4.65.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.31.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=4.25.0" \
    "peft>=0.13.0" \
    "scipy>=1.13.0" \
    "scikit-learn>=1.5.0" \
    "matplotlib>=3.9.0" \
    "seaborn>=0.13.0" \
    "evaluate>=0.4.2" \
    "google-generativeai>=0.8.3" \
    "einops" \
    "packaging" \
    "huggingface_hub[cli]" \
    || { echo "ERROR: Core dependency install failed"; exit 1; }

# ── Step 4: Flash-Attention (optional — SDPA is used as primary attn) ──────
# Flash-attn is installed for completeness; the codebase uses attn_impl="sdpa"
# or "eager" for all models, so this is NOT required for correctness.
# Failure here is non-fatal.
echo ""
echo "[5/5] Installing Flash-Attention 2.8.3 (optional, non-fatal)..."
FLASH_BASE="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"
FLASH_WHL="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp${PY_VER}-cp${PY_VER}-linux_x86_64.whl"

_flash_ok=0

# Attempt 1: prebuilt wheel from GitHub releases (fastest, ~200 MB)
echo "  Attempt 1/3: prebuilt wheel for cp${PY_VER}..."
if wget -q --timeout=120 "${FLASH_BASE}/${FLASH_WHL}" -O /tmp/flash_attn.whl 2>/dev/null \
        && [ -s /tmp/flash_attn.whl ]; then
    python3 -m pip install --no-deps /tmp/flash_attn.whl \
        && { echo "  ✓ flash_attn 2.8.3 installed from prebuilt wheel"; _flash_ok=1; } \
        || echo "  Prebuilt wheel rejected (ABI mismatch?); trying pip..."
else
    echo "  Prebuilt wheel not found for cp${PY_VER}; trying pip..."
fi

# Attempt 2: pip with prebuilt (some versions have index entries)
if [ "$_flash_ok" -eq 0 ]; then
    echo "  Attempt 2/3: pip install flash-attn==2.8.3 ..."
    python3 -m pip install flash-attn==2.8.3 --no-build-isolation 2>&1 | tail -3 \
        && { echo "  ✓ flash_attn installed via pip"; _flash_ok=1; } \
        || echo "  pip install failed; trying latest release..."
fi

# Attempt 3: latest flash-attn from pip (gives up on 2.8.3 pin)
if [ "$_flash_ok" -eq 0 ]; then
    echo "  Attempt 3/3: pip install flash-attn (latest) ..."
    python3 -m pip install flash-attn --no-build-isolation 2>&1 | tail -3 \
        && { echo "  ✓ flash_attn installed (latest version)"; _flash_ok=1; } \
        || echo "  WARNING: All flash-attn install attempts failed."
fi

if [ "$_flash_ok" -eq 0 ]; then
    echo "  WARNING: flash_attn not installed. The pipeline uses sdpa/eager attention"
    echo "           for all models, so this does NOT affect correctness or results."
fi

# ── Verification ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Verifying installations..."
echo "============================================================"

python3 -c "
import sys

# Required packages — any failure here is a hard error.
required = {
    'torch': None,
    'bitsandbytes': None,
    'transformers': None,
    'peft': None,
    'datasets': None,
    'evaluate': None,
    'accelerate': None,
    'scipy': None,
    'sklearn': 'scikit-learn',
    'pandas': None,
    'matplotlib': None,
    'seaborn': None,
    'google.generativeai': 'google-generativeai',
    'huggingface_hub': None,
    'einops': None,
}

# Optional packages — failure is a warning, not an error.
optional = {
    'flash_attn': 'flash-attn (optional)',
}

failed = []
for pkg, display_name in required.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {display_name or pkg}: {version}')
    except ImportError as e:
        print(f'  ✗ {display_name or pkg}: FAILED — {e}')
        failed.append(pkg)

for pkg, display_name in optional.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {display_name or pkg}: {version}')
    except ImportError:
        print(f'  ~ {display_name or pkg}: not installed (optional)')

import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    cc = torch.cuda.get_device_capability(0)
    print(f'  Compute capability: {cc[0]}.{cc[1]}')

if failed:
    print(f'\nERROR: {len(failed)} required package(s) failed: {failed}')
    sys.exit(1)
else:
    print('\nAll required packages verified successfully.')
"

echo ""
echo "============================================================"
echo " Installation complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
