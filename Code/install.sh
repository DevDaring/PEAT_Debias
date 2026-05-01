#!/usr/bin/env bash
# ============================================================================
# PEAT — Probability-Equalized Adapter Tuning
# Installation script for Ubuntu 22.04 LTS, Python 3.12, CUDA 12.4
# ============================================================================
set -euo pipefail

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
# Accept Python 3.10 or 3.12 — both have prebuilt flash_attn wheels for CUDA 12
if ! python3 -c "import sys; assert sys.version_info[:2] >= (3,10), f'Need 3.10+, got {sys.version_info[:2]}'"; then
    echo "ERROR: Python 3.10+ is required. Aborting."
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
echo "  Python minor tag: cp${PY_VER}"

echo "[1/5] Checking OS..."
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: Linux x86_64 is required for Flash-Attention 2.8.3 wheel. Aborting."
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
python3 -m pip install --upgrade pip setuptools wheel

# ── Step 2: Install PyTorch ────────────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch 2.5.1 + CUDA 12.4..."
python3 -m pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ── Step 3: Install all dependencies ───────────────────────────────────────
echo ""
echo "[4/5] Installing project dependencies..."
python3 -m pip install \
    "numpy<2.0" \
    transformers==4.46.0 \
    accelerate==0.34.0 \
    datasets==2.16.0 \
    bitsandbytes==0.46.1 \
    pandas==2.2.2 \
    tqdm==4.65.0 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    sentencepiece==0.2.0 \
    protobuf==4.25.0 \
    peft==0.13.0 \
    scipy==1.13.0 \
    scikit-learn==1.5.0 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    evaluate==0.4.2 \
    google-generativeai==0.8.3 \
    xformers==0.0.28.post3 \
    huggingface_hub

# ── Step 4: Flash-Attention 2.8.3 ─────────────────────────────────────────
echo ""
echo "[5/5] Installing Flash-Attention 2.8.3 (cu12/torch2.5/cp${PY_VER})..."
FLASH_WHL="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp${PY_VER}-cp${PY_VER}-linux_x86_64.whl"
wget -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_WHL}" \
    -O /tmp/flash_attn.whl
if [ -s /tmp/flash_attn.whl ]; then
    python3 -m pip install --no-deps /tmp/flash_attn.whl
else
    echo "  WARNING: flash_attn wheel not found for cp${PY_VER}, building from source (slow)..."
    python3 -m pip install flash-attn==2.8.3 --no-build-isolation
fi

# ── Verification ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Verifying installations..."
echo "============================================================"

python3 -c "
import sys

packages = {
    'torch': None,
    'bitsandbytes': None,
    'flash_attn': None,
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
}

failed = []
for pkg, display_name in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {display_name or pkg}: {version}')
    except ImportError as e:
        print(f'  ✗ {display_name or pkg}: FAILED — {e}')
        failed.append(pkg)

import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    cc = torch.cuda.get_device_capability(0)
    print(f'  Compute capability: {cc[0]}.{cc[1]}')

if failed:
    print(f'\nERROR: {len(failed)} package(s) failed to import: {failed}')
    sys.exit(1)
else:
    print('\nAll packages verified successfully.')
"

echo ""
echo "============================================================"
echo " Installation complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
