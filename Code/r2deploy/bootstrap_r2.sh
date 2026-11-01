#!/usr/bin/env bash
# PEAT revision-2 worker bootstrap (Jarvis Labs / Akash / Vast).
#
#   bash bootstrap_r2.sh <worker-name> [smoke|worker]
#
# Expects $HOME/peat_r2.env (mode 600) holding HF_KEY and GH_TOKEN; it is never
# echoed. Clones the repository, installs the pinned stack once, then runs the
# 2-sample smoke test ("smoke") or the claiming worker loop ("worker").
set -uo pipefail
WORKER="${1:?worker name}"
MODE="${2:-smoke}"
BASE="${PEAT_BASE:-$HOME}"
ENVF="$HOME/peat_r2.env"
REPO="$BASE/PEAT_Debias"

sed -i 's/\r$//' "$ENVF"
set -a; . "$ENVF"; set +a
export HF_TOKEN="$HF_KEY" HUGGING_FACE_HUB_TOKEN="$HF_KEY" HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="${HF_HOME:-$BASE/hf_cache}" TOKENIZERS_PARALLELISM=false

if [ ! -d "$REPO/.git" ]; then
  git clone -q "https://x-access-token:${GH_TOKEN}@github.com/DevDaring/PEAT_Debias.git" "$REPO"
fi
cd "$REPO"
git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/DevDaring/PEAT_Debias.git"
git config user.name "peat-r2-${WORKER}"
git config user.email "peat-r2-worker@users.noreply.github.com"
git pull -q --rebase --autostash || true

cd "$REPO/Code"
umask 077
printf 'HF_KEY=%s\nHF_TOKEN=%s\n' "$HF_KEY" "$HF_KEY" > .env
umask 022

if [ ! -f "$BASE/.peat_r2_installed" ]; then
  python3 -m pip install -q --upgrade pip
  python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
    python3 -m pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  python3 -m pip install -q "transformers>=5.0.0" "accelerate>=1.1.0" "datasets>=3.0.0,<4.0.0" \
    "peft>=0.13.0" "pandas>=2.2.0" "scipy>=1.13.0" "scikit-learn>=1.5.0" "python-dotenv>=1.0.0" \
    "sentencepiece>=0.2.0" "protobuf>=4.25.0" einops safetensors tqdm requests "huggingface_hub" \
  && touch "$BASE/.peat_r2_installed"
fi
python3 - <<'PY'
import torch, transformers, peft, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| datasets", datasets.__version__)
PY

if [ "$MODE" = "smoke" ]; then
  python3 run_r2.py --smoke 2>&1 | tee "$BASE/smoke_${WORKER}.log"
else
  nohup python3 run_r2.py --worker "$WORKER" >> "$BASE/worker_${WORKER}.log" 2>&1 &
  echo "worker started pid $!"
fi
