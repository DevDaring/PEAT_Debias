#!/usr/bin/env bash
# ============================================================================
# bootstrap.sh — runs ON the Vast.ai VM. Full unattended setup + launch.
#
# Sequence:
#   1. Load secrets from $HOME/peat.env (scp'd in by vastctl; chmod 600; never committed)
#   2. Clone repo, install deps (install.sh)
#   3. Force a PREBUILT flash-attn wheel matching the torch ABI (no source build)
#   4. HF login for gated models (Gemma-3, Llama-3.1)
#   5. Dry run (8-step preflight) — hard gate
#   6. 2-row smoke test in an ISOLATED clone (cannot poison the real run's state)
#   7. Start 15-min auto-push (background)
#   8. Start main execution (background, resumable)
#
# SECRETS: no `set -x`; secrets only via env/peat.env; nothing sensitive echoed.
# Idempotent + resumable: safe to re-run after a pre-emption (run_all.py resumes
# from state/, which auto-push persists).
# ============================================================================
set -uo pipefail   # deliberately NO -x
export DEBIAN_FRONTEND=noninteractive

REPO="https://github.com/DevDaring/PEAT_Debias.git"
ROOT="$HOME/PEAT_Debias"
SMOKE="$HOME/PEAT_smoke"
ENVFILE="$HOME/peat.env"
BLOG="$HOME/bootstrap.log"

say() { echo "[bootstrap $(date -u +%H:%M:%SZ)] $*" | tee -a "$BLOG"; }

# ── 1. secrets ─────────────────────────────────────────────────────────────
[ -f "$ENVFILE" ] || { say "FATAL: $ENVFILE not found (scp step failed)"; exit 1; }
set -a; # shellcheck disable=SC1090
source "$ENVFILE"; set +a
export HF_TOKEN="${HF_Classic_Token:-${HF_TOKEN:-}}"
export GH_TOKEN="${Github_Classic_Token:-${GH_TOKEN:-}}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ── 2. clone + install ─────────────────────────────────────────────────────
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
[ -d "$ROOT/.git" ] || git clone --depth 1 "$REPO" "$ROOT"
install -m 600 "$ENVFILE" "$ROOT/Code/.env"     # gitignored; run_all loads it
cd "$ROOT/Code"
say "installing dependencies (install.sh)..."
bash install.sh || say "install.sh returned non-zero (continuing; checking core deps below)"

# ── 3. guaranteed prebuilt flash-attn (ABI auto-detected; no source build) ──
say "installing prebuilt flash-attn matching torch ABI..."
python3 - <<'PY' 2>&1 | tee -a "$BLOG"
import subprocess, sys
try:
    import torch
    abi = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"
    cp  = f"cp{sys.version_info.major}{sys.version_info.minor}"
    whl = (f"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
           f"flash_attn-2.8.3+cu12torch2.5cxx11abi{abi}-{cp}-{cp}-linux_x86_64.whl")
    print(f"[flash] torch={torch.__version__} cuda={torch.version.cuda} abi={abi} py={cp}")
    print(f"[flash] wheel={whl}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                        "--no-build-isolation", whl])
    print("[flash] installed OK" if r.returncode == 0 else
          "[flash] wheel install failed — code uses sdpa/eager, so this is non-fatal")
except Exception as e:
    print(f"[flash] skipped ({type(e).__name__}); sdpa/eager path unaffected")
PY

# hard gate on the packages the pipeline actually needs
python3 -c "import torch,transformers,peft,datasets,accelerate,scipy,sklearn,pandas; \
print('[deps] core OK; cuda=',torch.cuda.is_available())" \
  || { say "FATAL: core dependency import failed"; exit 1; }

# ── 4. HF auth for gated models ────────────────────────────────────────────
python3 - <<'PY'
import os
tok = os.environ.get("HF_TOKEN")
if tok:
    try:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
        print("[hf] login OK")
    except Exception as e:
        print(f"[hf] login failed: {type(e).__name__} (gated models may 401)")
else:
    print("[hf] no HF token — gated Gemma/Llama will be skipped")
PY

# ── 5. dry run (hard gate) ─────────────────────────────────────────────────
say "dry run (8-step preflight)..."
python3 -c "from peat.dryrun import run_dryrun; import sys; sys.exit(0 if run_dryrun(skip_if_recent=False) else 1)" \
  || { say "FATAL: dry run failed — see logs/dryrun.log"; exit 1; }
say "dry run PASSED"

# ── 6. 2-row smoke test in an isolated clone (state isolation) ─────────────
say "smoke test (2 data points, isolated clone)..."
rm -rf "$SMOKE"
git clone --depth 1 "$REPO" "$SMOKE" >/dev/null 2>&1
install -m 600 "$ENVFILE" "$SMOKE/Code/.env"
( cd "$SMOKE/Code" && python3 run_all.py --smoke-test ) \
  || { say "FATAL: smoke test failed"; exit 1; }
say "smoke test PASSED"
rm -rf "$SMOKE"

# ── 7. auto-push (background, survives disconnect/pre-emption) ──────────────
say "starting 15-min auto-push..."
GH_TOKEN="$GH_TOKEN" REPO_DIR="$ROOT" AUTOPUSH_INTERVAL="${AUTOPUSH_INTERVAL:-900}" \
  setsid nohup bash "$ROOT/Code/vast/autopush.sh" >>"$HOME/autopush.log" 2>&1 &
echo $! > "$HOME/autopush.pid"

# ── 8. main execution (background, resumable) ──────────────────────────────
say "starting MAIN execution (run_all.py)..."
cd "$ROOT/Code"
setsid nohup python3 run_all.py >>"$HOME/mainrun.log" 2>&1 &
echo $! > "$HOME/mainrun.pid"
say "main run started (pid $(cat "$HOME/mainrun.pid")). Tail: ~/mainrun.log"
say "bootstrap complete."
