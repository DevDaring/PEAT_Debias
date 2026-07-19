# PEAT on Vast.ai — unattended, secret-safe, pre-emption-resilient

Local controller + on-VM bootstrap for running the PEAT pipeline on a rented
Vast.ai GPU. Designed for interruptible instances: results (and the small
resumable state file) are pushed to GitHub every 15 minutes, so a pre-emption
costs at most one interval.

## Files
| File | Runs on | Purpose |
|------|---------|---------|
| `vastctl.py` | local | search / provision / deploy / monitor / pull / destroy. Reads `PHD_VAST_AI_KEY` from `Code/.env`; **never logs any secret**. |
| `bootstrap.sh` | VM | install deps → ABI-matched prebuilt flash-attn → dry-run → isolated 2-row smoke test → launch main run + auto-push. |
| `autopush.sh` | VM | every 15 min, commit `results/` (+ `state/run_state.json`) and push. Stages nothing else; scans staged diff for token/`.env` and aborts if found. |

## Secrets — how they are kept safe
- `Code/.env` is git-ignored (`.env`, `*.env`); no `.env` is tracked.
- The Vast key reaches the CLI only via the child process `VAST_API_KEY` env var — never on a command line, never printed.
- Only the keys the VM needs (`HF_Classic_Token`, `Github_Classic_Token`, `TextBelt_API_KEY`, `PHONE_NO`, `Mistral_API_Key`) are copied to the VM as `/root/peat.env` (mode 600) via `scp`, then the local temp copy is deleted.
- No script uses `set -x`. The GitHub token value is redacted from every log line.
- `autopush.sh` only stages `results/` and `state/run_state.json`; a pre-commit scan blocks any `.env`/`.pem`/`.key` file or `ghp_…`/`AKIA…`/`sk-…`/PEM-shaped string.

> ⚠️ The git remote in this clone has a PAT embedded in its URL. Rotate it in
> GitHub → Settings → Developer settings → Tokens, and set the remote to the
> plain `https://github.com/DevDaring/PEAT_Debias.git` (auth is supplied at
> push time from `.env`, not stored in the remote).

## Optimal-VRAM instance
The paper's hardware is A100-SXM4-80GB; Llama-3.1-8B peaks ~34 GB, so 80 GB is
comfortable and 48 GB also fits every model. Current cheapest reliable offers
(from `vastctl.py search`): A100-SXM4-80GB ≈ $1.04/hr, A100-PCIE-80GB ≈ $0.87/hr,
RTX-PRO-5000-48GB ≈ $0.63/hr. At the 50 GPU-h budget that is roughly $32–52.

## Usage
```bash
# 1. read-only: list offers + price
python Code/vast/vastctl.py search

# 2. provision cheapest reliable A100-80GB
python Code/vast/vastctl.py up --tier a100          # prints new contract id

# 3. once status shows 'running' + SSH is up, push secrets + launch bootstrap
python Code/vast/vastctl.py up --id <ID> --deploy

# 4. watch (dry-run → smoke → main run)
python Code/vast/vastctl.py status  --id <ID>
python Code/vast/vastctl.py monitor --id <ID> --every 30

# 5. when finished: pull results locally, verify, then destroy
python Code/vast/vastctl.py pull --id <ID>
python Code/vast/vastctl.py down --id <ID>
```

## Flash-attention note
`models.py` uses `attn_implementation="sdpa"` (or `eager`), **not**
`flash_attention_2` — the comments record a transformers-5.x / flash-attn-2.8.x
API conflict. On an A100, PyTorch SDPA already dispatches to the FlashAttention-2
kernel, so the speedup comes from the prebuilt **PyTorch** wheel. `bootstrap.sh`
still installs the matching prebuilt `flash-attn` wheel (ABI auto-detected from
`torch.compiled_with_cxx11_abi()`) for completeness; failure there is non-fatal.
