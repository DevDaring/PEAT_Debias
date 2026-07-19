#!/usr/bin/env bash
# ============================================================================
# autopush.sh — periodic, secret-safe checkpoint of results/ to GitHub.
#
# Runs ON the VM. Guards against pre-emption: every INTERVAL seconds it commits
# results/ (+ the small resumable state file) and pushes to the repo.
#
# SECRETS POLICY (hard requirements):
#   * NO `set -x` anywhere (would echo the token).
#   * Token read only from $GH_TOKEN env; never written to a tracked file,
#     never printed. Any token value is redacted from all log output.
#   * Only results/ and state/run_state.json are ever staged. A pre-commit
#     secret scan aborts the commit if a .env/key file or token-shaped string
#     is somehow staged.
# ============================================================================
set -uo pipefail   # deliberately NO -x

REPO_DIR="${REPO_DIR:-$HOME/PEAT_Debias}"
BRANCH="${AUTOPUSH_BRANCH:-main}"
INTERVAL="${AUTOPUSH_INTERVAL:-900}"          # 15 minutes
SLUG="${AUTOPUSH_SLUG:-DevDaring/PEAT_Debias}"
LOG="${AUTOPUSH_LOG:-$HOME/autopush.log}"

log() { echo "[autopush $(date -u +%H:%M:%SZ)] $*" >> "$LOG"; }

redact() { # strip the token from anything before it is logged
  local s="$1"
  [ -n "${GH_TOKEN:-}" ] && s="${s//$GH_TOKEN/***}"
  printf '%s' "$s" | sed -E 's#(x-access-token|ghp_[A-Za-z0-9]*):[^@]*@#\1:***@#g'
}

cd "$REPO_DIR" 2>/dev/null || { log "repo dir '$REPO_DIR' missing"; exit 1; }
if [ -z "${GH_TOKEN:-}" ]; then log "GH_TOKEN unset — refusing to run"; exit 1; fi

# Belt-and-suspenders: make sure secrets can never be staged on this clone.
for pat in '.env' '*.env' 'secrets.json' '*.pem' '*.key'; do
  grep -qxF "$pat" .gitignore 2>/dev/null || printf '%s\n' "$pat" >> .gitignore
done
git config user.email "peat-vm@autopush.local"
git config user.name  "PEAT VM Autopush"

secret_scan() {
  local staged; staged="$(git diff --cached --name-only)"
  if printf '%s\n' "$staged" | grep -Eiq '(^|/)\.env$|\.env$|(^|/)secrets\.json$|\.pem$|\.key$'; then
    log "BLOCKED: secret-like filename staged"; return 1
  fi
  if git diff --cached -- . | grep -Eq 'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----'; then
    log "BLOCKED: token-shaped content in staged diff"; return 1
  fi
  return 0
}

push_once() {
  # Stage only results + the small resumable state file (force, since state/ is gitignored).
  git add -A -- results Code/results 2>/dev/null || true
  [ -f state/run_state.json ]      && git add -f state/run_state.json 2>/dev/null || true
  [ -f Code/state/run_state.json ] && git add -f Code/state/run_state.json 2>/dev/null || true

  if git diff --cached --quiet; then log "no changes"; return 0; fi
  if ! secret_scan; then git reset -q; return 1; fi

  git commit -q -m "results: VM autosave $(date -u +%Y-%m-%dT%H:%M:%SZ)" || { log "commit noop"; return 0; }

  local out rc
  out="$(git -c credential.helper= \
        push "https://x-access-token:${GH_TOKEN}@github.com/${SLUG}.git" "HEAD:${BRANCH}" 2>&1)"
  rc=$?
  log "$(redact "$out")"
  if [ $rc -eq 0 ]; then log "pushed to ${BRANCH}"; else log "push failed rc=$rc (token redacted)"; fi
}

log "starting: interval=${INTERVAL}s branch=${BRANCH} repo=${REPO_DIR}"
trap 'log "final push before exit"; push_once; log "stopped"; exit 0' TERM INT
while true; do push_once; sleep "$INTERVAL"; done
