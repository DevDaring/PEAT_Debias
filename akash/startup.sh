#!/usr/bin/env bash
# =============================================================================
# PEAT — Akash Container Startup Script
#
# Runs inside nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 on first boot.
# Injected into container as ENV_VAR STARTUP_B64 (base64-encoded) by deploy.py.
# Secrets arrive via GIT_TOKEN and ENV_B64 env vars — never baked into image.
# =============================================================================
set -euo pipefail

mkdir -p /workspace
LOG=/workspace/startup.log
exec > >(tee -a "$LOG") 2>&1

echo "======================================================"
echo " PEAT Container Startup — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================"

# ── 1. System packages ────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y \
    openssh-server curl wget git \
    build-essential ninja-build \
    software-properties-common ca-certificates \
    tmux htop vim nano lsb-release gnupg \
    2>/dev/null
echo "  System packages OK."

# ── 2. Python 3.12 (deadsnakes PPA) ──────────────────────────────────────
echo "[2/7] Installing Python 3.12..."
add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null
apt-get update -q 2>/dev/null
apt-get install -y python3.12 python3.12-dev python3.12-venv 2>/dev/null
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100
update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 100
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 2>/dev/null
echo "  Python: $(python3 --version)"

# ── 3. SSH — root login, password auth (minimum security for single project)
echo "[3/7] Configuring SSH..."
mkdir -p /run/sshd /root/.ssh
ssh-keygen -A 2>/dev/null  # generate host keys if missing
{
    echo "PermitRootLogin yes"
    echo "PasswordAuthentication yes"
    echo "ChallengeResponseAuthentication no"
    echo "UsePAM no"
    echo "Port 22"
    echo "X11Forwarding no"
    echo "AcceptEnv LANG LC_*"
} > /etc/ssh/sshd_config.d/99-peat.conf
echo "root:peat2026!" | chpasswd
/usr/sbin/sshd       # daemonises
echo "  SSH ready — root / peat2026!"

# Write MOTD so user sees status on login
cat > /etc/motd << 'MOTD'

  ╔═══════════════════════════════════════════════════════╗
  ║          PEAT — Akash A100 Research Container          ║
  ╠═══════════════════════════════════════════════════════╣
  ║  Dry-run log : /workspace/dryrun.log                  ║
  ║  Setup log   : /workspace/startup.log                 ║
  ║  Attach tmux : tmux attach -t peat                    ║
  ║  Run pipeline: cd /workspace/PEAT_Debias/Code         ║
  ║                python3 run_all.py                     ║
  ╚═══════════════════════════════════════════════════════╝

MOTD

# ── 4. Clone repository ───────────────────────────────────────────────────
echo "[4/7] Cloning repository..."
cd /workspace
git clone "https://${GIT_TOKEN}@github.com/DevDaring/PEAT_Debias.git" PEAT_Debias \
    2>&1 | grep -v "Cloning into"
echo "  Repo cloned to /workspace/PEAT_Debias"

# ── 5. Restore .env from base64 payload ──────────────────────────────────
echo "[5/7] Restoring .env..."
echo "${ENV_B64}" | base64 -d > /workspace/PEAT_Debias/Code/.env
echo "  .env written."

# ── 6. Install Python packages (global, no venv) ─────────────────────────
echo "[6/7] Installing Python packages (this takes ~25 min)..."
cd /workspace/PEAT_Debias/Code
bash install.sh 2>&1 | tee /workspace/install.log
echo "  Packages installed."

# ── 7. Launch dry run in a detached tmux session ─────────────────────────
echo "[7/7] Starting dry run in tmux session 'peat'..."

# Write a self-contained dry-run script
cat > /workspace/run_dryrun.sh << 'SCRIPT'
#!/usr/bin/env bash
cd /workspace/PEAT_Debias/Code
echo "=== PEAT Dry Run — $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
python3 -c "
import sys, os
os.chdir('/workspace/PEAT_Debias/Code')
sys.path.insert(0, '.')
from peat.dryrun import run_dryrun
ok = run_dryrun(skip_if_recent=False)
print()
print('=== DRY RUN:', 'PASSED ✓' if ok else 'FAILED ✗', '===')
sys.exit(0 if ok else 1)
" 2>&1 | tee /workspace/dryrun.log
echo "Dry run exit code: $?"
SCRIPT
chmod +x /workspace/run_dryrun.sh

tmux new-session -d -s peat -x 220 -y 50
tmux send-keys -t peat "bash /workspace/run_dryrun.sh" Enter
echo "  Dry run running in tmux — SSH in and: tmux attach -t peat"

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Container ready.  SSH: root@<host>:<port>  pw: peat2026!"
echo " Dry-run status:   tail -f /workspace/dryrun.log"
echo " Attach session:   tmux attach -t peat"
echo "======================================================"

# Keep container alive as PID 1
tail -f /dev/null
