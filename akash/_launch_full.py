"""
Launch FULL production run (no --smoke-test) in tmux on VM.
Cleans smoke-test state/logs first, pulls latest code.
Log: /workspace/full_run.log
"""
import paramiko, time

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31133
USER = "root"
PASS = "peat2026!"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to {HOST}:{PORT} ...")
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
print("Connected!")

def run(cmd, timeout=60):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    return out, err

# 1. Kill any stale tmux session
out, _ = run("tmux kill-session -t peat_full 2>/dev/null; tmux kill-session -t peat 2>/dev/null; echo OK")
print(f"Kill stale sessions: {out}")

# 2. Pull latest code
out, _ = run("cd /workspace/PEAT_Debias && git pull origin main 2>&1")
print(f"Git pull: {out}")

# 3. Clean smoke-test state (run_state.json + peat checkpoints)
#    Keep: dryrun_passed, HuggingFace model cache, datasets
cmds = [
    "rm -f /workspace/PEAT_Debias/Code/state/run_state.json && echo 'run_state cleared'",
    "rm -rf /workspace/PEAT_Debias/Code/state/peat && echo 'peat checkpoints cleared'",
    "rm -f /workspace/PEAT_Debias/Code/results/raw /workspace/PEAT_Debias/Code/results/aggregated /workspace/PEAT_Debias/Code/results/figures 2>/dev/null; rm -rf /workspace/PEAT_Debias/results 2>/dev/null; echo 'results cleared'",
    # Clear old logs
    "truncate -s 0 /workspace/PEAT_Debias/Code/logs/baselines.log 2>/dev/null; true",
    "truncate -s 0 /workspace/PEAT_Debias/Code/logs/training.log 2>/dev/null; true",
    "truncate -s 0 /workspace/PEAT_Debias/Code/logs/evaluation.log 2>/dev/null; true",
    "truncate -s 0 /workspace/PEAT_Debias/Code/logs/dataset_preflight.log 2>/dev/null; true",
    "truncate -s 0 /workspace/PEAT_Debias/Code/logs/dryrun.log 2>/dev/null; true",
    # Clear workspace-level test logs (keep startup.log and install.log as reference)
    "rm -f /workspace/smoke_test.log /workspace/dryrun.log && echo 'workspace logs cleared'",
    # Clear tmp test scripts
    "rm -f /tmp/_test_nom_baselines.py 2>/dev/null; true",
    # Clear Python __pycache__ to avoid stale bytecode
    "find /workspace/PEAT_Debias/Code -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; echo 'pycache cleared'",
]
for cmd in cmds:
    out, err = run(cmd)
    if out:
        print(f"  {out}")

# 4. Create fresh full_run log
run("truncate -s 0 /workspace/full_run.log 2>/dev/null; touch /workspace/full_run.log")

# 5. Verify disk after cleanup
out, _ = run("df -h /workspace | tail -1")
print(f"Disk after clean: {out}")

# 6. Launch full pipeline in tmux (detached)
launch_cmd = (
    "tmux new-session -d -s peat_full "
    "\"cd /workspace/PEAT_Debias/Code && "
    "python3 run_all.py 2>&1 | tee /workspace/full_run.log\""
)
out, err = run(launch_cmd)
if err and "duplicate" not in err.lower():
    print(f"Launch warning: {err}")
print("Full pipeline launched in tmux session 'peat_full'")

# 7. Wait 8s and show initial output
print("Waiting 8s for startup...")
time.sleep(8)
out, _ = run("tmux capture-pane -t peat_full -p 2>/dev/null | tail -15")
print("=== Initial output ===")
print(out)

client.close()
print("\n=== PIPELINE LAUNCHED ===")
print("Monitor with:  python akash/tail_full.py")
print("Check cells:   python akash/check_cells.py")
print("Or SSH:        tail -100 /workspace/full_run.log")
