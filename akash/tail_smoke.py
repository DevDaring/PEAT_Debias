"""Tail /workspace/smoke_test.log on the VM until the process exits."""
import paramiko, sys, time

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31133
USER = "root"
PW   = "peat2026!"

LOG = "/workspace/smoke_test.log"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PW, timeout=15)
print(f"Connected — tailing {LOG}  (Ctrl-C to stop)\n{'='*60}")

# Stream the log; stop when the python3 run_all.py process is gone
_, stdout, _ = ssh.exec_command(
    f"tail -n 50 -f {LOG} &"
    " TAIL_PID=$!;"
    " while pgrep -f 'run_all.py' > /dev/null 2>&1; do sleep 5; done;"
    " sleep 3; kill $TAIL_PID 2>/dev/null",
    timeout=7200,
)
try:
    for line in iter(stdout.readline, ""):
        print(line, end="", flush=True)
except KeyboardInterrupt:
    print("\n[Interrupted]")

ssh.close()
print("\n=== Done ===")
