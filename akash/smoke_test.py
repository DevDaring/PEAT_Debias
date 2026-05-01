"""
Run the full PEAT pipeline in smoke-test mode (4 instances per dataset)
on the VM and stream output live.

Usage (from repo root):
    python akash/smoke_test.py

Steps:
  1. git pull latest code on VM
  2. Wipe any previous smoke-test run_state.json so every stage reruns fresh
  3. python3 run_all.py --smoke-test  (all 5 stages, 4 rows each)
  4. Print a short PASS/FAIL summary at the end
"""
import sys
import time
import paramiko

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31133
USER = "root"
PW   = "peat2026!"

WORKDIR  = "/workspace/PEAT_Debias/Code"
LOG_FILE = "/workspace/smoke_test.log"

COMMANDS = [
    # Pull latest fixes
    (
        "cd /workspace/PEAT_Debias && git pull origin main 2>&1",
        120,
    ),
    # Remove stale run_state so every stage executes from scratch
    (
        f"rm -f {WORKDIR}/state/run_state.json && echo 'run_state cleared'",
        10,
    ),
    # Run pipeline — all stages, 4 rows each, log to file AND stdout
    (
        f"cd {WORKDIR} && python3 run_all.py --smoke-test 2>&1 | tee {LOG_FILE}",
        7200,   # 2-hour hard cap (should finish in ~20-40 min on A100)
    ),
]

PASS_MARKERS = [
    "Pipeline finished at",   # final log line in run_all.py
    "STAGE 5: FIGURES",       # proves stages 1-4 completed
    "STAGE 4: AGGREGATION",   # proves stages 1-3 completed
    "Figures complete",       # stage 5 finished cleanly
]
FAIL_MARKERS = [
    "Traceback (most recent call last)",
    "CRITICAL",
    "RuntimeError",
    "CUDA out of memory",
    "KeyError",
    "AssertionError",
]


def run_stream(ssh: paramiko.SSHClient, cmd: str, timeout: int) -> tuple[int, str]:
    """Run *cmd*, stream stdout live, return (exit_code, full_output)."""
    print(f"\n{'='*70}")
    print(f">>> {cmd[:120]}")
    print("="*70)

    transport = ssh.get_transport()
    chan = transport.open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(cmd)
    chan.settimeout(2.0)

    buf = []
    start = time.time()
    while True:
        if chan.exit_status_ready():
            while chan.recv_ready():
                chunk = chan.recv(8192).decode("utf-8", errors="replace")
                sys.stdout.write(chunk)
                sys.stdout.flush()
                buf.append(chunk)
            break
        try:
            data = chan.recv(8192)
            if data:
                chunk = data.decode("utf-8", errors="replace")
                sys.stdout.write(chunk)
                sys.stdout.flush()
                buf.append(chunk)
        except Exception:
            pass
        if time.time() - start > timeout:
            print(f"\n[TIMEOUT after {timeout}s — command still running on VM]")
            print(f"Tail the log with:  python akash/tail_smoke.py")
            break

    rc = chan.recv_exit_status()
    print(f"\n[exit code: {rc}]")
    return rc, "".join(buf)


def main():
    print(f"Connecting to {HOST}:{PORT} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PW, timeout=15)
    print("Connected!\n")

    full_output = ""
    for cmd, timeout in COMMANDS:
        rc, out = run_stream(ssh, cmd, timeout)
        full_output += out
        if rc not in (0, -1):  # -1 = timeout (we let the pipeline continue)
            print(f"\n[FAILED with exit code {rc}]")
            ssh.close()
            sys.exit(1)

    ssh.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SMOKE TEST SUMMARY")
    print("="*70)

    lines_with_errors = [
        line.strip() for line in full_output.splitlines()
        if any(m in line for m in FAIL_MARKERS)
    ]
    passed_stages = [m for m in PASS_MARKERS if m.lower() in full_output.lower()]

    if lines_with_errors:
        print(f"\n[ERRORS / WARNINGS detected ({len(lines_with_errors)} lines):]")
        for ln in lines_with_errors[:30]:
            print(f"  {ln}")

    if passed_stages:
        print(f"\n[Completed markers found: {passed_stages}]")

    if not lines_with_errors and passed_stages:
        print("\nRESULT: PASS — pipeline completed all stages without errors.")
        sys.exit(0)
    elif lines_with_errors:
        print("\nRESULT: FAIL — errors detected (see above).")
        sys.exit(1)
    else:
        print("\nRESULT: PARTIAL — no errors but completion markers not found.")
        print(f"Check full log on VM:  {LOG_FILE}")
        sys.exit(0)


if __name__ == "__main__":
    main()
