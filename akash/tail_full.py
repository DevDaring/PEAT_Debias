"""Tail the full production run log (/workspace/full_run.log)."""
import paramiko, time, sys

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31133
USER = "root"
PASS = "peat2026!"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
print("Connected — tailing /workspace/full_run.log  (Ctrl-C to stop)")
print("=" * 60)

transport = client.get_transport()
channel = transport.open_session()
channel.get_pty()
channel.exec_command("tail -f /workspace/full_run.log")

try:
    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode(errors="replace")
            print(data, end="", flush=True)
        elif channel.exit_status_ready():
            break
        else:
            time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    channel.close()
    client.close()
