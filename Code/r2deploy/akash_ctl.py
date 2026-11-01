#!/usr/bin/env python3
"""
Akash Network controller for revision-2 workers (Akash Console API).

  python Code/r2deploy/akash_ctl.py balance
  python Code/r2deploy/akash_ctl.py up --account AKASH_API_KEY_08 [--deposit 2]
  python Code/r2deploy/akash_ctl.py status --account ... --dseq N
  python Code/r2deploy/akash_ctl.py close --account ... --dseq N

The container starts sshd with the local ~/.ssh/id_rsa.pub as the only
authorised key and then idles; secrets are copied afterwards with scp exactly
as for Jarvis Labs (bootstrap_r2.sh). Keys are read from Code/.env and never
printed. Accounts are tried in the order the user approved.
"""

import argparse
import base64
import json
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

ENV = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
API = "https://console-api.akash.network"
ACCOUNTS = ["AKASH_API_KEY_08", "AKASH_API_2009_KEY", "AKASH_API_KEY_88", "Bharat_AKASH_API_KEY"]
GPUS = [("rtx4090", "24Gi"), ("rtx3090", "24Gi"), ("a100", "40Gi"), ("a100", "80Gi"), ("a6000", "48Gi"),
        ("l40s", "48Gi"), ("a40", "48Gi"), ("h100", "80Gi"), ("rtx6000ada", "48Gi")]
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"


def api(key, method, ep, body=None):
    r = requests.request(method, API + ep, headers={"x-api-key": key, "Content-Type": "application/json"},
                         json=body, timeout=60)
    if not r.ok:
        raise RuntimeError(f"{method} {ep} -> {r.status_code}: {r.text[:400]}")
    return r.json()


def sdl(max_price_uact: int) -> str:
    pub = (Path.home() / ".ssh" / "id_rsa.pub").read_text().strip()
    start = f"""set -e
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server git curl >/dev/null
mkdir -p /root/.ssh /run/sshd && echo '{pub}' > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
sed -i 's/#\\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
/usr/sbin/sshd
sleep infinity
"""
    b64 = base64.b64encode(start.encode()).decode()
    gpu = "\n".join(f"                - model: {m}\n                  ram: {r}" for m, r in GPUS)
    return f"""version: "2.0"
services:
  peat:
    image: {IMAGE}
    env:
      - "S={b64}"
    command: ["/bin/bash", "-c", "echo $S | base64 -d | bash"]
    expose:
      - port: 22
        as: 22
        to:
          - global: true
profiles:
  compute:
    peat:
      resources:
        cpu: {{units: 8}}
        memory: {{size: 48Gi}}
        storage:
          - size: 120Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
{gpu}
  placement:
    akash:
      pricing:
        peat: {{denom: uact, amount: {max_price_uact}}}
deployment:
  peat:
    akash: {{profile: peat, count: 1}}
"""


def balance(key):
    return api(key, "GET", "/v1/balances")["data"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action")
    ap.add_argument("--account", default=ACCOUNTS[0])
    ap.add_argument("--deposit", type=float, default=2.0)
    ap.add_argument("--max-price", type=int, default=600)  # uact/block (~$0.36/h at 6 blocks/min)
    ap.add_argument("--dseq")
    ap.add_argument("--wait-bids", type=int, default=180)
    a = ap.parse_args()
    if a.action == "balance":
        for acc in ACCOUNTS:
            try:
                print(acc, balance(ENV[acc].strip()))
            except Exception as e:
                print(acc, "error", str(e)[:120])
        return
    key = ENV[a.account].strip()
    if a.action == "up":
        d = api(key, "POST", "/v1/deployments", {"data": {"sdl": sdl(a.max_price), "deposit": a.deposit}})["data"]
        dseq, manifest = d["dseq"], d["manifest"]
        Path(__file__).with_name(f"akash_{a.account}_manifest.txt").write_text(manifest)
        print("dseq", dseq)
        t0, bids = time.time(), []
        while time.time() - t0 < a.wait_bids:
            bids = api(key, "GET", f"/v1/bids?dseq={dseq}").get("data", [])
            if bids:
                time.sleep(20)
                bids = api(key, "GET", f"/v1/bids?dseq={dseq}").get("data", [])
                break
            time.sleep(6)
        if not bids:
            print("no bids; closing")
            api(key, "DELETE", f"/v1/deployments/{dseq}")
            return
        for b in sorted(bids, key=lambda b: float(b["bid"]["price"]["amount"])):
            print("bid", b["bid"]["id"]["provider"][-8:], b["bid"]["price"]["amount"], b["bid"]["price"]["denom"],
                  json.dumps(b["bid"].get("resources_offer", ""))[:160])
        best = min(bids, key=lambda b: float(b["bid"]["price"]["amount"]))
        bid = best["bid"]["id"]
        api(key, "POST", "/v1/leases", {"manifest": manifest, "leases": [
            {"dseq": dseq, "gseq": bid["gseq"], "oseq": bid["oseq"], "provider": bid["provider"]}]})
        print("lease created at", best["bid"]["price"]["amount"], "uact/block")
    elif a.action == "status":
        print(json.dumps(api(key, "GET", f"/v1/leases?dseq={a.dseq}"), indent=1)[:3000])
    elif a.action == "close":
        print(api(key, "DELETE", f"/v1/deployments/{a.dseq}"))
    elif a.action == "list":
        print(json.dumps(api(key, "GET", "/v1/deployments?limit=10"), indent=1)[:2000])


if __name__ == "__main__":
    main()
