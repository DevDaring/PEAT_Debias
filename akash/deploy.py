#!/usr/bin/env python3
"""
PEAT — Akash A100 Deployment Script
=====================================
Creates an A100 40GB VM on Akash Network, installs all packages,
pulls the repo, restores .env, and starts the dry run automatically.

Usage (from repo root):
    python akash/deploy.py

What it does:
    1. Reads secrets from Code/.env
    2. Builds the Akash SDL (base64-encodes .env + startup.sh — keeps SDL clean)
    3. POST /v1/deployments  — creates on-chain deployment ($5 deposit)
    4. GET  /v1/bids?dseq=   — waits for provider bids (polls every 5 s)
    5. POST /v1/leases       — accepts cheapest bid, starts container
    6. Polls lease status until SSH host/port are available
    7. Prints SSH connection command

Container lifecycle:
    - SSH root / peat2026!  (port shown after deployment)
    - Dry run auto-starts in tmux session 'peat'
    - Attach with: tmux attach -t peat
    - Run full pipeline: python3 run_all.py (from /workspace/PEAT_Debias/Code)

To close the deployment (stops billing):
    python akash/deploy.py --close <dseq>
"""

import base64
import json
import sys
import time
import argparse
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
ENV_FILE    = REPO_ROOT / "Code" / ".env"
STARTUP_SH  = Path(__file__).resolve().parent / "startup.sh"

# ── Akash Console API ─────────────────────────────────────────────────────
API_BASE    = "https://console-api.akash.network"
USDC_DENOM  = "ibc/170C677610AC31DF0904FFE09CD3B5C657492170E7E52372E48756B71E56F2F1"

# ── SDL constants ──────────────────────────────────────────────────────────
# Max bid price: 50 000 µUSDC/block (~$30/hour).  Actual A100 bids ~$2-4/hour.
MAX_BID_UUSDC   = 50_000
DEPOSIT_USD     = 20        # initial escrow deposit in dollars (≥ $5 required)
CPU_UNITS       = 12
MEMORY_GB       = 64
STORAGE_GB      = 250       # models (~25 GB) + datasets (~5 GB) + OS/packages
GPU_MODEL       = "a100"
GPU_RAM         = "40Gi"
GPU_INTERFACE   = "pcie"
# Base image: CUDA 12.4 + cuDNN + Ubuntu 22.04 (matches install.sh requirements)
CONTAINER_IMAGE = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"


# =============================================================================
# Helpers
# =============================================================================

def _ensure_requests():
    """Import requests, install if missing."""
    try:
        import requests as _r
        return _r
    except ImportError:
        import subprocess
        print("Installing 'requests' library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests", "-q"],
                       check=True)
        import requests as _r
        return _r


def read_env(path: Path) -> dict:
    """Parse .env file -> {KEY: VALUE} (strips quotes, skips comments)."""
    env: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get_required(env: dict, *names: str) -> str:
    """Return first non-empty match from env dict; raise if none found."""
    for name in names:
        if env.get(name, "").strip():
            return env[name].strip()
    raise KeyError(f"None of {names} found in .env — please add it.")


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# =============================================================================
# SDL builder
# =============================================================================

def build_sdl(git_token: str, env_b64: str, startup_b64: str) -> str:
    """
    Build the Akash SDL YAML string.

    Secrets are passed as env vars — they live only in the SDL string in memory
    (never written to disk unless the user explicitly saves it).
    The container command decodes STARTUP_B64 and executes it.
    """
    sdl = f"""\
version: "2.0"

services:
  peat:
    image: {CONTAINER_IMAGE}
    env:
      - "GIT_TOKEN={git_token}"
      - "ENV_B64={env_b64}"
      - "STARTUP_B64={startup_b64}"
    command:
      - /bin/bash
      - -c
      - "echo $STARTUP_B64 | base64 -d | bash"
    expose:
      - port: 22
        as: 22
        to:
          - global: true

profiles:
  compute:
    peat:
      resources:
        cpu:
          units: {CPU_UNITS}
        memory:
          size: {MEMORY_GB}Gi
        storage:
          - size: {STORAGE_GB}Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
                - model: {GPU_MODEL}
                  ram: {GPU_RAM}
                  interface: {GPU_INTERFACE}
  placement:
    akash:
      pricing:
        peat:
          denom: {USDC_DENOM}
          amount: {MAX_BID_UUSDC}

deployment:
  peat:
    akash:
      profile: peat
      count: 1
"""
    return sdl


# =============================================================================
# Akash Console API calls
# =============================================================================

def api(requests_lib, method: str, endpoint: str, akash_key: str,
        body=None) -> dict:
    """Single authenticated request to the Akash Console API."""
    url = f"{API_BASE}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Accept":       "application/json",
        "x-api-key":    akash_key,
        "User-Agent":   "PEAT-Deployer/1.0",
    }
    resp = requests_lib.request(
        method, url, headers=headers,
        json=body, timeout=60,
        verify=True,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Akash API {method} {endpoint} -> HTTP {resp.status_code}:\n{resp.text}"
        )
    return resp.json()


def create_deployment(req, akash_key: str, sdl: str) -> tuple[str, str]:
    """POST /v1/deployments — returns (dseq, manifest)."""
    print(f"  Creating deployment (deposit: ${DEPOSIT_USD})...")
    resp = api(req, "POST", "/v1/deployments", akash_key,
               body={"data": {"sdl": sdl, "deposit": DEPOSIT_USD}})
    dseq     = resp["data"]["dseq"]
    manifest = resp["data"]["manifest"]
    print(f"  Deployment created — dseq: {dseq}")
    return dseq, manifest


def wait_for_bids(req, akash_key: str, dseq: str,
                  max_attempts: int = 30, interval: int = 5) -> list:
    """
    Poll GET /v1/bids?dseq= until bids appear.
    Typical wait: 30-60 s.  max_attempts × interval = 150 s timeout.
    """
    print(f"  Waiting for provider bids (up to {max_attempts * interval}s)...",
          flush=True)
    for attempt in range(1, max_attempts + 1):
        resp = api(req, "GET", f"/v1/bids?dseq={dseq}", akash_key)
        bids = resp.get("data", [])
        if bids:
            print(f"  {len(bids)} bid(s) received.")
            return bids
        print(f"    [{attempt}/{max_attempts}] no bids yet...", end="\r", flush=True)
        time.sleep(interval)
    raise TimeoutError("No bids received — try increasing MAX_BID_UUSDC or retry later.")


def select_cheapest_bid(bids: list) -> dict:
    """Return the bid with the lowest price amount."""
    return min(bids, key=lambda b: int(b["bid"]["price"]["amount"]))


def create_lease(req, akash_key: str, manifest: str,
                 dseq: str, bid: dict) -> dict:
    """POST /v1/leases — accept the selected bid."""
    bid_id = bid["bid"]["id"]
    print(f"  Accepting bid from provider: {bid_id['provider']}")
    print(f"  Price: {bid['bid']['price']['amount']} {bid['bid']['price']['denom']}")
    resp = api(req, "POST", "/v1/leases", akash_key, body={
        "manifest": manifest,
        "leases": [{
            "dseq":     dseq,
            "gseq":     bid_id["gseq"],
            "oseq":     bid_id["oseq"],
            "provider": bid_id["provider"],
        }],
    })
    print("  Lease created — container starting...")
    return resp["data"]


def wait_for_ssh(req, akash_key: str, dseq: str,
                 max_attempts: int = 60, interval: int = 10) -> tuple[str, int]:
    """
    Poll lease status until the SSH forwarded port is available.
    Returns (host, external_port).
    """
    print(f"  Waiting for SSH port assignment (container is installing packages ~25 min)...")
    print("  You can also check: https://console.akash.network")
    for attempt in range(1, max_attempts + 1):
        try:
            resp = api(req, "GET", f"/v1/leases?dseq={dseq}", akash_key)
            leases = resp.get("data", {}).get("leases", [])
            for lease in leases:
                status = lease.get("status") or {}
                fwd = status.get("forwarded_ports", {})
                for _svc, ports in fwd.items():
                    for p in ports:
                        if p.get("port") == 22 and p.get("externalPort"):
                            host = p.get("host", "")
                            ext  = int(p["externalPort"])
                            if host:
                                return host, ext
        except Exception:
            pass  # lease status may not be available immediately
        print(f"    [{attempt}/{max_attempts}] container starting...",
              end="\r", flush=True)
        time.sleep(interval)
    # Timed out — return placeholder
    return "<provider-host>", 0


def close_deployment(req, akash_key: str, dseq: str):
    """DELETE /v1/deployments/{dseq} — stops billing and releases resources."""
    resp = api(req, "DELETE", f"/v1/deployments/{dseq}", akash_key)
    ok = resp.get("data", {}).get("success", False)
    print(f"  Deployment {dseq} closed: {ok}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PEAT Akash Deployment")
    parser.add_argument("--show-sdl", action="store_true",
                        help="Print the SDL and exit (no deployment).")
    parser.add_argument("--close", metavar="DSEQ",
                        help="Close an existing deployment by dseq.")
    args = parser.parse_args()

    req = _ensure_requests()

    # ── Load secrets ──────────────────────────────────────────────────────
    if not ENV_FILE.exists():
        sys.exit(f"ERROR: {ENV_FILE} not found. Create it from Code/.env.template")
    if not STARTUP_SH.exists():
        sys.exit(f"ERROR: {STARTUP_SH} not found.")

    env         = read_env(ENV_FILE)
    akash_key   = get_required(env, "Akash_API_Key", "AKASH_API_KEY")
    git_token   = get_required(env, "Github_Classic_Token", "GITHUB_TOKEN")
    env_content = ENV_FILE.read_text(encoding="utf-8")
    startup_content = STARTUP_SH.read_text(encoding="utf-8")

    env_b64     = b64(env_content)
    startup_b64 = b64(startup_content)

    # ── Close mode ────────────────────────────────────────────────────────
    if args.close:
        print(f"Closing deployment {args.close}...")
        close_deployment(req, akash_key, args.close)
        return

    # ── Build SDL ─────────────────────────────────────────────────────────
    sdl = build_sdl(git_token, env_b64, startup_b64)

    if args.show_sdl:
        # Print SDL with secrets redacted
        safe = sdl.replace(git_token, "GIT_TOKEN_REDACTED")
        safe = safe.replace(env_b64,  "ENV_B64_REDACTED")
        safe = safe.replace(startup_b64, "STARTUP_B64_REDACTED")
        print(safe)
        return

    # ── Deploy ────────────────────────────────────────────────────────────
    print("=" * 60)
    print(" PEAT — Akash A100 Deployment")
    print("=" * 60)
    print(f"  Image   : {CONTAINER_IMAGE}")
    print(f"  GPU     : {GPU_MODEL} {GPU_RAM} {GPU_INTERFACE}")
    print(f"  CPU     : {CPU_UNITS} cores")
    print(f"  Memory  : {MEMORY_GB} GB")
    print(f"  Storage : {STORAGE_GB} GB")
    print(f"  Max bid : {MAX_BID_UUSDC} µUSDC/block (~${MAX_BID_UUSDC/1_000_000*600:.0f}/hr max)")
    print()

    try:
        # Step 1: Create deployment
        dseq, manifest = create_deployment(req, akash_key, sdl)

        # Step 2: Wait for bids
        bids = wait_for_bids(req, akash_key, dseq)

        # Step 3: Select cheapest bid + create lease
        best_bid = select_cheapest_bid(bids)
        price_per_block = int(best_bid["bid"]["price"]["amount"])
        price_per_hour  = price_per_block / 1_000_000 * 600
        print(f"  Best bid: {price_per_block} µUSDC/block (~${price_per_hour:.2f}/hr)")
        lease_data = create_lease(req, akash_key, manifest, dseq, best_bid)

        # Step 4: Wait for SSH
        host, port = wait_for_ssh(req, akash_key, dseq)

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nIf you see HTTP 403, check your Akash_API_Key in Code/.env")
        sys.exit(1)

    # ── Print connection info ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" DEPLOYMENT LIVE")
    print("=" * 60)
    print(f"  DSEQ    : {dseq}")
    print(f"  Provider: {best_bid['bid']['id']['provider']}")
    print(f"  Cost    : ~${price_per_hour:.2f}/hour")
    print()
    if port:
        print(f"  SSH     : ssh root@{host} -p {port}")
        print(f"  Password: peat2026!")
    else:
        print("  SSH port not yet assigned — check console.akash.network")
        print(f"  Once available:  ssh root@<host> -p <port>   pw: peat2026!")
    print()
    print("  Container is installing packages (~25 min).")
    print("  After install, dry run starts automatically in tmux.")
    print("  SSH in and: tmux attach -t peat")
    print()
    print("  To monitor install progress before SSH is ready:")
    print(f"    https://console.akash.network/deployments/{dseq}")
    print()
    print("  To close deployment when done:")
    print(f"    python akash/deploy.py --close {dseq}")
    print("=" * 60)

    # Save dseq for reference
    dseq_file = Path(__file__).parent / "last_dseq.txt"
    dseq_file.write_text(f"dseq={dseq}\nprovider={best_bid['bid']['id']['provider']}\n"
                         f"ssh=root@{host}:{port}\n")
    print(f"  Connection info saved to: akash/last_dseq.txt")


if __name__ == "__main__":
    main()
