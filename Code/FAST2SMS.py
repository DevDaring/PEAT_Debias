"""Fast2SMS smoke-test — standalone script.

Usage:
    python Code/FAST2SMS.py [optional message]

Reads FAST2SMS_API_KEY and PHONE_NO from Code/.env (the same .env used by the
pipeline).  If the keys are absent the script exits cleanly with an
explanation instead of crashing.

Exit codes:
  0 — SMS sent and Fast2SMS returned {"return": true}
  1 — key missing, or HTTP error, or Fast2SMS returned {"return": false}
"""

import os
import sys
from pathlib import Path

# ── Load .env from the Code/ directory (same location the pipeline uses) ──
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
    else:
        print(f"[FAST2SMS] WARNING: .env not found at {_env_path}", file=sys.stderr)
except ImportError:
    print("[FAST2SMS] WARNING: python-dotenv not installed; reading os.environ only",
          file=sys.stderr)

import requests

FAST2SMS_API_KEY = os.environ.get("FAST2SMS_API_KEY", "").strip()
PHONE_NO = os.environ.get("PHONE_NO", "").strip()

_API_URL = "https://www.fast2sms.com/dev/bulkV2"


def send_sms(message: str) -> bool:
    """Send *message* via Fast2SMS.  Returns True on confirmed delivery.

    Reads credentials from environment (populated from .env by the loader
    at the top of this module).

    Args:
        message: Text to send (truncated to 160 characters).

    Returns:
        True if Fast2SMS confirmed delivery (HTTP 200 + ``{"return": true}``).
        False on any error, including missing credentials.
    """
    if not FAST2SMS_API_KEY:
        print("[FAST2SMS] FAST2SMS_API_KEY not set in .env — cannot send SMS.")
        return False

    if not PHONE_NO:
        print("[FAST2SMS] PHONE_NO not set in .env — cannot send SMS.")
        return False

    # Fast2SMS requires digits only; strip +, spaces, country-code formatting
    phone_digits = PHONE_NO.replace("+", "").replace(" ", "").replace("-", "").strip()

    payload = {
        "sender_id": "FSTSMS",
        "message":   message[:160],
        "language":  "english",
        "route":     "q",
        "numbers":   phone_digits,
    }
    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type":  "application/x-www-form-urlencoded",
    }

    try:
        resp = requests.post(_API_URL, data=payload, headers=headers, timeout=20)
    except requests.exceptions.RequestException as exc:
        print(f"[FAST2SMS] Network error: {exc}")
        return False

    print(f"[FAST2SMS] HTTP {resp.status_code}")
    try:
        body = resp.json()
        print(f"[FAST2SMS] Response body: {body}")
    except ValueError:
        print(f"[FAST2SMS] Raw response: {resp.text[:400]}")
        return False

    if resp.status_code != 200:
        print(f"[FAST2SMS] ERROR: expected HTTP 200, got {resp.status_code}")
        return False

    confirmed = body.get("return") is True
    if confirmed:
        print("[FAST2SMS] SMS delivered successfully.")
    else:
        print(f"[FAST2SMS] Fast2SMS did not confirm delivery: {body}")
    return confirmed


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "PEAT test notification: pipeline OK"
    print(f"[FAST2SMS] Sending: {msg!r}")
    success = send_sms(msg)
    sys.exit(0 if success else 1)
