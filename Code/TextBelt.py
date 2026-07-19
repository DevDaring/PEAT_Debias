import requests
import atexit
import sys
import signal
import traceback

# ================== CONFIG ==================
PHONE = "PHONE_NO from .env"      # Your Airtel number with +91
API_KEY= "TextBelt_API_KEY from .env"   # Paste your $3 TextBelt key here
# ==========================================

def send_sms(msg: str):
    try:
        r = requests.post(
            "https://textbelt.com/text",
            data={"phone": PHONE, "message": msg, "key": API_KEY},
            timeout=20
        )
        data = r.json()
        if data.get("success"):
            print(f"[Notifier] SMS queued. Quota left: {data.get('quotaRemaining')}")
        else:
            print(f"[Notifier] TextBelt error: {data.get('error')}")
    except Exception as e:
        print(f"[Notifier] Failed: {e}")

# 1) Normal exit / crash
atexit.register(send_sms, "✅ VM Program finished or crashed!")

# 2) SIGTERM / Ctrl+C
def on_signal(signum, frame):
    send_sms("⚠️ VM Program was interrupted")
    sys.exit(0)

signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)

if __name__ == "__main__":
    try:
        # ========== YOUR PROGRAM HERE ==========
        import time
        print("Working...")
        time.sleep(5)
        # =======================================

    except Exception as e:
        err = traceback.format_exc()[-400:]
        send_sms(f"❌ VM crashed:\n{err}")
        raise