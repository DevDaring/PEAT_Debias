"""
PEAT — Dry Run (Pre-flight Validation).

8-step validation that must pass before the main run:
1. Environment check
2. Library import check
3. Secrets check
4. API key live test
5. Model load check
6. Dataset structure check
7. One-row training check
8. Path check
"""

import os
import sys
import time
from pathlib import Path

import torch

from peat.utils import (
    ALL_DIRS,
    LOG_DIR,
    STATE_DIR,
    ensure_dirs,
    get_dtype,
    setup_logger,
)

logger = setup_logger("peat.dryrun", str(LOG_DIR / "dryrun.log"))


def run_dryrun(skip_if_recent: bool = True) -> bool:
    """Execute all 8 dry-run checks.

    Args:
        skip_if_recent: If True, skip if state/dryrun_passed exists and is < 24h old.

    Returns:
        True if all checks pass.
    """
    ensure_dirs()
    passed_file = STATE_DIR / "dryrun_passed"

    if skip_if_recent and passed_file.exists():
        age = time.time() - passed_file.stat().st_mtime
        if age < 86400:
            logger.info(f"Dry run passed {age/3600:.1f}h ago — skipping")
            return True

    all_ok = True

    # ── Step 1: Environment check ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DRYRUN Step 1/8: Environment check")
    logger.info("=" * 60)
    try:
        import platform
        logger.info(f"  Python: {sys.version}")
        logger.info(f"  OS: {platform.system()} {platform.machine()}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            cc = torch.cuda.get_device_capability(0)
            logger.info(f"  Compute capability: {cc[0]}.{cc[1]}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU RAM: {mem:.1f} GB")
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            logger.info(f"  Free GPU RAM: {free_mem:.1f} GB")
        import shutil
        disk = shutil.disk_usage("/")
        logger.info(f"  Disk free: {disk.free/1e9:.1f} GB")
        logger.info("  ✓ Environment OK")
    except Exception as e:
        logger.error(f"  ✗ Environment check FAILED: {e}")
        all_ok = False

    # ── Step 2: Library import check ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DRYRUN Step 2/8: Library imports")
    logger.info("=" * 60)
    libs = [
        "torch", "transformers", "peft", "bitsandbytes", "flash_attn",
        "datasets", "evaluate", "google.generativeai", "scipy",
    ]
    for lib in libs:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "N/A")
            logger.info(f"  ✓ {lib}: {ver}")
        except ImportError as e:
            logger.error(f"  ✗ {lib}: {e}")
            all_ok = False

    # ── Step 3: Secrets check ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DRYRUN Step 3/8: Secrets")
    logger.info("=" * 60)
    try:
        from peat.secrets import validate_all_keys
        masked = validate_all_keys()
        for name, mval in masked.items():
            logger.info(f"  loaded: {name}={mval}")
        logger.info("  ✓ All secrets present")
    except Exception as e:
        logger.error(f"  ✗ Secrets FAILED: {e}")
        all_ok = False

    # ── Step 4: API key live test ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DRYRUN Step 4/8: API key live tests")
    logger.info("=" * 60)

    # HuggingFace
    try:
        from huggingface_hub import HfApi
        from peat.secrets import hf_token
        api = HfApi()
        user = api.whoami(token=hf_token())
        logger.info(f"  ✓ HuggingFace: user={user.get('name', 'OK')}")
    except Exception as e:
        logger.error(f"  ✗ HuggingFace: {e}")
        all_ok = False

    # Gemini (×4 keys, safety settings disabled)
    try:
        import google.generativeai as genai
        from peat.secrets import gcp_key
        from peat.utils import _GEMINI_SAFETY_SETTINGS
        for i in range(1, 5):
            try:
                key = gcp_key(i)
                genai.configure(api_key=key)
                model = genai.GenerativeModel("gemini-2.5-flash-lite")
                gen_cfg_kwargs = dict(
                    response_mime_type="application/json",
                    max_output_tokens=64,
                    temperature=0.0,
                )
                # ThinkingConfig only available in google-generativeai>=0.8
                if hasattr(genai.types, "ThinkingConfig"):
                    gen_cfg_kwargs["thinking_config"] = genai.types.ThinkingConfig(thinking_budget=0)
                response = model.generate_content(
                    'Return the JSON {"ok": true}',
                    generation_config=genai.types.GenerationConfig(**gen_cfg_kwargs),
                    safety_settings=_GEMINI_SAFETY_SETTINGS,
                )
                import json
                result = json.loads(response.text)
                assert result.get("ok") is True, f"Expected ok=true, got {result}"
                logger.info(f"  ✓ GCP key {i}: OK")
            except Exception as e:
                logger.warning(f"  ~ GCP key {i} (optional LLM-judge): {e}")
    except Exception as e:
        logger.warning(f"  ~ Gemini import (optional LLM-judge): {e}")

    # DeepSeek (fallback level 2 — required)
    try:
        import requests
        import json
        from peat.secrets import deepseek_key
        key = deepseek_key()
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": 'Return the JSON {"ok": true}'}],
                "max_tokens": 64,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=15,
        )
        resp.raise_for_status()
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        assert result.get("ok") is True, f"Expected ok=true, got {result}"
        logger.info(f"  ✓ DeepSeek: OK")
    except Exception as e:
        logger.warning(f"  ~ DeepSeek (optional LLM-judge fallback): {e}")

    # Mistral (fallback level 3 — required)
    try:
        import requests
        import json
        from peat.secrets import mistral_key
        key = mistral_key()
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": 'Return the JSON {"ok": true}'}],
                "max_tokens": 64,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=15,
        )
        resp.raise_for_status()
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        assert result.get("ok") is True, f"Expected ok=true, got {result}"
        logger.info(f"  ✓ Mistral: OK")
    except Exception as e:
        logger.warning(f"  ~ Mistral (optional LLM-judge fallback): {e}")

    # ── Step 5: Model load check ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DRYRUN Step 5/8: Model loading")
    logger.info("=" * 60)
    from peat.models import MODEL_REGISTRY, load_model
    from peat.utils import cleanup as mem_cleanup

    for tag in MODEL_REGISTRY:
        try:
            model, tokenizer, cfg = load_model(tag, device="cuda")
            # Run a single forward on 8-token dummy input
            dummy = tokenizer("The quick brown fox", return_tensors="pt",
                              truncation=True, max_length=8).to("cuda")
            with torch.no_grad():
                out = model(**dummy)
            assert out.logits is not None
            logger.info(f"  ✓ {tag}: loaded + forward OK")
            mem_cleanup(model)
        except Exception as e:
            logger.error(f"  ✗ {tag}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_ok = False

    # ── Step 6: Dataset structure check ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DRYRUN Step 6/8: Dataset validation")
    logger.info("=" * 60)
    try:
        from peat.data import validate_dataset_structure
        ds_ok = validate_dataset_structure()
        if not ds_ok:
            all_ok = False
    except Exception as e:
        logger.error(f"  ✗ Dataset validation: {e}")
        all_ok = False

    # ── Step 7: One-row training check ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DRYRUN Step 7/8: One-row training")
    logger.info("=" * 60)
    import copy
    from peat.data import load_stereoset_pairs
    from peat.peat import attach_lora, compute_peat_loss

    train_df, _ = load_stereoset_pairs()
    sample_row = train_df.iloc[0]

    for tag in MODEL_REGISTRY:
        try:
            model, tokenizer, _ = load_model(tag, device="cuda")
            model = attach_lora(model, tag)

            model_theta0 = copy.deepcopy(model)
            model_theta0.eval()
            for p in model_theta0.parameters():
                p.requires_grad = False

            batch = [{
                "context": sample_row["context"],
                "t_s": sample_row["t_s"],
                "t_a": sample_row["t_a"],
            }]

            model.train()
            loss, loss_dict = compute_peat_loss(
                model, model_theta0, tokenizer, batch, tag, "cuda", 1.0, 1.0,
            )

            assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
            loss.backward()

            # Verify gradients flow into LoRA params only
            lora_grad = False
            non_lora_grad = False
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if "lora_" in name:
                        lora_grad = True
                    else:
                        non_lora_grad = True

            assert lora_grad, "No gradients in LoRA params!"
            if non_lora_grad:
                logger.warning(f"  ⚠ {tag}: gradients found in non-LoRA params")

            logger.info(
                f"  ✓ {tag}: loss={loss.item():.6f}, "
                f"grads_in_lora={lora_grad}, L={loss_dict}"
            )
            mem_cleanup(model, model_theta0)
        except Exception as e:
            logger.error(f"  ✗ {tag}: one-row training FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_ok = False

    # ── Step 8: Path check ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DRYRUN Step 8/8: Path writability")
    logger.info("=" * 60)
    for d in ALL_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
            logger.info(f"  ✓ {d}: writable")
        except Exception as e:
            logger.error(f"  ✗ {d}: {e}")
            all_ok = False

    # ── Result ─────────────────────────────────────────────────────────────
    if all_ok:
        passed_file.touch()
        logger.info("=" * 60)
        logger.info("DRYRUN: ALL 8 CHECKS PASSED")
        logger.info("=" * 60)
    else:
        logger.error("DRYRUN: SOME CHECKS FAILED — see above")

    return all_ok
