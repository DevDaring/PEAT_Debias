# Model Loading — How Every Model Is Loaded in PEAT

This document explains exactly how all six models are loaded in the PEAT pipeline, including the specific workarounds required for each model. Any future developer or coding tool should follow these patterns exactly.

---

## 1. Overview

All models are loaded via `Code/peat/models.py`. The public API is:

```python
from peat.models import load_model

model, tokenizer, config_dict = load_model(tag, device="cuda")
```

`tag` is one of the six short identifiers: `bert-base`, `modernbert-base`, `nomicbert`, `qwen2.5-1.5b`, `gemma-3-4b`, `llama-3.1-8b`.

---

## 2. Precision Policy (applies to all models)

- GPU with compute capability ≥ 8.0 (e.g. A100): **bfloat16**
- Older GPU: **float16**
- No 4-bit or 8-bit quantisation — all baselines must use the same dtype for fair comparison.

```python
# Code/peat/utils.py
def get_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16
```

---

## 3. Model Registry

Defined in `Code/peat/models.py` as `ModelSpec` dataclasses:

| Tag | HuggingFace ID | Type | Attention | Special |
|---|---|---|---|---|
| `bert-base` | `google-bert/bert-base-uncased` | encoder | `eager` | — |
| `modernbert-base` | `answerdotai/ModernBERT-base` | encoder | `sdpa` | — |
| `nomicbert` | `nomic-ai/nomic-bert-2048` | encoder | `eager` | `trust_remote_code=True` |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | causal | `sdpa` | — |
| `gemma-3-4b` | `google/gemma-3-4b-it` | causal | `sdpa` | gated (needs HF token) |
| `llama-3.1-8b` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | causal | `sdpa` | gated (needs HF token) |

Flash-Attention 2 (`flash_attention_2`) is NOT used. All causal models use `sdpa` (PyTorch scaled dot-product attention), which activates the Flash kernel automatically on CUDA without the incompatibilities of the `flash_attn` package.

---

## 4. Encoder Loading (`bert-base`, `modernbert-base`, `nomicbert`)

```python
model = AutoModelForMaskedLM.from_pretrained(
    hf_id,
    torch_dtype=dtype,          # bfloat16 on A100
    attn_implementation=attn,   # "sdpa" for modernbert, omitted for bert/nomic
    trust_remote_code=True,     # nomicbert only
    low_cpu_mem_usage=False,    # materialise real tensors, avoid meta-tensor crash on .to()
)
model.tie_weights()             # resolves tied meta-tensors in custom loaders
model = model.to(dtype=dtype, device="cuda")
model.eval()
```

**Why `low_cpu_mem_usage=False`?**  
With `True` (the default), weights are created as "meta tensors" (no memory) and materialised lazily. Custom loaders (NomicBERT, ModernBERT) can leave tied weights (e.g. `decoder.weight`) as meta tensors after materialisation. Calling `.to(device)` on a meta tensor crashes. Setting `False` forces all weights to be real CPU tensors before the `.to()` call. `tie_weights()` is called after loading as a safety net.

**NomicBERT specific notes:**
- `torch_dtype` kwarg is silently ignored by NomicBERT's custom loader — dtype must be applied via `.to(dtype=dtype)` after loading.
- The model class is `NomicBertForPreTraining` (an MLM pretraining wrapper). Its `.forward()` does **not** accept `output_hidden_states=True`. Its output has **no** `last_hidden_state` attribute — only `logits` of vocab-size shape `(batch, seq, 30528)`.
- To get hidden states (dim 768), access the inner encoder directly:
  ```python
  _enc = getattr(model, "bert", None)  # NomicBertForPreTraining wraps model.bert
  out = _enc(input_ids=..., attention_mask=...)  # returns BaseModelOutput
  hidden = out.last_hidden_state  # shape: (batch, seq, 768) ✓
  ```

---

## 5. Causal LM Loading (`qwen2.5-1.5b`, `gemma-3-4b`, `llama-3.1-8b`)

```python
model = AutoModelForCausalLM.from_pretrained(
    hf_id,
    torch_dtype=dtype,              # bfloat16 on A100
    attn_implementation="sdpa",     # all causal models
    token=hf_token,                 # gated models only
    device_map="cuda:0",            # load directly to GPU (no CPU intermediate)
)
model.eval()
```

**Why `device_map="cuda:0"` instead of `.to(device)`?**  
Gemma-3-4b has a multimodal architecture with a vision tower. Loading to CPU first and then calling `.to("cuda")` leaves the vision tower's meta tensors unresolved, causing a crash. `device_map="cuda:0"` loads all shards directly to the GPU via `accelerate`, bypassing this entirely.

**Gated models (Gemma, Llama):**  
Require a HuggingFace token. The token is read from `HF_Classic_Token` in `.env` via `peat/secrets.py`. You must have accepted the model's terms on huggingface.co before the token will grant access.

**Gemma config nesting:**  
`Gemma3Config` nests per-layer settings (e.g. `num_hidden_layers`, `hidden_size`) under `model.config.text_config`, not directly under `model.config`. Always use:
```python
_cfg = model.config
_text_cfg = getattr(_cfg, "text_config", _cfg)
num_layers = getattr(_cfg, "num_hidden_layers", getattr(_text_cfg, "num_hidden_layers", 0))
```

---

## 6. Local Cache Strategy

Both `load_encoder` and `load_causal` try the local HuggingFace cache first:

```python
try:
    model = AutoModel.from_pretrained(hf_id, local_files_only=True, ...)
except OSError:
    model = AutoModel.from_pretrained(hf_id, ...)  # download
```

This avoids unnecessary network calls on repeated runs and works correctly in air-gapped environments after the first download.

---

## 7. Baseline Hidden-State Extraction for NomicBERT

Several baselines (`bias_edit`, `fair_steer`, `know_bias`) need hidden states from encoder models. They pass `output_hidden_states=True` to the model. NomicBERT rejects this with `TypeError`. The fix pattern used in all three files:

```python
try:
    out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-1][:, 0, :]   # CLS token, last layer
except TypeError:
    # NomicBERT: access inner encoder directly
    _enc = getattr(model, "bert", getattr(model, "encoder", None))
    _kw = {k: v for k, v in inputs.items()
           if k in ("input_ids", "attention_mask", "token_type_ids")}
    if _enc is not None:
        out = _enc(**_kw)
    else:
        out = model(**_kw)
    # out is BaseModelOutput with last_hidden_state of shape (batch, seq, 768)
    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state[:, 0, :]
    else:
        hidden = out[0][:, 0, :]
```

This pattern is also used in `peat/eval.py` at line ~626 (the canonical reference implementation).

---

## 8. LoRA Target Modules per Model

```python
# bert-base
["query", "key", "value", "dense", "intermediate.dense", "output.dense"]

# modernbert-base
["Wqkv", "Wo", "Wi"]

# nomicbert
["Wqkv", "out_proj", "fc11", "fc12", "fc2"]

# qwen2.5-1.5b, gemma-3-4b, llama-3.1-8b (all causal LMs)
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

LoRA is applied to the last 2 transformer layers only (indices `[num_layers-2, num_layers-1]`).

---

## 9. Quick Reference — Common Errors and Solutions

| Error | Cause | Fix |
|---|---|---|
| `TypeError: forward() got unexpected keyword argument 'output_hidden_states'` | NomicBERT's `NomicBertForPreTraining` doesn't accept this kwarg | Use `model.bert` inner encoder (see §7) |
| `mat1 and mat2 shapes cannot be multiplied (1×30528 and 768×64)` | Using `out.logits` (vocab size) instead of `last_hidden_state` (768) from NomicBERT | Same fix as above — access `model.bert` |
| `RuntimeError: Expected all tensors to be on the same device` | Causal model loaded to CPU, then GPU operations attempted | Use `device_map="cuda:0"` in loader |
| `RuntimeError: Cannot copy out of meta tensor` | `low_cpu_mem_usage=True` with custom loader leaves meta tensors | Set `low_cpu_mem_usage=False` and call `tie_weights()` |
| `ValueError: Access to model is restricted` | Gated model, missing or invalid HF token | Set `HF_Classic_Token` in `.env` and accept terms on huggingface.co |
| `KeyError: Unknown model tag` | Tag not in registry | Use one of the 6 valid tags listed in §3 |
