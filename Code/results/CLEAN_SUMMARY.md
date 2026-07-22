# PEAT — Clean Results Summary (paper-ready)

_Deduplicated per-instance CSVs; BBQ excluded (degraded LLM-judge); inconsistent/incomplete cells flagged._


## PEAT (Stereotype Score on CrowS-Pairs, mean ± SD)

| Model | n seeds | SS mean | SD | Gender | Flag |
|---|---|---|---|---|---|
| bert-base | 5 | 58.04 | 0.45 | 49.4 | ok |
| modernbert-base | 5 | 58.18 | 0.32 | 49.9 | ok |
| nomicbert | 5 | 56.96 | 0.31 | 52.8 | ok |
| qwen2.5-1.5b | 5 | 53.42 | 0.27 | 47.9 | ok |
| gemma-3-4b | 3 | 55.84 | 0.19 | 50.9 | ok |
| llama-3.1-8b | 3 | 55.75 | 0.08 | 50.5 | ok |

## Baselines (SS, mean over seeds) — WP-A: must differ from Base

| Method | Model | n | SS mean | SD | Flag |
|---|---|---|---|---|---|
| auto_debias | bert-base | 3 | 58.07 | 0.36 | ok |
| auto_debias | modernbert-base | 3 | 58.05 | 0.17 | ok |
| auto_debias | nomicbert | 3 | 57.21 | 0.22 | ok |
| auto_debias | qwen2.5-1.5b | 3 | 54.44 | 0.05 | ok |
| base | bert-base | 3 | 57.89 | 0.00 | ok |
| base | modernbert-base | 3 | 59.08 | 0.00 | ok |
| base | nomicbert | 1 | 56.70 | 0.00 | ok |
| base | qwen2.5-1.5b | 1 | 54.51 | 0.00 | ok |
| bias_edit | bert-base | 3 | 57.91 | 0.31 | ok |
| bias_edit | modernbert-base | 3 | 58.07 | 0.30 | ok |
| bias_edit | nomicbert | 3 | 56.81 | 0.74 | high SD |
| bias_edit | qwen2.5-1.5b | 3 | 54.07 | 0.28 | ok |
| bias_unlearn | bert-base | 3 | 58.02 | 0.27 | ok |
| bias_unlearn | modernbert-base | 3 | 58.16 | 0.38 | ok |
| bias_unlearn | nomicbert | 3 | 57.27 | 0.17 | ok |
| bias_unlearn | qwen2.5-1.5b | 3 | 33.47 | 0.69 | high SD |
| cda | bert-base | 3 | 58.27 | 0.06 | ok |
| cda | modernbert-base | 3 | 58.24 | 0.22 | ok |
| cda | nomicbert | 3 | 56.98 | 0.36 | ok |
| cda | qwen2.5-1.5b | 3 | 56.85 | 0.06 | ok |
| fair_steer | bert-base | 3 | 56.90 | 0.00 | ok |
| fair_steer | modernbert-base | 3 | 57.82 | 0.00 | ok |
| fair_steer | nomicbert | 3 | 57.16 | 0.00 | ok |
| fair_steer | qwen2.5-1.5b | 3 | 55.50 | 0.00 | ok |
| know_bias | bert-base | 3 | 57.29 | 0.00 | ok |
| know_bias | modernbert-base | 3 | 57.63 | 0.00 | ok |
| know_bias | nomicbert | 3 | 55.17 | 0.00 | ok |
| know_bias | qwen2.5-1.5b | 3 | 57.49 | 0.00 | ok |
| lora_vanilla_sft | bert-base | 5 | 58.69 | 0.29 | ok |
| lora_vanilla_sft | modernbert-base | 5 | 57.45 | 0.33 | ok |
| lora_vanilla_sft | nomicbert | 5 | 56.98 | 0.29 | ok |
| lora_vanilla_sft | qwen2.5-1.5b | 5 | 56.33 | 0.25 | ok |
| self_debias | bert-base | 3 | 54.24 | 0.00 | ok |
| self_debias | modernbert-base | 3 | 51.86 | 0.00 | ok |
| self_debias | nomicbert | 1 | 52.59 | 0.00 | ok |
| self_debias | qwen2.5-1.5b | 1 | 42.77 | 0.00 | ok |

## Loss-term ablation

| Variant | Model | n | SS mean | SD |
|---|---|---|---|---|
| A1_neut_only | bert-base | 1 | 59.88 | 0.00 |
| A1_neut_only | modernbert-base | 1 | 58.29 | 0.00 |
| A1_neut_only | nomicbert | 1 | 57.69 | 0.00 |
| A1_neut_only | qwen2.5-1.5b | 1 | 55.50 | 0.00 |
| A2_neut_pair | bert-base | 1 | 57.43 | 0.00 |
| A2_neut_pair | modernbert-base | 1 | 57.76 | 0.00 |
| A2_neut_pair | nomicbert | 1 | 57.29 | 0.00 |
| A2_neut_pair | qwen2.5-1.5b | 1 | 54.51 | 0.00 |
| A3_neut_kl | bert-base | 1 | 60.34 | 0.00 |
| A3_neut_kl | modernbert-base | 1 | 58.16 | 0.00 |
| A3_neut_kl | nomicbert | 1 | 57.36 | 0.00 |
| A3_neut_kl | qwen2.5-1.5b | 1 | 53.78 | 0.00 |

## Placement factorial

| Variant | Model | n | SS mean | SD |
|---|---|---|---|---|
| fact_mlm_all | bert-base | 1 | 59.08 | 0.00 |
| fact_mlm_all | qwen2.5-1.5b | 1 | 59.88 | 0.00 |
| fact_mlm_first2 | bert-base | 1 | 57.16 | 0.00 |
| fact_mlm_first2 | qwen2.5-1.5b | 1 | 55.64 | 0.00 |
| fact_peat_all | bert-base | 1 | 57.49 | 0.00 |
| fact_peat_all | qwen2.5-1.5b | 1 | 53.58 | 0.00 |
| fact_peat_first2 | bert-base | 1 | 58.42 | 0.00 |
| fact_peat_first2 | qwen2.5-1.5b | 1 | 54.24 | 0.00 |

## PEAT-CB (coverage-balanced) — target categories vs overall

| Model | n | SS mean | SD | disability | sexual-orientation |
|---|---|---|---|---|---|
| bert-base | 3 | 57.82 | 0.09 | 63.3 | 62.7 |
| qwen2.5-1.5b | 3 | 53.07 | 0.03 | 70.0 | 76.2 |

## Notes

- BBQ excluded: Gemini API quota exhausted during the run; the LLM-judge produced degraded/partial scores. Extrinsic fairness is reported via HONEST, Bias-in-Bios (TPR gap), and StereoSet-heldout.
- crows-choice generation metric disabled (auxiliary, not a main result).
- All per-instance CSVs deduplicated by idx (keep-last); 0 contaminated rows.
