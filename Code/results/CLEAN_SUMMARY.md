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
| llama-3.1-8b | 3 | 55.79 | 0.11 | 50.8 | ok |

## Baselines
_(baseline stage not yet complete)_


## Notes

- BBQ excluded: Gemini API quota exhausted during the run; the LLM-judge produced degraded/partial scores. Extrinsic fairness is reported via HONEST, Bias-in-Bios (TPR gap), and StereoSet-heldout.
- crows-choice generation metric disabled (auxiliary, not a main result).
- All per-instance CSVs deduplicated by idx (keep-last); 0 contaminated rows.
