# PEAT — Probability-Equalized Adapter Tuning for Bias Mitigation in Language Models

PEAT trains a parameter-efficient LoRA adapter that equalizes the probability of stereotypical and anti-stereotypical token completions at masked positions, without degrading language-modeling utility. It is evaluated against nine comparative methods (including five recent SOTA bias mitigation approaches) on six language models (three encoders, three causal decoders), under a single fixed compute and precision configuration.

## Hardware Requirements

- **GPU**: Single NVIDIA A100-40GB or A6000-48GB
- **Disk**: ~150 GB free space
- **RAM**: 32+ GB system RAM
- **Estimated runtime**: ~200 GPU-hours total

## OS Requirements

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.12 (required for Flash-Attention 2.8.3 wheel)
- **CUDA**: 12.4

> ⚠️ **Flash-Attention 2.8.3** wheel only works for `cu12/torch2.5/cp312` on **Linux x86_64**. It will not install on Windows or macOS.

## Setup

### Step 1: Configure secrets
```bash
cp .env.template .env
# Edit .env and fill in your API keys
```

### Step 2: Install dependencies
```bash
bash install.sh
```

### Step 3: Run the pipeline
```bash
python3 run_all.py
```

That's it. The pipeline runs end-to-end without human intervention.

## What Runs

### Models

| Tag | HuggingFace ID | Type | Parameters |
|-----|----------------|------|------------|
| `bert-base` | `google-bert/bert-base-uncased` | Encoder MLM | 110M |
| `modernbert-base` | `answerdotai/ModernBERT-base` | Encoder MLM | 150M |
| `neobert` | `chandar-lab/NeoBERT` | Encoder MLM | 250M |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | Causal LM | 1.5B |
| `gemma-3-4b` | `google/gemma-3-4b-it` | Causal LM | 4B |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | Causal LM | 3B |

### Methods

1. **Base** — No mitigation (floor)
2. **CDA** — Counterfactual Data Augmentation (Zmigrod et al., ACL 2019)
3. **Self-Debias** — Inference-time decoding modification (Schick et al., TACL 2021)
4. **Auto-Debias** — JS-divergence debiasing (Guo et al., ACL 2022)
5. **BiasEdit** — Lightweight editor networks (Xu et al., TrustNLP@NAACL 2025)
6. **FairSteer** — Activation steering (Li et al., ACL 2025 Findings)
7. **BiasUnlearn** — Dual-pathway unlearning (Liu et al., EMNLP 2025)
8. **KnowBias** — Bias-neuron enhancement (Pan et al., arXiv 2601.21864, 2026)
9. **LoRA + Vanilla SFT** — Internal ablation (same adapter, standard loss)
10. **PEAT** — Ours

### Datasets

- **StereoSet** (Nadeem et al., 2021) — Training source
- **CrowS-Pairs** (Nangia et al., 2020) — Primary test set (1,508 pairs, 9 bias categories)
- **BBQ** (Parrish et al., 2022) — Extrinsic sanity check (causal LMs only)
- **GLUE** (Wang et al., 2018) — Utility evaluation for encoders (8 tasks)
- **WikiText-103** (Merity et al., 2016) — Perplexity evaluation for decoders

## Output Layout

```
results/
  raw/                         # per-row CSVs flushed every 50 rows
    peat/<model>/<seed>/<config>.csv
    baseline_<name>/<model>/<seed>.csv
  aggregated/
    table1_headline.csv        # method × model headline
    table2_per_category.csv    # PEAT per-category SS
    table3_ablations.csv
    table4_selector.csv        # SHA vs grid vs random
    table5_scaling.csv         # PEAT on Gemma + Llama
  figures/
    fig1_ss_vs_compute.pdf
    fig2_per_category_heatmap.pdf
logs/
  install.log
  dataset_preflight.log
  dryrun.log
  training.log
  baselines.log
  evaluation.log
state/
  run_state.json
  dryrun_passed
```

## Resuming

If the run is interrupted, simply re-run `python3 run_all.py`. The launcher reads `state/run_state.json` and resumes at the first incomplete cell.

## Troubleshooting

- **Flash-Attention import fails** → Check Python is 3.12, CUDA is 12.4, OS is Linux x86_64.
- **Gated model 403** → Request access on HuggingFace for `google/gemma-3-4b-it` and `meta-llama/Llama-3.2-3B-Instruct` and ensure `HF_KEY` belongs to that account.
- **OOM on Llama-3.2-3B** → Reduce per-device batch size; never enable 8-bit/4-bit because the uniform precision policy forbids it.
- **Gemini returns truncated JSON** → Already handled by `thinking_budget=0` and `max_output_tokens=4096`; if it persists, advance round-robin key.

## Citations

```bibtex
@inproceedings{nadeem2021stereoset,
  title={StereoSet: Measuring stereotypical bias in pretrained language models},
  author={Nadeem, Moin and Bethke, Anna and Reddy, Siva},
  booktitle={ACL},
  year={2021}
}

@inproceedings{nangia2020crows,
  title={CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models},
  author={Nangia, Nikita and Vania, Clara and Bhatt, Rasika and Bowman, Samuel R},
  booktitle={EMNLP},
  year={2020}
}

@inproceedings{salazar2020masked,
  title={Masked Language Model Scoring},
  author={Salazar, Julian and Liang, Davis and Nguyen, Toan Q and Kirchhoff, Katrin},
  booktitle={ACL},
  year={2020}
}

@article{liang2022holistic,
  title={Holistic Evaluation of Language Models},
  author={Liang, Percy and others},
  journal={arXiv:2211.09110},
  year={2022}
}

@inproceedings{wang2018glue,
  title={GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and others},
  booktitle={EMNLP},
  year={2018}
}

@article{merity2016pointer,
  title={Pointer Sentinel Mixture Models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  journal={arXiv:1609.07843},
  year={2016}
}

@inproceedings{parrish2022bbq,
  title={BBQ: A Hand-Built Bias Benchmark for Question Answering},
  author={Parrish, Alicia and others},
  booktitle={ACL Findings},
  year={2022}
}

@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@article{huber1964robust,
  title={Robust Estimation of a Location Parameter},
  author={Huber, Peter J},
  journal={Annals of Mathematical Statistics},
  year={1964}
}

@inproceedings{jamieson2016nonstochastic,
  title={Non-stochastic Best Arm Identification and Hyperparameter Optimization},
  author={Jamieson, Kevin and Talwalkar, Ameet},
  booktitle={AISTATS},
  year={2016}
}

@inproceedings{zmigrod2019cda,
  title={Counterfactual Data Augmentation for Mitigating Gender Stereotypes},
  author={Zmigrod, Ran and others},
  booktitle={ACL},
  year={2019}
}

@article{schick2021self,
  title={Self-Diagnosis and Self-Debiasing of Large Language Models},
  author={Schick, Timo and Udupa, Sahana and Sch{\"u}tze, Hinrich},
  journal={TACL},
  year={2021}
}

@inproceedings{guo2022auto,
  title={Auto-Debias: Debiasing Masked Language Models with Automated Biased Prompts},
  author={Guo, Yue and others},
  booktitle={ACL},
  year={2022}
}

@inproceedings{xu2025biasedit,
  title={BiasEdit: Debiasing Stereotyped Language Models via Model Editing},
  author={Xu, Xin and others},
  booktitle={TrustNLP@NAACL},
  year={2025}
}

@inproceedings{li2025fairsteer,
  title={FairSteer: Steering Language Models to Be Fair},
  author={Li, Jingyu and others},
  booktitle={ACL Findings},
  year={2025}
}

@inproceedings{liu2025biasunlearn,
  title={BiasUnlearn: Mitigating Social Bias via Dual-Pathway Unlearning},
  author={Liu, Wei and others},
  booktitle={EMNLP},
  year={2025}
}

@article{pan2026knowbias,
  title={KnowBias: Detecting and Mitigating Biases via Knowledge-Aware Neuron Enhancement},
  author={Pan, Yiwen and others},
  journal={arXiv:2601.21864},
  year={2026}
}

@inproceedings{goldfarb2021intrinsic,
  title={Intrinsic Bias Metrics Do Not Correlate with Application Bias},
  author={Goldfarb-Tarrant, Seraphina and others},
  booktitle={ACL},
  year={2021}
}
```

## License & Contact

This code is provided for research purposes. Please cite our paper if you use PEAT in your work.
