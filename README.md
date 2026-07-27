# Test-Time Scaling via Error Localization — reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/test-time-scaling-via-error-localization/blob/main/notebooks/ttel_reproduction.py)

We tested the paper’s causal claim that feedback-conditioned token probabilities can locate a failed suffix, allowing a model to retain its prefix and spend test-time compute more efficiently. **Assessment: partially reproduced.** On 24 deterministic stdin-only LiveCodeBench V6 tasks × four seeds, Qwen3-4B-Thinking-2507 with filtered TTEL reached **32.3% pass@4 at 5,460 mean generated tokens**, versus **4.2% at 5,976 tokens** for independent sampling. Removing the null-feedback filter increased candidate spikes **9.6×** and moved branches from 15.2% to 44.6% of the trace. Its downstream penalty appeared with executable feedback (−1.0 point, +2.5% tokens) but not with generic feedback.

The paper’s Qwen3-8B result at 64 attempts was 71.0% for TTEL, 64.6% for independent sampling, and 56.7% for refinement; TTEL used 360k mean generated tokens versus 735k for independent sampling. Our substitution used the available 4B model, four attempts, a 1,536-token cap, and an executable 24/131-task stratified subset, so it tests the mechanism rather than reproducing the full benchmark estimate.

- [Detailed tutorial-style report](reports/ttel-reproduction/report.md)
- [Self-contained marimo notebook](notebooks/ttel_reproduction.py)
- [Machine-readable measurements](results/summary.json)

All formal runs used the configured Kubernetes backend on NVIDIA RTX PRO 6000 Blackwell GPUs, with 16 GPUs concurrently at peak and four GPUs per experiment. The fresh evidence window lasted 2.31 hours.

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Publication surface | Not run as an experiment (publication surface) | Report, figures, notebook, and reusable implementation | — |
| [`orx/expanded-24-task-independent`](https://github.com/alphaXiv/test-time-scaling-via-error-localization/tree/orx/expanded-24-task-independent) | Matched independent sampling baseline | `bash run.sh` | 4.2% pass@4; 5,976 tokens | Kubernetes, 4× RTX PRO 6000 Blackwell, 43.7 min |
| [`orx/expanded-24-task-refinement`](https://github.com/alphaXiv/test-time-scaling-via-error-localization/tree/orx/expanded-24-task-refinement) | Whole-trajectory refinement baseline | `bash run.sh` | 29.2% pass@4; 5,610 tokens; TTEL +3.1 points | Kubernetes, 4× RTX PRO 6000 Blackwell, 41.4 min |
| [`orx/expanded-24-task-filtered-ttel`](https://github.com/alphaXiv/test-time-scaling-via-error-localization/tree/orx/expanded-24-task-filtered-ttel) | Reconstructed filtered TTEL | `bash run.sh` | 32.3% pass@4; 5,460 tokens; aligned | Kubernetes, 4× RTX PRO 6000 Blackwell, 40.5 min |
| [`orx/expanded-24-task-no-null-ttel`](https://github.com/alphaXiv/test-time-scaling-via-error-localization/tree/orx/expanded-24-task-no-null-ttel) | Remove null-feedback filter | `bash run.sh` | 9.6× spikes; downstream degradation not observed | Kubernetes, 4× RTX PRO 6000 Blackwell, 41.0 min |
| [`orx/expanded-24-task-environment-ttel`](https://github.com/alphaXiv/test-time-scaling-via-error-localization/tree/orx/expanded-24-task-environment-ttel) | Rich public-test feedback | `bash run.sh` | 33.3% pass@4; 5,274 tokens; directional alignment | Kubernetes, 4× RTX PRO 6000 Blackwell, 38.8 min |
| [`orx/expanded-24-task-environment-no-null`](https://github.com/alphaXiv/test-time-scaling-via-error-localization/tree/orx/expanded-24-task-environment-no-null) | Rich-feedback null-filter factorial | `bash run.sh` | 32.3% pass@4; 5,407 tokens; null filter helped | Kubernetes, 4× RTX PRO 6000 Blackwell, 41.4 min |
| [`orx/expanded-24-task-syntax-aware-ttel`](https://github.com/alphaXiv/test-time-scaling-via-error-localization/tree/orx/expanded-24-task-syntax-aware-ttel) | Syntax-boundary sensitivity | `bash run.sh` | 26.0% pass@4; raw-token branching was stronger | Kubernetes, 4× RTX PRO 6000 Blackwell, 40.2 min |

Early startup branches exposed a Kubernetes command-expansion error and an unnecessary NCCL barrier; they produced no scientific measurements. The successful branches use the corrected committed manifest and filesystem-sharded four-seed runner.

## Run locally

The formal evidence comes from the linked Kubernetes branches and terminal logs. To inspect or adapt the implementation:

```bash
python -m pip install -r requirements.txt
torchrun --standalone --nproc_per_node=4 run_reproduction.py
```

The script prints one terminal `ORX_RESULT_JSON` record so the evidence remains auditable through `orx logs`. Candidate programs are executed in isolated subprocesses with time, memory, and file-size limits.
