# Test-Time Scaling via Error Localization — reproduction

This repository contains a fresh, bounded reproduction of the central claims in
arXiv:2607.21453. Formal results are produced on Kubernetes experiment branches;
`main` is the publication surface. The public report, figures, notebook, and exact
experiment provenance will be added after the fresh runs finish.

## Reproduction protocol

- Benchmark: deterministic, difficulty-stratified 12-task subset of the public
  LiveCodeBench V6 (`v6`) code-generation split.
- Model: `Qwen/Qwen3-4B-Thinking-2507`.
- Conditions: independent sampling, whole-trajectory refinement, TTEL with generic
  feedback, TTEL without the null baseline, TTEL with execution feedback, and a
  syntax-aware branch-point sensitivity check.
- Budget: four attempts per task, 1,536 generated tokens per attempt, four seeds.
- Compute: Kubernetes, four NVIDIA RTX PRO 6000 Blackwell GPUs per formal run.

The implementation prints a single `ORX_RESULT_JSON=` record at the end of every
run so that the complete evidence is preserved in the OpenResearch terminal log.
