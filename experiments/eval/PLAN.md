# Evaluation plan: SFT vs RL effects inside Qwen3-30B-A3B

Project: *Disentangle SFT and RL Effects in Mixture of Experts Models* (`5880ProjectProposal.pdf`). Main question:
how SFT and GRPO change the MoE model's parameters, experts and routing, and how those changes relate to
capability gains and losses. Method baseline: *Scalpel vs. Hammer* (Rajani et al. 2025, `scalpel_happer.pdf`).

Runs to analyse (registry: `run_info` in `jobs/configs/eval.sh`; each run is measured against the model its stage
started from, and `REF_OVERRIDE=$BASE_HF` gives the cumulative change since the base model):

| Run | Reference | Status |
|---|---|---|
| `sft`: SFT on OpenR1 rows `iloc[1::3]` | base | done (job 29366121) |
| `grpo`: GRPO on rows `iloc[0::3]` | base | colleague's run |
| SFT → GRPO, GRPO → SFT (second stages) | final checkpoint of stage 1 | to add when run |
| router-frozen ablations | same as their unfrozen counterpart | to add when run |

Rows `iloc[2::3]` are never trained on and provide the held-out probe text.

Code: `jobs/lib/eval/`. Outputs: `$SCR/eval/`; small CSV/JSON summaries are copied to `experiments/eval/results/`.

## 1. Parameter change: where do SFT and RL write? (`eval_param_diff.sbatch`, CPU)

For each saved checkpoint and each weight tensor: relative change ‖W − W_ref‖_F / ‖W_ref‖_F (the paper's normalised
Frobenius norm) and RMS change per entry, in fp32. Per (layer, module): attention q/k/v/o, q/k norms, **router**,
experts gate/up/down pooled over the 128 experts. A **per-expert** table is also written (48 × 128 experts).

Answers: magnitude SFT vs RL; which layers and modules change most (do q/k dominate as in the paper? mid-layer
experts for SFT?); how the change grows over training (10 log-spaced checkpoints: steps 1, 2, 5, … 976).

## 2. Routing change: do experts get chosen differently, and why? (`eval_probe.sbatch`, 1 H200)

Teacher forcing on fixed probe text, reference model and checkpoint on the same tokens. Probes:

| Probe | Text | Purpose |
|---|---|---|
| `openr1_heldout` | 300 held-out OpenR1 rows (`iloc[2::3]`), R1 solution incl. `<think>` trace (≤ 4096 tokens) | in-distribution math |
| `math500` | 500 MATH-500 problems + human reference solutions | the paper's KL probe, comparability |
| `general` | 300 MMLU questions (the question text itself) | non-math text |

A hook on every layer's router recomputes its top-8 choice (fp32) per token:
R = reference router on reference inputs, C = checkpoint router on checkpoint inputs,
F = **reference** router on **checkpoint** inputs (counterfactual). Per layer and probe:

- `overlap_total` = |C ∩ R| / 8: how much routing changed
- `overlap_inputs` = |F ∩ R| / 8: change caused by drifted hidden states alone
- `overlap_router` = |C ∩ F| / 8: change caused by the router weights alone (exactly 1 with a frozen router)
- expert load per expert (reference and checkpoint) and its Jensen-Shannon divergence

This split is what the router-freezing ablation needs: a frozen router can still route differently through its inputs,
and the decomposition shows which effect dominates in SFT vs RL.

Also in the same pass: **KL(ref ‖ ckpt)** per token (paper Sec. 4.1) and the reverse direction, per probe → output-drift
curves over training, math vs general text.

## 3. Expert specialization: are updates localized? (`eval_specialization.sbatch`, CPU)

Math affinity of each expert in the reference model = log(share of math-probe tokens) − log(share of general-probe
tokens). Per checkpoint and layer: rank correlation between affinity and the expert's update size, share of update
energy in the 10% most math-affine experts (0.10 = no preference), Gini of update energy over experts
(spread vs localized), and max/median update. This answers the proposal's "distributed across experts vs
localized to math experts", separately for SFT and RL.

## 4. Benchmarks: how the internal changes show up in capability (`eval_bench.sbatch`, 1 H200 per model)

| Benchmark | Role | Questions | Samples | Grading |
|---|---|---|---|---|
| AIME 2026, HMMT Feb 2026 | math, after the Qwen3 release (no contamination) | 30, ~30 | 16 | slime `math` grader (= GRPO reward) |
| MATH-500, GSM8K | math, comparability with the paper | 500, 1,319 | 4, 1 | same |
| AIME 2024 | math (possibly contaminated), optional | 30 | 16 | same |
| GPQA-Diamond | science reasoning + knowledge | 198 | 4 | option letter in `\boxed{}` |
| MMLU, MMLU-Pro | general knowledge (forgetting) | 14,042, 12,032 | 1 | same |

Zero-shot, base chat template, identical prompt tokens for all models; math prompt = the GRPO / R1 instruction.
Temperature 0.6, top-p 0.95, ≤ 30k tokens (16k for GSM8K / MMLU / MMLU-Pro). Reported: accuracy (avg@n) with 95% CI,
pass@1/4/n, truncation and no-answer rates, response length, and per-subject accuracy (which knowledge is lost).
Models: base, final checkpoint of each run (any intermediate checkpoint also works).

Not included: USAMO 2026, which is proof-based and needs human or LLM-judge grading. IFEval needs checkers that are
not installed and can be added. AIME 2026 / HMMT Feb 2026 come from MathArena's Hub copies; the prep job reports it
if those repos are named differently.

## Checkpoints

The proposal mentions ~20 checkpoints. The runs save 10 log-spaced ones, which are dense early, where SFT changes fastest
(paper Fig. 4). Every HF checkpoint is 57 GB, so 20 per run would be ~1.1 TB per run across ≥ 4 runs. Kept at 10. All
runs use the same step schedule, so their curves align. Raising it means `SFT_N_LOG_CKPTS` in a run's config.

## Validation before the real runs

| Test | Checks |
|---|---|
| `jobs/lib/eval/test_scoring.py` | answer extraction, pass@k, CI. **Passed** (agent node) |
| `jobs/test_eval_tiny.sbatch` (gpu_devel 15 min) | tiny MoE from `verify_router_freeze`, known answers: identical copy → parameter change 0, KL 0, all routing overlaps exactly 1; frozen-router SFT → router change exactly 0 and `overlap_router` exactly 1 in every layer; control → router changed, KL > 0; specialization script runs |
| `jobs/test_eval_30b.sbatch` (gpu_devel 30 min) | real 30B: SGLang serves the SFT model, 3 questions per benchmark (≤ 2048 tokens); probe analysis with base + SFT on one GPU (memory, router hooks on 48 layers) |

## Order and cost

```bash
P=$(sbatch --parsable jobs/prep_eval_bench.sbatch)                       # CPU ~15 min
sbatch --dependency=afterok:$P jobs/test_eval_tiny.sbatch                 # gpu_devel 15 min
sbatch --dependency=afterok:$P jobs/test_eval_30b.sbatch                  # gpu_devel 30 min
# after both pass, for the SFT run:
sbatch jobs/eval_param_diff.sbatch sft                                    # CPU, <= 8 h
sbatch jobs/eval_probe.sbatch sft                                         # 1 H200, est. 1-3 h (limit 24 h)
sbatch jobs/eval_specialization.sbatch sft                                # CPU, minutes (after the two above)
sbatch jobs/eval_bench.sbatch base;  sbatch jobs/eval_bench.sbatch sft_final   # 1 H200 each, <= 24 h
```

Every GPU job uses 1 GPU. The total for base + SFT is ≤ ~75 GPU-hours (likely far less), small against the ~1,500 GPU-hour
project budget. The same commands with `grpo` / `grpo_final` analyse the GRPO run.
