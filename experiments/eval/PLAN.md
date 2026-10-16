# Evaluation of the post-trained Qwen3-30B-A3B models

Follows the analysis of *Scalpel vs. Hammer: GRPO Amplifies Existing Capabilities, SFT Replaces Them*
(Rajani et al., ICML 2025, `scalpel_happer.pdf`), applied to our runs: Qwen3-30B-A3B-Base, SFT on OpenR1-Math rows
`iloc[1::3]` (`experiments/sft_openr1math_idx3k1/`) and GRPO on rows `iloc[0::3]` (`experiments/grpo_openr1math_idx3k0/`).
Everything below runs now for **base vs. SFT** and is re-run unchanged for GRPO when that run finishes.

Code: `jobs/lib/eval/`. Paths: `jobs/configs/eval.sh`. Outputs: `$SCR/eval/`; small summaries are copied to
`experiments/eval/results/`.

## A. Benchmarks (paper Sec. 3.3, Fig. 3): does training help in-domain and hurt out-of-domain?

| Benchmark | Domain | Questions | Samples/question | Max response tokens | Grading |
|---|---|---|---|---|---|
| MATH-500 | math, in-domain | 500 | 4 | 30,000 | slime `math` grader (last `\boxed{}` vs answer; MATH normalisation / sympy) = the GRPO reward |
| AIME 2024 | hard math | 30 | 16 | 30,000 | same |
| GSM8K (test) | easy math | 1,319 | 1 | 16,384 | same |
| GPQA-Diamond | science MC, reasoning + knowledge | 198 | 4 | 30,000 | option letter in `\boxed{}` (fallback "answer is (X)") |
| MMLU (test) | knowledge MC | 14,042 | 1 | 16,384 | same |
| MMLU-Pro (test) | harder knowledge MC, 10 options | 12,032 | 1 | 16,384 | same |

- Models: base (`Qwen3-30B-A3B-Base`), SFT final (`step_00976`), later GRPO final. The script also takes any
  intermediate checkpoint.
- Prompt (identical tokens for every model, base chat template, zero-shot): math = *"Please reason step by step, and
  put your final answer within \boxed{}."* + problem, which is the GRPO prompt and the instruction R1 used for the SFT data. MC = the
  same instruction asking for the option letter, then question and `A) ...` options. GPQA options are shuffled once with a
  fixed per-question seed.
- Decoding: temperature 0.6, top-p 0.95, no top-k (the DeepSeek-R1 / Qwen3 evaluation setting; the paper reused its
  training temperature 0.7). Stop at `<|im_end|>` or `<|endoftext|>`. SGLang, bf16, 1 H200 per model.
- Reported per benchmark: accuracy = mean over all samples (avg@n), 95% CI over questions, pass@1/4/n
  (whether a correct answer is reachable at all, which connects to "GRPO amplifies, SFT replaces"), truncation rate, no-answer rate,
  mean response length, and accuracy per subject / category (MMLU subjects, MMLU-Pro categories, GPQA domains,
  MATH subjects). The categories show *which* knowledge degrades.
- Differences from the paper: IFEval is not included (its instruction checkers are not installed; can be added). The base
  model is evaluated zero-shot with the chat template like the trained models, so its scores measure the same
  prompt format rather than its best few-shot score. A low base score on format alone shows up as a high
  no-answer rate.

## B. Output-distribution drift (paper Sec. 4.1, Fig. 4): KL(base ‖ checkpoint) over training

- Teacher forcing on held-out text: the 500 MATH-500 problems with their reference solutions, formatted as one chat turn
  (same prompt as A, then solution + `<|im_end|>`). Per solution token, the full-vocabulary
  KL(p_base ‖ p_ckpt) (the paper's direction) and the reverse direction, in fp32, averaged token-weighted and per question.
- For each of the 10 saved SFT checkpoints (steps 1, 2, 5, 10, 21, 46, 98, 211, 454, 976) → a KL-vs-step curve. The GRPO
  run saves at the same steps, so the two curves can be compared directly.
- Base model and one checkpoint on one H200 (2 × 57 GB bf16).

## C. Parameter-level change (paper Sec. 4.2, Figs. 5–6), extended to MoE

- For every weight tensor and checkpoint: relative change ‖W_ckpt − W_base‖_F / ‖W_base‖_F (the paper's normalised
  Frobenius norm) and RMS change ‖ΔW‖_F / √numel. Computed in fp32 and streamed from safetensors on CPU.
- Aggregated per (layer, module): attention q/k/v/o, q_norm/k_norm, **router** (`mlp.gate`), experts gate/up/down
  pooled over the 128 experts, plus a per-expert table (which experts moved), layer norms, embeddings, LM head.
- Questions: do q/k change most, as in the paper? Are mid-layer expert MLPs changed more by SFT than by GRPO? How much does the
  router move? That last one is specific to MoE and relates to our router-freezing setup.

## Validation before the real runs

| Test | What it checks |
|---|---|
| `jobs/lib/eval/test_scoring.py` (runs anywhere) | answer extraction (`\boxed{}` nesting, `\text{(B)}`, out-of-range letters), pass@k, CI. **Passed on the login/agent node** |
| `jobs/test_eval_tiny.sbatch` (gpu_devel, 15 min) | known answers on the tiny MoE from `verify_router_freeze`: bit-identical round-trip copy → parameter change 0 and KL 0; frozen-router SFT → router change exactly 0, attention changed; control SFT → router changed, KL > 0 |
| `jobs/test_eval_30b.sbatch` (gpu_devel, 30 min) | real 30B: SGLang serves the final SFT checkpoint, 3 questions of every benchmark (≤ 2048 tokens), prompts / stop tokens / grading visible in the log; KL with base + SFT on one GPU fits in memory |

## Order and cost (all 1-GPU jobs)

```bash
P=$(sbatch --parsable jobs/prep_eval_bench.sbatch)                        # CPU, ~10 min
sbatch --dependency=afterok:$P jobs/test_eval_tiny.sbatch                  # gpu_devel 15 min
sbatch --dependency=afterok:$P jobs/test_eval_30b.sbatch                   # gpu_devel 30 min
# after both tests pass:
sbatch jobs/eval_bench.sbatch base                                         # 1 H200, <= 24 h (resubmit to resume)
sbatch jobs/eval_bench.sbatch sft_final                                    # 1 H200, <= 24 h
sbatch jobs/eval_kl.sbatch sft                                             # 1 H200, <= 12 h
sbatch jobs/eval_param_diff.sbatch sft                                     # CPU, <= 8 h
```

Worst case is about 60 GPU-hours. The 30B test measures generation speed, so the time limits can be tightened after it runs.
Disk: < 1 GB per model for the samples.
