# SFT: Qwen3-30B-A3B-Base on OpenR1-Math-220k (first 3001 samples)

Single source of truth for the values: `jobs/configs/sft_openr1_3001.sh` (+ defaults in `jobs/lib/run_30b.sh`).
Exact argument list of every launch: `$SFT_RUN/train_args_<timestamp>.txt`. Full resolved config: slime prints all
arguments at the top of the job log.

## How to reproduce (other account, own node)

1. Set up the environment: `env_guide.txt` (container, afai_env, `SCR`), then `jobs/download_30b.sbatch` +
   `jobs/convert_30b.sbatch` for the base model.
2. `cd ~/afai_project`
   `P=$(sbatch --parsable jobs/prep_openr1_sft.sbatch)`
   `T=$(sbatch --parsable --dependency=afterok:$P jobs/test_sft_ckpt_tiny.sbatch)`   (needs `jobs/verify_router_freeze.sbatch` run once, for the tiny model)
   `sbatch --dependency=afterok:$T jobs/sft_openr1_30b.sbatch`
3. If the job stops with `STATUS = needs_resume ...`, submit `jobs/sft_openr1_30b.sbatch` again.
4. Check `$SCR/data/openr1_math_sft/manifest.json`: dataset revision and `output_sha256` should match the table below.

## Model and software

| | |
|---|---|
| Base model | `Qwen/Qwen3-30B-A3B-Base` (HF), converted to Megatron torch_dist with `jobs/convert_30b.sbatch` |
| Framework | slime v0.4.0 (`0b0c277`), Megatron-LM `4b4acac`, image `slimerl/slime@sha256:626fbff9…` (see SETUP.md) |
| Precision | bf16 params/activations, fp32 router (`--moe-router-dtype fp32`), fp32 grad accumulation/all-reduce (`--accumulate-allreduce-grads-in-fp32`), fp32 attention softmax |
| Router | **trained** (no `--freeze-params-name-list`); aux load-balancing loss coefficient 0 (`--moe-aux-loss-coeff 0`, slime's default for this model) |
| Dropout | 0 (attention and hidden) |

## Data

| | |
|---|---|
| Dataset | `open-r1/OpenR1-Math-220k`, config `default`, split `train` (93,733 rows) |
| Revision | `e4e141ec9dea9f8326f4d347be56105859b2bd68`; source shard sha256 `ccc3a95e4875dc4cff9bec2f0fb1c93e4c9cbb24fb24853fd0b96c416e1a263c` |
| Selection | **first 3001 rows in dataset order** (`data/train-00000-of-00010.parquet`, rows 0..3000), no filtering, no dedup |
| Fields used | `messages` (user problem + assistant solution with `<think>…</think>` reasoning); `uuid` kept for traceability |
| Output file | `$SCR/data/openr1_math_sft/openr1_math_default_train_first3001.jsonl`, sha256 `74ea9d14dc38c013afd4225a88b5f9050d6f728f21909e5f53ed7b63b667b5cf` (first uuid `586fd646-…`, last `e2ae0cf1-…`) |
| Token lengths | 17.80M tokens per epoch (17.50M trained); per sample mean 5932, median 4867, p95 14299, p99 17025, max 20088; 53 samples > 16384 |
| Prompt formatting | Qwen3 chat template of the base model's tokenizer |
| Loss mask | `--loss-mask-type qwen3`: loss on all assistant tokens (incl. `<think>` reasoning and `<|im_end|>`), none on the user turn or the `<|im_start|>assistant\n` header. (slime's default `qwen` mask renders the assistant turn alone, which makes the Qwen3 template **drop the reasoning**; checked in the prep job.) |
| Truncation | none (`--max-tokens-per-gpu` ≥ longest sample) |

## Training

| | |
|---|---|
| Objective | `--loss-type sft_loss` (token-level cross-entropy), `--calculate-per-token-loss` (mean over all trained tokens in the global batch) |
| Epochs / steps | 3 epochs; `num_rollout = 3 * (3001 // 32) = 279` optimizer steps (8928 samples). slime's loader wraps across epochs, so batches can straddle epoch boundaries and the last 75 samples of epoch 3's order are not used |
| Batch | global batch 32 sequences per optimizer step (`--rollout-batch-size 32 --global-batch-size 32`); sequences packed into micro-batches of ≤ 20480 tokens per GPU (`--use-dynamic-batch-size`) |
| Sampling / order | each epoch is a fresh permutation: `random.Random(42 + epoch).shuffle` (`--rollout-shuffle`, `--rollout-seed 42`); Megatron seed 1234 |
| Optimizer | Adam (Megatron distributed optimizer; precision-aware, states on CPU: `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`), β1 0.9, β2 0.95, ε 1e-8, weight decay 0.1, grad clip 1.0 |
| LR schedule | peak 1e-5, linear warmup over 5% of steps (`--lr-warmup-fraction 0.05`), cosine decay to 1e-6 over all 279 steps |
| Parallelism (8 GPUs) | TP 4 + sequence parallel, EP 8, PP 1, CP 1 → DP 2; full activation recompute (uniform, 1 layer) |
| Hardware | 1 node, 8× H200 or 8× B200 (`-p gpu_h200,gpu_b200`), 48 CPUs, 1000 GB RAM |

## Checkpoints (`$SFT_RUN = $SCR/runs/sft_openr1math_3001_qwen3_30b_a3b_base`)

| What | Where | When |
|---|---|---|
| 10 weights-only, bf16, HF safetensors | `hf_ckpts/step_NNNNN/` | optimizer steps 1, 2, 3, 7, 12, 23, 43, 80, 149, 279 (`round(279 ** (i/9))`, collisions filled with the next unused step) |
| 1 full resumable Megatron checkpoint (model, optimizer, scheduler, RNG, data cursor) | `ckpt/iter_NNNNNNN/` + `ckpt/rollout/` | when < 45 min + 2 step-times + last full-save time remain in the job, or at step 279. Older full checkpoints are deleted after a new one succeeds |

Logs: `ckpt_log.jsonl` (every save with timings), `jobs.log` (one line per Slurm job), `STATUS`.
Implementation: `jobs/lib/sft_ckpt.py`; mechanics tested by `jobs/test_sft_ckpt_tiny.sbatch`.

## Choices made without a spec (change in the config file if needed)

Epochs (3), peak LR (1e-5), warmup (5%), min LR (1e-6), global batch (32), β2 (0.95), weight decay (0.1).
