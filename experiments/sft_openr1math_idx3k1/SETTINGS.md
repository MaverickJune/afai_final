# SFT: Qwen3-30B-A3B-Base on OpenR1-Math-220k, every third row (0-based 1, 4, 7, ...), 1 epoch

Single source of truth for the values: `jobs/configs/sft_openr1_idx3k1.sh` (+ defaults in `jobs/lib/run_30b.sh`).

Where the settings are recorded:

| File | Contents |
|---|---|
| this file | the training-relevant settings in words, incl. the important defaults |
| `resolved_args.json` (this folder) | **every** resolved argument incl. all slime / Megatron / TransformerEngine defaults, from the CPU dry run of the exact launch command (`num_rollout` is derived at startup: 976) |
| `$SFT_RUN/train_args_<timestamp>.txt` | the explicit command-line arguments of each launch |
| `$SFT_RUN/resolved_args_<jobid>_<timestamp>.json` | every resolved argument of each real launch (copy in `logs/resolved_args_*`) |
| job log `logs/sft_openr1_30b_<jobid>.out` | slime/Megatron also print all arguments at startup |

## How to reproduce (other account, own node)

1. Set up the environment: `env_guide.txt` (container, afai_env, `SCR`), then `jobs/download_30b.sbatch` +
   `jobs/convert_30b.sbatch` for the base model.
2. `cd ~/afai_project`
   `P=$(sbatch --parsable jobs/prep_openr1_sft.sbatch)`   (CPU only, ~20 min; any CPU partition works, e.g. `-p devel -c 4 --mem=32G`)
   `T=$(sbatch --parsable --dependency=afterok:$P jobs/test_sft_ckpt_tiny.sbatch)`   (needs `jobs/verify_router_freeze.sbatch` run once, for the tiny model; also dry-runs the real 30B arguments)
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
| Revision | `e4e141ec9dea9f8326f4d347be56105859b2bd68` (sha256 of all 10 source shards in manifest.json) |
| Selection | **0-based rows 1, 4, 7, ..., 93730 of the train split in dataset order** = pandas `iloc[1::3]` over `data/train-0000{0..9}-of-00010.parquet` concatenated in order → **31,244 rows**. Row 0 is not used. No filtering, no dedup |
| Fields used | `messages` (user problem + assistant solution with `<think>…</think>` reasoning); `uuid` and original `row_index` kept for traceability |
| Output file | `$SCR/data/openr1_math_sft/openr1_math_default_train_idx3k1.jsonl`, sha256 `4da3b63698b9b258aaaf9b7e37373b0edcb84daccd87d53a3cf699ebb7184c59` (first row index 1 / uuid `1d57f4b3-…`, last row index 93730 / uuid `2e8d3afa-…`) |
| Token lengths | 186.46M tokens (183.30M trained); per sample mean 5968, median 4903, p95 14104, p99 16877, max 22169; 514 samples > 16384, 4 > 20480, none > 32768 |
| Prompt formatting | Qwen3 chat template of the base model's tokenizer |
| Loss mask | `--loss-mask-type qwen3`: loss on all assistant tokens (incl. `<think>` reasoning and `<|im_end|>`), none on the user turn or the `<|im_start|>assistant\n` header. (slime's default `qwen` mask renders the assistant turn alone, which makes the Qwen3 template **drop the reasoning**; checked in the prep job.) |
| Truncation | none (`--max-tokens-per-gpu` ≥ longest sample) |

## Training

| | |
|---|---|
| Objective | `--loss-type sft_loss` (token-level cross-entropy), `--calculate-per-token-loss` (mean over all trained tokens in the global batch) |
| Epochs / steps | 1 epoch; `num_rollout = 31244 // 32 = 976` optimizer steps (31,232 samples); the last 12 samples of the shuffled order are not used |
| Batch | global batch 32 sequences per optimizer step (`--rollout-batch-size 32 --global-batch-size 32`); sequences packed into micro-batches of ≤ 24576 tokens per GPU (`--use-dynamic-batch-size`) |
| Token sampling / decoding | **not applicable**: SFT generates no text. The model is trained with teacher forcing on the dataset's assistant tokens; no SGLang engine is started (`--debug-train-only`). slime's decoding settings are therefore unused and left at their defaults (`--rollout-temperature 1.0`, `--rollout-top-p 1.0`, `--rollout-top-k -1`, eval settings unset; they also appear in `resolved_args.json`). Decoding only matters for GRPO rollouts and for evaluating the checkpoints, which have to be specified separately. |
| Data sampling / order | one random permutation of the 31,244 rows: `random.Random(42 + 0).shuffle` (`--rollout-shuffle`, `--rollout-seed 42`), consumed in order, 32 per step; Megatron seed 1234 |
| Optimizer | Adam (Megatron distributed optimizer; precision-aware, states on CPU: `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`), β1 0.9, β2 0.95, ε 1e-8, weight decay 0.1, grad clip 1.0 |
| LR schedule | peak 1e-5, linear warmup over 5% of steps (`--lr-warmup-fraction 0.05`), cosine decay to 1e-6 over all 976 steps (warmup ≈ 49 steps) |
| Parallelism (8 GPUs) | TP 4 + sequence parallel, EP 8, PP 1, CP 1 → DP 2; full activation recompute (uniform, 1 layer) |
| Hardware | 1 node, 8× H200 or 8× B200 (`-p gpu_h200,gpu_b200`), 48 CPUs, 1000 GB RAM |

## Checkpoints (`$SFT_RUN = $SCR/runs/sft_openr1math_idx3k1_qwen3_30b_a3b_base`)

| What | Where | When |
|---|---|---|
| 10 weights-only, bf16, HF safetensors | `hf_ckpts/step_NNNNN/` | optimizer steps 1, 2, 5, 10, 21, 46, 98, 211, 454, 976 (`round(976 ** (i/9))`, i = 0..9) |
| 1 full resumable Megatron checkpoint (model, optimizer, scheduler, RNG, data cursor) | `ckpt/iter_NNNNNNN/` + `ckpt/rollout/` | when < 45 min + 2 step-times + last full-save time remain in the job, or at step 976. Older full checkpoints are deleted after a new one succeeds |

Logs: `ckpt_log.jsonl` (every save with timings), `jobs.log` (one line per Slurm job), `STATUS`.
Implementation: `jobs/lib/sft_ckpt.py`; mechanics tested by `jobs/test_sft_ckpt_tiny.sbatch`.

## Choices made without a spec (change in the config file if needed)

Peak LR (1e-5), warmup (5%), min LR (1e-6), global batch (32), β2 (0.95), weight decay (0.1).
Specified by the user: dataset/subset/split, row selection (0-based index % 3 == 1), 1 epoch, no router freezing,
10 log-spaced bf16 weights-only checkpoints, one full resumable checkpoint near the end.
