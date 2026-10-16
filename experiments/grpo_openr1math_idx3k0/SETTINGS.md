# GRPO: Qwen3-30B-A3B-Base on OpenR1-Math-220k, every third row (0-based 0, 3, 6, ...), 1 epoch

Runs **before** the SFT run (`experiments/sft_openr1math_idx3k1/`). The two use disjoint rows of the same split:
GRPO `iloc[0::3]`, SFT `iloc[1::3]`. The SFT run is currently configured to start from the base model; whether it
starts from this run's final checkpoint is not decided yet.

Single source of truth for the values: `jobs/configs/grpo_openr1_idx3k0.sh` (+ defaults in `jobs/lib/run_30b.sh`).

Where the settings are recorded:

| File | Contents |
|---|---|
| this file | the training-relevant settings in words, incl. the important defaults |
| `resolved_args.json` (this folder) | **every** resolved argument incl. all slime / Megatron / TransformerEngine / SGLang defaults, from the CPU dry run of the exact launch command (written by `jobs/prep_openr1_grpo.sbatch`; `num_rollout` is derived at startup: 976) |
| `$GRPO_RUN/train_args_<timestamp>.txt` | the explicit command-line arguments of each launch |
| `$GRPO_RUN/resolved_args_<jobid>_<timestamp>.json` | every resolved argument of each real launch (copy in `logs/resolved_args_*`) |
| job log `logs/grpo_openr1_30b_<jobid>.out` | slime/Megatron also print all arguments at startup |

## How to reproduce (other account, own node)

1. Set up the environment: `env_guide.txt` (container, afai_env, `SCR`, verification jobs incl.
   `verify_router_freeze`), then `jobs/download_30b.sbatch` + `jobs/convert_30b.sbatch` (base model).
2. `cd ~/afai_project`
   `P=$(sbatch --parsable jobs/prep_openr1_grpo.sbatch)`   (CPU only)
   `T=$(sbatch --parsable --dependency=afterok:$P jobs/test_grpo_ckpt_tiny.sbatch)`   (1 GPU, needs the tiny model from `verify_router_freeze`)
   `sbatch --dependency=afterok:$T jobs/grpo_openr1_30b.sbatch`
3. If the job stops with `STATUS = needs_resume ...`, submit `jobs/grpo_openr1_30b.sbatch` again.
4. Check `$SCR/data/openr1_math_grpo/manifest.json`: revision and `output_sha256` should match the table below.

## Model and software

| | |
|---|---|
| Base model | `Qwen/Qwen3-30B-A3B-Base` (HF), converted to Megatron torch_dist with `jobs/convert_30b.sbatch` |
| Framework | slime v0.4.0 (`0b0c277`), Megatron-LM `4b4acac`, SGLang 0.5.15.post1, image `slimerl/slime@sha256:626fbff9…` (see SETUP.md) |
| Precision | bf16 params/activations, fp32 router (`--moe-router-dtype fp32`), fp32 grad accumulation/all-reduce, fp32 attention softmax; SGLang rollout in bf16 |
| Router | **trained** (no `--freeze-params-name-list`); aux load-balancing loss coefficient 0 (`--moe-aux-loss-coeff 0`) |
| Dropout | 0 (attention and hidden) |

## Data

| | |
|---|---|
| Dataset | `open-r1/OpenR1-Math-220k`, config `default`, split `train` (93,733 rows) |
| Revision | `e4e141ec9dea9f8326f4d347be56105859b2bd68` (pinned in `prep_openr1_grpo.py`; the SFT data's snapshot) |
| Selection | **0-based rows 0, 3, 6, ..., 93732 of the train split in dataset order** = pandas `iloc[0::3]` over `data/train-0000{0..9}-of-00010.parquet` concatenated in order → **31,245 rows**. No filtering, no dedup (same practice as the SFT run) |
| Question types | QUESTION_TYPES |
| Prompt | one user turn: `"Please reason step by step, and put your final answer within \boxed{}.\n\n"` + the row's user message verbatim (the same user turn the SFT run trains on). The dataset card says R1 generated the solutions with this instruction prepended; `messages` stores the problem without it. Wrapped in the base model's Qwen3 chat template (`--apply-chat-template`, generation prompt `<\|im_start\|>assistant\n`; thinking is neither forced nor disabled). The prep log prints the rendered first prompt |
| Label | the row's `answer` (string, as in the dataset) |
| Output file | `$SCR/data/openr1_math_grpo/openr1_math_default_train_idx3k0_grpo.jsonl`, sha256 OUTPUT_SHA256 |
| Prompt lengths | PROMPT_TOKEN_STATS |

## Reward

| | |
|---|---|
| Type | `--rm-type math` (slime `grade_answer_verl`): 1 if the **last `\boxed{…}` anywhere in the response** equals the label after slime's MATH normalization or sympy equivalence, else 0. No format reward, no length penalty. Truncated responses are graded like any other |
| Why not slime's example (`deepscaler`) | it gives 0 to every response without `</think>`; the base model is not trained on that format yet |
| Coverage (information only) | REWARD_COVERAGE (from manifest.json: rows whose R1 reference solution earns reward 1, overall and by question type, a rough ceiling for the mean training reward; plus the dataset's own math_verify count. MCQ labels are sometimes the letter and sometimes the option's value) |

## Training

| | |
|---|---|
| Algorithm | GRPO (`--advantage-estimator grpo`): 8 responses per prompt, advantage = (reward − group mean) / (group std + 1e-6); PPO-style clipped objective, ε = 0.2 low / 0.28 high (`--eps-clip 0.2 --eps-clip-high 0.28`); KL loss on with coefficient 0 (`--use-kl-loss --kl-loss-coef 0 --kl-loss-type low_var_kl`: reference log-probs are computed and logged, no KL penalty); entropy coefficient 0 |
| Rollout sampling | temperature 1.0, top-p 1.0, top-k −1 (no truncation), max response 8192 tokens; one SGLang engine with TP 8, colocated with training (`--colocate`: the GPUs alternate between SGLang and Megatron), `--sglang-mem-fraction-static 0.7` |
| Batch | 32 prompts × 8 responses = 256 per step, global batch 256 → **1 optimizer step per rollout** (on-policy); `--balance-data`; micro-batches packed to ≤ 20480 tokens per GPU (`--use-dynamic-batch-size`) |
| Epochs / steps | 1 epoch; `num_rollout = 31245 // 32 = 976` steps (31,232 prompts); the last 13 prompts of the shuffled order are not used |
| Data order | one random permutation of the 31,245 rows: `random.Random(42 + 0).shuffle` (`--rollout-shuffle`, `--rollout-seed 42`), consumed in order, 32 per step; Megatron seed 1234 |
| Optimizer | Adam (Megatron distributed optimizer; precision-aware, states on CPU), β1 0.9, β2 0.98, ε 1e-8, weight decay 0.1, grad clip 1.0 |
| LR schedule | **constant 1e-6** (no warmup, no decay) |
| Parallelism (8 GPUs) | training: TP 4 + sequence parallel, EP 8, PP 1, CP 1 → DP 2; full activation recompute (uniform, 1 layer) |
| Hardware | 1 node, 8× H200 or 8× B200 (`-p gpu_h200,gpu_b200`), 48 CPUs, 1000 GB RAM |

## Evaluation

None during training (no `--eval-interval`): benchmarks are run on the saved checkpoints after post-training.
Progress during the run: the per-step training reward (`rollout/raw_reward`, `./jobstat.sh grpo`). With 1 epoch,
every step's 32 prompts are new to the model.

## Checkpoints (`$GRPO_RUN = $SCR/runs/grpo_openr1math_idx3k0_qwen3_30b_a3b_base`)

| What | Where | When |
|---|---|---|
| 10 weights-only, bf16, HF safetensors | `hf_ckpts/step_NNNNN/` | optimizer steps 1, 2, 5, 10, 21, 46, 98, 211, 454, 976 (`round(976 ** (i/9))`, i = 0..9) |
| 1 full resumable Megatron checkpoint (model, optimizer, scheduler, RNG, data cursor) | `ckpt/iter_NNNNNNN/` + `ckpt/rollout/` | when < 45 min + 2 step-times + last full-save time remain in the job, or at step 976. Older full checkpoints are deleted after a new one succeeds |

Logs: `ckpt_log.jsonl` (every save with timings), `jobs.log` (one line per Slurm job), `STATUS`.
Implementation: `jobs/lib/sft_ckpt.py` (shared with the SFT run); GRPO mechanics tested by `jobs/test_grpo_ckpt_tiny.sbatch`.

## Choices made without a spec (change in the config file if needed)

From slime's Qwen3-30B-A3B GRPO example (`scripts/run-qwen3-30B-A3B.sh`): LR 1e-6 constant, β2 0.98, weight decay 0.1,
32 prompts × 8 responses, global batch 256, max response 8192, rollout temperature 1.0, clip 0.2/0.28,
KL coefficient 0, entropy 0, `--sglang-mem-fraction-static 0.7`.
Specified by the user: dataset/split, row selection (0-based index % 3 == 0), no filtering (as in SFT), 1 epoch,
router trained, `math` reward with the boxed instruction, slime's example hyperparameters except its in-training
AIME eval (dropped), KL reference model kept (coefficient 0) for monitoring, the SFT run's checkpoint scheme,
GRPO before SFT, prompt = SFT user turn with the dataset's generation instruction prepended.
