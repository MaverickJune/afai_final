# Configuration of the Qwen3-30B-A3B-Base GRPO run on OpenR1-Math-220k (default/train), 0-based rows 0, 3, 6, ...
# (pandas iloc[0::3] -> 31,245 prompts, disjoint from the SFT run's iloc[1::3]), 1 epoch, router trained.
# Sourced (on the host, after jobs/slime_common.sh) by jobs/grpo_openr1_30b.sbatch (training),
# jobs/prep_openr1_grpo.sbatch (DRY_RUN argument check) and jobs/test_grpo_ckpt_tiny.sbatch (mechanics test).
# Read at job start, so edits apply to already-queued jobs. Settings are documented in
# experiments/grpo_openr1math_idx3k0/SETTINGS.md -- keep the two in sync.

GRPO_NAME=grpo_openr1math_idx3k0_qwen3_30b_a3b_base
GRPO_DATA_DIR=$SCR/data/openr1_math_grpo
GRPO_DATA=$GRPO_DATA_DIR/openr1_math_default_train_idx3k0_grpo.jsonl
GRPO_RUN=${GRPO_RUN:-$SCR/runs/$GRPO_NAME}

# Environment for jobs/lib/run_30b.sh and jobs/lib/sft_ckpt.py (passed with `slime_exec env ...`).
# sft_ckpt.py drives both the SFT and the GRPO run; its variables keep the SFT_ prefix.
GRPO_ENV=(
    FREEZE_ROUTER=0                       # router trains (no freezing), as in the SFT run
    PROMPT_DATA="$GRPO_DATA"
    NUM_EPOCH=1                           # 31245 // 32 = 976 steps (1 epoch)
    ROLLOUT_BS=32 N_SAMPLES=8             # 32 prompts x 8 responses per step
    GLOBAL_BS=256                         # = 32 x 8: 1 optimizer step per rollout
    MAX_RESP_LEN=8192
    LR=1e-6
    MAX_TOKENS_PER_GPU=20480
    SAVE_INTERVAL=1                       # sft_ckpt.py decides what to save at each step
    TRAIN_ENTRY="$PWD/jobs/lib/sft_ckpt.py"
    SFT_CKPT_DIR="$GRPO_RUN"
    SFT_N_LOG_CKPTS=10
    SFT_SAVE_MARGIN_MIN=45
    PASS_ENV="SFT_CKPT_DIR SFT_N_LOG_CKPTS JOB_END_EPOCH SFT_SAVE_MARGIN_MIN SFT_STOP_AFTER_STEP"
)

# Extra train.py arguments (override run_30b.sh defaults; argparse keeps the last occurrence).
# No eval during training (sft_ckpt.py requires that); benchmarks run on the saved checkpoints after post-training.
GRPO_ARGS=(
    --rm-type math                        # last \boxed{} anywhere in the response vs label (deepscaler needs </think>)
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
    # SGLang's MoE kernel. "auto" picks Triton on H200 but FlashInfer TRT-LLM on B200, which creates a symlink inside
    # the read-only image at startup and fails (Errno 12, job 29879427). Triton on both GPU types.
    --sglang-moe-runner-backend triton
    --save-hf "$GRPO_RUN/hf_ckpts/_rollout_{rollout_id}"   # path is a placeholder; sft_ckpt.py picks hf_ckpts/step_NNNNN
    --custom-megatron-init-path sft_ckpt.actor_init
)
