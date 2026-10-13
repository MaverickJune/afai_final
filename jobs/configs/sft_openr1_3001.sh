# Configuration of the Qwen3-30B-A3B-Base SFT run on the first 3001 rows of OpenR1-Math-220k.
# Sourced (on the host, after jobs/slime_common.sh) by jobs/sft_openr1_30b.sbatch (training),
# jobs/prep_openr1_sft.sbatch (DRY_RUN argument check) and jobs/test_sft_ckpt_tiny.sbatch (mechanics test).
# Read at job start, so edits apply to already-queued jobs. Settings are documented in
# experiments/sft_openr1math_3001/SETTINGS.md -- keep the two in sync.

SFT_NAME=sft_openr1math_3001_qwen3_30b_a3b_base
SFT_DATA_DIR=$SCR/data/openr1_math_sft
SFT_DATA=$SFT_DATA_DIR/openr1_math_default_train_first3001.jsonl
SFT_RUN=${SFT_RUN:-$SCR/runs/$SFT_NAME}

# Environment for jobs/lib/run_30b.sh and jobs/lib/sft_ckpt.py (passed with `slime_exec env ...`).
SFT_ENV=(
    FREEZE_ROUTER=0                       # router trains (no freezing)
    PROMPT_DATA="$SFT_DATA"
    NUM_EPOCH=3
    ROLLOUT_BS=32 GLOBAL_BS=32            # 1 optimizer step per rollout; 3001 // 32 = 93 steps/epoch -> 279 steps
    LR=1e-5
    MAX_TOKENS_PER_GPU=${SFT_MAX_TOKENS_PER_GPU:-20480}
    SAVE_INTERVAL=1                       # sft_ckpt.py decides what to save at each step
    TRAIN_ENTRY="$PWD/jobs/lib/sft_ckpt.py"
    SFT_CKPT_DIR="$SFT_RUN"
    SFT_N_LOG_CKPTS=10
    SFT_SAVE_MARGIN_MIN=45
    PASS_ENV="SFT_CKPT_DIR SFT_N_LOG_CKPTS JOB_END_EPOCH SFT_SAVE_MARGIN_MIN SFT_STOP_AFTER_STEP"
)

# Extra train.py arguments (override run_30b.sh defaults; argparse keeps the last occurrence).
SFT_ARGS=(
    --loss-mask-type qwen3                # keeps the <think> reasoning in the trained tokens (default "qwen" drops it)
    --lr-decay-style cosine --min-lr 1e-6 --lr-warmup-fraction 0.05
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
    --save-hf "$SFT_RUN/hf_ckpts/_rollout_{rollout_id}"   # path is a placeholder; sft_ckpt.py picks hf_ckpts/step_NNNNN
    --custom-megatron-init-path sft_ckpt.actor_init
)
