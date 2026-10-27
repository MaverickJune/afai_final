# Configuration of the Qwen3-30B-A3B-Base SFT run on OpenR1-Math-220k (default/train), 0-based rows 1, 4, 7, ...
# (pandas iloc[1::3] -> 31,244 rows), 1 epoch.
# Sourced (on the host, after jobs/slime_common.sh) by jobs/sft_openr1_30b.sbatch (training),
# jobs/prep_openr1_sft.sbatch (DRY_RUN argument check) and jobs/test_sft_ckpt_tiny.sbatch (mechanics test).
# Read at job start, so edits apply to already-queued jobs. Settings are documented in
# experiments/sft_openr1math_idx3k1/SETTINGS.md -- keep the two in sync.

# Start model, chosen at submission with SFT_FROM (default base). It also picks the run dir, so every submission of a
# run (incl. resubmissions after a time-limit stop) needs the same SFT_FROM. Same data and recipe for both:
#   base  Qwen3-30B-A3B-Base (same paths as the run_30b.sh defaults)
#   grpo  final model of the GRPO run (experiments/grpo_openr1math_idx3k0, step 976), converted to torch_dist first
#         by `CONVERT_SRC=grpo_final sbatch jobs/convert_30b.sbatch` (same paths there)
SFT_FROM=${SFT_FROM:-base}
case $SFT_FROM in
    base)
        SFT_NAME=sft_openr1math_idx3k1_qwen3_30b_a3b_base
        SFT_HF_CKPT=$SCR/models/Qwen3-30B-A3B-Base
        SFT_REF_LOAD=$SCR/ckpt/Qwen3-30B-A3B-Base_torch_dist ;;
    grpo)
        SFT_NAME=sft_openr1math_idx3k1_qwen3_30b_a3b_grpo_idx3k0
        SFT_HF_CKPT=$SCR/runs/grpo_openr1math_idx3k0_qwen3_30b_a3b_base/hf_ckpts/step_00976
        SFT_REF_LOAD=$SCR/ckpt/grpo_openr1math_idx3k0_step00976_torch_dist
        # Stop here: with a missing start checkpoint Megatron only warns and trains from random weights
        # (exit_on_missing_checkpoint is false).
        [ -f "$SFT_REF_LOAD/latest_checkpointed_iteration.txt" ] || {
            echo "SFT_FROM=grpo: no $SFT_REF_LOAD; run CONVERT_SRC=grpo_final sbatch jobs/convert_30b.sbatch" >&2
            return 1
        } ;;
    *) echo "unknown SFT_FROM=$SFT_FROM (base | grpo)" >&2; return 1 ;;
esac
SFT_DATA_DIR=$SCR/data/openr1_math_sft
SFT_DATA=$SFT_DATA_DIR/openr1_math_default_train_idx3k1.jsonl
SFT_RUN=${SFT_RUN:-$SCR/runs/$SFT_NAME}

# Environment for jobs/lib/run_30b.sh and jobs/lib/sft_ckpt.py (passed with `slime_exec env ...`).
SFT_ENV=(
    HF_CKPT="$SFT_HF_CKPT"                # start model: HF config + tokenizer (--hf-checkpoint)
    REF_LOAD="$SFT_REF_LOAD"              # start model: torch_dist weights, loaded while ckpt/ is empty (--ref-load)
    FREEZE_ROUTER=0                       # router trains (no freezing)
    PROMPT_DATA="$SFT_DATA"
    NUM_EPOCH=1
    ROLLOUT_BS=32 GLOBAL_BS=32            # 1 optimizer step per rollout; 31244 // 32 = 976 steps (1 epoch)
    LR=1e-5
    MAX_TOKENS_PER_GPU=${SFT_MAX_TOKENS_PER_GPU:-24576}   # >= longest sample (22,169 tokens): no truncation
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
