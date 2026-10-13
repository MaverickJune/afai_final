#!/bin/bash
#SBATCH --job-name=test_sft_ckpt_tiny
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out
# Cheap end-to-end test of the SFT checkpoint machinery (jobs/lib/sft_ckpt.py) on 1 GPU, using the tiny random
# Qwen3-MoE from verify_router_freeze and the REAL prepared OpenR1 data, with the same config file as the 30B run.
#   phase A: 6-step run, forced full save + stop after step 3   (= the time-limit path)
#   phase B: same command again -> resumes at step 4, runs to step 6, final full save
# Checks: STATUS, one full checkpoint left, HF checkpoints at the log-spaced steps {1, 2, 6},
#         resumed start step, HF export == torch_dist->HF conversion of the final full checkpoint.
set -euo pipefail
source jobs/slime_common.sh
SFT_RUN=$SCR/verify/sft_ckpt_tiny
source jobs/configs/sft_openr1_idx3k1.sh
echo "host: $(hostname)  start: $(date)"
TINY=$SCR/verify/router_freeze
rm -rf "$SFT_RUN"   # outputs of a previous run of this test only
mkdir -p "$SFT_RUN"

TEST_ENV=(
    "${SFT_ENV[@]}"
    NGPUS=1 MODEL_ARGS_FILE="$PWD/jobs/lib/tiny_moe_model_args.sh"
    HF_CKPT="$TINY/hf_tiny_moe" REF_LOAD="$TINY/torch_dist"
    NUM_ROLLOUT=6 ROLLOUT_BS=4 GLOBAL_BS=4 SFT_N_LOG_CKPTS=3 LR=1e-4
    SFT_SAVE_MARGIN_MIN=0 JOB_END_EPOCH=$(( $(date +%s) + 1500 ))
    RAY_CPUS=$SLURM_CPUS_PER_TASK RAY_OBJ_STORE_GB=16
)

echo "=== phase A"
slime_exec env "${TEST_ENV[@]}" SFT_STOP_AFTER_STEP=3 bash jobs/lib/run_30b.sh sft "$SFT_RUN" "${SFT_ARGS[@]}"
echo "STATUS after A: $(cat "$SFT_RUN/STATUS")"; ls "$SFT_RUN/ckpt" "$SFT_RUN/hf_ckpts"

echo "=== phase B (resume)"
slime_exec env "${TEST_ENV[@]}" bash jobs/lib/run_30b.sh sft "$SFT_RUN" "${SFT_ARGS[@]}"
echo "STATUS after B: $(cat "$SFT_RUN/STATUS")"; ls "$SFT_RUN/ckpt" "$SFT_RUN/hf_ckpts"

echo "=== checks"
cat "$SFT_RUN/ckpt_log.jsonl"
fail=0
grep -q "^done" "$SFT_RUN/STATUS" || { echo "FAIL: STATUS not done"; fail=1; }
[ "$(ls -d "$SFT_RUN"/ckpt/iter_* | wc -l)" = 1 ] && [ -d "$SFT_RUN/ckpt/iter_0000005" ] \
    || { echo "FAIL: expected exactly ckpt/iter_0000005"; fail=1; }
[ "$(ls "$SFT_RUN/hf_ckpts")" = "$(printf 'step_00001\nstep_00002\nstep_00006')" ] \
    || { echo "FAIL: hf_ckpts should be step_00001 step_00002 step_00006"; fail=1; }
grep -q '"event": "schedule".*"start_rollout_id": 3' "$SFT_RUN/ckpt_log.jsonl" \
    || { echo "FAIL: phase B did not resume at rollout 3"; fail=1; }
slime_exec python jobs/lib/convert_torch_dist_to_hf.py --input-dir "$SFT_RUN/ckpt/iter_0000005" \
    --output-dir "$SFT_RUN/from_full_ckpt" --origin-hf-dir "$TINY/hf_tiny_moe" --vocab-size 151936
slime_exec python jobs/lib/compare_hf.py "$SFT_RUN/from_full_ckpt" "$SFT_RUN/hf_ckpts/step_00006" \
    || { echo "FAIL: HF export differs from the full checkpoint"; fail=1; }
slime_exec python jobs/lib/compare_hf.py "$TINY/hf_tiny_moe" "$SFT_RUN/hf_ckpts/step_00006" \
    --expect-changed 'mlp\.gate\.weight$' || { echo "FAIL: router did not train"; fail=1; }
du -sh "$SFT_RUN"/*
[ $fail = 0 ] && echo "SFT CKPT TEST: PASS" || { echo "SFT CKPT TEST: FAIL"; exit 1; }
