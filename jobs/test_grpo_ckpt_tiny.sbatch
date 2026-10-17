#!/bin/bash
#SBATCH --job-name=test_grpo_ckpt_tiny
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out
# Cheap end-to-end test of GRPO (colocated SGLang rollout, MoE weight sync) with the checkpoint machinery
# (jobs/lib/sft_ckpt.py) on 1 GPU, using the tiny random Qwen3-MoE from verify_router_freeze and the REAL prepared
# GRPO prompts, with the same config file as the 30B run.
#   phase A: 6-step run, forced full save + stop after step 3   (= the time-limit path)
#   phase B: same command again -> resumes at step 4, runs to step 6, final full save
# Step 0 also dry-runs the real 30B argument list and saves every resolved argument to
# experiments/grpo_openr1math_idx3k0/resolved_args.json.
# The random model never solves anything, so this test uses --rm-type random to get a learning signal
# (the math reward itself is checked on the R1 reference solutions by the prep job). JOB_END_EPOCH is not set, so
# the time-based full save is off: phase A's stop is forced, and SGLang startup cannot trigger an early stop.
# Checks: Megatron -> SGLang weight sync bit-exact (phase A), STATUS, one full checkpoint left, HF checkpoints at the
#         log-spaced steps {1, 2, 6}, resumed start step, HF export == torch_dist->HF conversion of the final full
#         checkpoint, router trained.
set -euo pipefail
source jobs/slime_common.sh
GRPO_RUN=$SCR/verify/grpo_ckpt_tiny
source jobs/configs/grpo_openr1_idx3k0.sh
echo "host: $(hostname)  start: $(date)"

echo "=== [0] DRY_RUN of the REAL 30B launch arguments (parse + full slime/Megatron validation on this GPU node)"
(
    unset GRPO_RUN
    source jobs/configs/grpo_openr1_idx3k0.sh
    slime_exec env "${GRPO_ENV[@]}" NGPUS=8 DRY_RUN=1 \
        DRY_RUN_ARGS_JSON="$PWD/experiments/grpo_openr1math_idx3k0/resolved_args.json" \
        bash jobs/lib/run_30b.sh grpo "$GRPO_RUN" "${GRPO_ARGS[@]}"
)

TINY=$SCR/verify/router_freeze
rm -rf "$GRPO_RUN"   # outputs of a previous run of this test only
mkdir -p "$GRPO_RUN"

TEST_ENV=(
    "${GRPO_ENV[@]}"
    NGPUS=1 MODEL_ARGS_FILE="$PWD/jobs/lib/tiny_moe_model_args.sh"
    HF_CKPT="$TINY/hf_tiny_moe" REF_LOAD="$TINY/torch_dist"
    NUM_ROLLOUT=6 ROLLOUT_BS=4 N_SAMPLES=4 GLOBAL_BS=16 MAX_RESP_LEN=512 SFT_N_LOG_CKPTS=3 LR=1e-4
    RAY_CPUS=$SLURM_CPUS_PER_TASK RAY_OBJ_STORE_GB=16
)
TEST_ARGS=(
    "${GRPO_ARGS[@]}"
    --rm-type random
    --rollout-max-prompt-len 3072   # the tiny model's context is 4096 tokens
    --sglang-cuda-graph-bs 1 2 4 8 16
)

echo "=== phase A"
# --check-weight-update-equal: SGLang snapshots the base weights it loaded, overwrites them with random values, and
# after the trainer's first weight push raises unless every tensor equals the snapshot. Phase A only: after a resume
# the trainer pushes trained weights, which differ from the base snapshot by design.
slime_exec env "${TEST_ENV[@]}" SFT_STOP_AFTER_STEP=3 bash jobs/lib/run_30b.sh grpo "$GRPO_RUN" "${TEST_ARGS[@]}" \
    --check-weight-update-equal
echo "STATUS after A: $(cat "$GRPO_RUN/STATUS")"; ls "$GRPO_RUN/ckpt" "$GRPO_RUN/hf_ckpts"

echo "=== phase B (resume)"
slime_exec env "${TEST_ENV[@]}" bash jobs/lib/run_30b.sh grpo "$GRPO_RUN" "${TEST_ARGS[@]}"
echo "STATUS after B: $(cat "$GRPO_RUN/STATUS")"; ls "$GRPO_RUN/ckpt" "$GRPO_RUN/hf_ckpts"

echo "=== checks"
cat "$GRPO_RUN/ckpt_log.jsonl"
fail=0
grep -q "^done" "$GRPO_RUN/STATUS" || { echo "FAIL: STATUS not done"; fail=1; }
[ "$(ls -d "$GRPO_RUN"/ckpt/iter_* | wc -l)" = 1 ] && [ -d "$GRPO_RUN/ckpt/iter_0000005" ] \
    || { echo "FAIL: expected exactly ckpt/iter_0000005"; fail=1; }
[ "$(ls "$GRPO_RUN/hf_ckpts")" = "$(printf 'step_00001\nstep_00002\nstep_00006')" ] \
    || { echo "FAIL: hf_ckpts should be step_00001 step_00002 step_00006"; fail=1; }
grep -q '"event": "schedule".*"start_rollout_id": 3' "$GRPO_RUN/ckpt_log.jsonl" \
    || { echo "FAIL: phase B did not resume at rollout 3"; fail=1; }
slime_exec python jobs/lib/convert_torch_dist_to_hf.py --input-dir "$GRPO_RUN/ckpt/iter_0000005" \
    --output-dir "$GRPO_RUN/from_full_ckpt" --origin-hf-dir "$TINY/hf_tiny_moe" --vocab-size 151936
slime_exec python jobs/lib/compare_hf.py "$GRPO_RUN/from_full_ckpt" "$GRPO_RUN/hf_ckpts/step_00006" \
    || { echo "FAIL: HF export differs from the full checkpoint"; fail=1; }
slime_exec python jobs/lib/compare_hf.py "$TINY/hf_tiny_moe" "$GRPO_RUN/hf_ckpts/step_00006" \
    --expect-changed 'mlp\.gate\.weight$' || { echo "FAIL: router did not train"; fail=1; }
du -sh "$GRPO_RUN"/*
[ $fail = 0 ] && echo "GRPO CKPT TEST: PASS" || { echo "GRPO CKPT TEST: FAIL"; exit 1; }
