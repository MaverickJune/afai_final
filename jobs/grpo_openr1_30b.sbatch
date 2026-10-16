#!/bin/bash
#SBATCH --job-name=grpo_openr1_30b
#SBATCH --partition=gpu_h200,gpu_b200
#SBATCH --gpus=8
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=1000G
#SBATCH -t 48:00:00
#SBATCH --output=logs/%x_%j.out
# Qwen3-30B-A3B-Base GRPO, OpenR1-Math rows 0,3,6,.. (31,245 prompts), 1 epoch, router trained
# (dataset: open-r1/OpenR1-Math-220k default/train, pandas iloc[0::3]; disjoint from the SFT run's iloc[1::3]).
# Settings: jobs/configs/grpo_openr1_idx3k0.sh, documented in experiments/grpo_openr1math_idx3k0/SETTINGS.md.
# Checkpoints (jobs/lib/sft_ckpt.py), under $GRPO_RUN:
#   hf_ckpts/step_NNNNN/  10 weights-only bf16 HF checkpoints at log-spaced steps of the whole run
#   ckpt/iter_NNNNNNN/    ONE full resumable Megatron checkpoint (near the job time limit, or at the end)
# Resume: submit this same script again (it continues from ckpt/). STATUS says "done" or "needs_resume ...".
set -euo pipefail
source jobs/slime_common.sh
source jobs/configs/grpo_openr1_idx3k0.sh
echo "host: $(hostname)  start: $(date)  job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p "$GRPO_RUN"
if [ -f "$GRPO_RUN/STATUS" ] && grep -q "^done" "$GRPO_RUN/STATUS"; then
    echo "run already finished: $(cat "$GRPO_RUN/STATUS")"; exit 0
fi
JOB_END_EPOCH=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
echo "job ends at $(date -d @"$JOB_END_EPOCH")"
{
    echo "job $SLURM_JOB_ID on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x8"
    echo "  start $(date)  end limit $(date -d @"$JOB_END_EPOCH")  repo $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >> "$GRPO_RUN/jobs.log"

slime_exec env "${GRPO_ENV[@]}" JOB_END_EPOCH="$JOB_END_EPOCH" RAY_CPUS="$SLURM_CPUS_PER_TASK" \
    bash jobs/lib/run_30b.sh grpo "$GRPO_RUN" "${GRPO_ARGS[@]}"

echo "STATUS: $(cat "$GRPO_RUN/STATUS" 2>/dev/null || echo missing)"
ls "$GRPO_RUN/hf_ckpts" "$GRPO_RUN/ckpt" 2>&1
du -sh "$GRPO_RUN"/hf_ckpts "$GRPO_RUN"/ckpt 2>/dev/null || true
echo "end: $(date)"
