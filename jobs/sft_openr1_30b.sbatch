#!/bin/bash
#SBATCH --job-name=sft_openr1_30b
#SBATCH --partition=gpu_h200,gpu_b200
#SBATCH --gpus=8
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=1000G
#SBATCH -t 48:00:00
#SBATCH --output=logs/%x_%j.out
# SFT of Qwen3-30B-A3B-Base on OpenR1-Math-220k (default/train) rows iloc[1::3] (31,244), 1 epoch, router NOT frozen.
# Settings: jobs/configs/sft_openr1_idx3k1.sh, documented in experiments/sft_openr1math_idx3k1/SETTINGS.md.
# Checkpoints (jobs/lib/sft_ckpt.py), under $SFT_RUN:
#   hf_ckpts/step_NNNNN/  10 weights-only bf16 HF checkpoints at log-spaced steps of the whole run
#   ckpt/iter_NNNNNNN/    ONE full resumable Megatron checkpoint (near the job time limit, or at the end)
# Resume: submit this same script again (it continues from ckpt/). STATUS says "done" or "needs_resume ...".
set -euo pipefail
source jobs/slime_common.sh
source jobs/configs/sft_openr1_idx3k1.sh
echo "host: $(hostname)  start: $(date)  job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p "$SFT_RUN"
if [ -f "$SFT_RUN/STATUS" ] && grep -q "^done" "$SFT_RUN/STATUS"; then
    echo "run already finished: $(cat "$SFT_RUN/STATUS")"; exit 0
fi
JOB_END_EPOCH=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
echo "job ends at $(date -d @"$JOB_END_EPOCH")"
{
    echo "job $SLURM_JOB_ID on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x8"
    echo "  start $(date)  end limit $(date -d @"$JOB_END_EPOCH")  repo $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >> "$SFT_RUN/jobs.log"

slime_exec env "${SFT_ENV[@]}" JOB_END_EPOCH="$JOB_END_EPOCH" RAY_CPUS="$SLURM_CPUS_PER_TASK" \
    bash jobs/lib/run_30b.sh sft "$SFT_RUN" "${SFT_ARGS[@]}"

echo "STATUS: $(cat "$SFT_RUN/STATUS" 2>/dev/null || echo missing)"
ls "$SFT_RUN/hf_ckpts" "$SFT_RUN/ckpt" 2>&1
du -sh "$SFT_RUN"/hf_ckpts "$SFT_RUN"/ckpt 2>/dev/null || true
echo "end: $(date)"
