#!/bin/bash
#SBATCH --job-name=prep_openr1_grpo
#SBATCH --partition=day
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH --output=logs/%x_%j.out
# CPU-only preparation + validation for jobs/grpo_openr1_30b.sbatch:
#   1) rows iloc[0::3] (0-based 0, 3, 6, ...) of open-r1/OpenR1-Math-220k default/train -> prompt/label jsonl,
#      manifest (reward coverage, prompt lengths)
#   2) checkpoint schedule of sft_ckpt.py for the configured run
# The DRY_RUN argument check runs at the start of jobs/test_grpo_ckpt_tiny.sbatch (needs a GPU:
# Megatron's validation queries the device).
set -euo pipefail
source jobs/slime_common.sh
source jobs/configs/grpo_openr1_idx3k0.sh
echo "host: $(hostname)  start: $(date)"

echo "=== [1] data"
slime_exec python jobs/lib/prep_openr1_grpo.py "$GRPO_DATA_DIR" "$SCR/models/Qwen3-30B-A3B-Base" 0 3

STEPS=$(( $(wc -l < "$GRPO_DATA") / 32 ))   # 1 epoch, 32 prompts per step
echo "=== [2] checkpoint schedule ($STEPS steps)"
slime_exec env PYTHONPATH="$PWD/jobs/lib" python -c "
import sft_ckpt
print('log-spaced HF checkpoint steps:', sft_ckpt.log_spaced_steps($STEPS, 10))
"
echo "end: $(date)"
