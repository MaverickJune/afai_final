#!/bin/bash
#SBATCH --job-name=prep_openr1_sft
#SBATCH --partition=day
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --output=logs/%x_%j.out
# CPU-only preparation + validation for jobs/sft_openr1_30b.sbatch:
#   1) first 3001 rows of open-r1/OpenR1-Math-220k (default/train) -> jsonl + manifest (revision, sha256, token stats)
#   2) loss-mask check with the Qwen3-30B-A3B-Base tokenizer (<think> must be trained)
#   3) checkpoint schedule of sft_ckpt.py for the configured run
#   4) DRY_RUN: slime parses + validates the exact training argument list
set -euo pipefail
source jobs/slime_common.sh
source jobs/configs/sft_openr1_3001.sh
echo "host: $(hostname)  start: $(date)"

echo "=== [1+2] data"
slime_exec python jobs/lib/prep_openr1_sft.py "$SFT_DATA_DIR" "$SCR/models/Qwen3-30B-A3B-Base" 3001

echo "=== [3] checkpoint schedule (279 steps = 3 epochs x 3001 // 32)"
slime_exec env PYTHONPATH="$PWD/jobs/lib" python -c "
import sft_ckpt
print('log-spaced HF checkpoint steps:', sft_ckpt.log_spaced_steps(279, 10))
for t, n in [(6, 3), (5, 10), (1000, 10)]:
    print(f'  check T={t} n={n}:', sft_ckpt.log_spaced_steps(t, n))
"

echo "=== [4] DRY_RUN of the exact training arguments"
slime_exec env "${SFT_ENV[@]}" NGPUS=8 DRY_RUN=1 bash jobs/lib/run_30b.sh sft "$SFT_RUN" "${SFT_ARGS[@]}"
echo "end: $(date)"
