#!/bin/bash
#SBATCH --job-name=eval_kl
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=192G
#SBATCH -t 12:00:00
#SBATCH --output=logs/%x_%j.out
# KL(base || checkpoint) per token on MATH-500 reference solutions, for every saved HF checkpoint of a run
# (jobs/lib/eval/kl_math500.py, afai_env, base + one checkpoint on one H200).
# usage: sbatch jobs/eval_kl.sbatch [sft|grpo] [LIMIT]
set -eo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
RUN_KIND=${1:-sft}; LIMIT=${2:-0}
case $RUN_KIND in sft) RUN=$SFT_RUN ;; grpo) RUN=$GRPO_RUN ;; *) echo "unknown run $RUN_KIND"; exit 1 ;; esac
mapfile -t CKPTS < <(run_ckpts "$RUN_KIND" "$RUN")
[ ${#CKPTS[@]} -gt 0 ] || { echo "no checkpoints in $RUN/hf_ckpts"; exit 1; }
printf '  %s\n' "${CKPTS[@]}"
activate_afai_env
echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$EVAL_ROOT/kl/$RUN_KIND
python jobs/lib/eval/kl_math500.py --base "$BASE_HF" --data "$BENCH_DIR/math500.jsonl" --out "$OUT" --limit "$LIMIT" \
    $(printf -- '--ckpt %s ' "${CKPTS[@]}")
mkdir -p experiments/eval/results/kl && cp "$OUT/kl_math500.csv" "experiments/eval/results/kl/kl_math500_$RUN_KIND.csv"
echo "end: $(date)"
