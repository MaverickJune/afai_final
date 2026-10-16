#!/bin/bash
#SBATCH --job-name=eval_param_diff
#SBATCH --partition=day
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only: per-tensor relative Frobenius change vs the base model for every saved HF checkpoint of a run
# (jobs/lib/eval/param_diff.py, afai_env). Reads 57 GB per checkpoint from scratch.
# usage: sbatch jobs/eval_param_diff.sbatch [sft|grpo]
set -eo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
RUN_KIND=${1:-sft}
case $RUN_KIND in sft) RUN=$SFT_RUN ;; grpo) RUN=$GRPO_RUN ;; *) echo "unknown run $RUN_KIND"; exit 1 ;; esac
mapfile -t CKPTS < <(run_ckpts "$RUN_KIND" "$RUN")
[ ${#CKPTS[@]} -gt 0 ] || { echo "no checkpoints in $RUN/hf_ckpts"; exit 1; }
printf '  %s\n' "${CKPTS[@]}"
activate_afai_env
echo "host: $(hostname)  start: $(date)"
OUT=$EVAL_ROOT/param_diff/$RUN_KIND
python jobs/lib/eval/param_diff.py --base "$BASE_HF" --out "$OUT" $(printf -- '--ckpt %s ' "${CKPTS[@]}")
mkdir -p experiments/eval/results/param_diff
cp "$OUT/param_diff_modules.csv" "experiments/eval/results/param_diff/param_diff_modules_$RUN_KIND.csv"
echo "end: $(date)"
