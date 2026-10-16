#!/bin/bash
#SBATCH --job-name=eval_param_diff
#SBATCH --partition=day
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only: per-tensor relative Frobenius change of every saved HF checkpoint of a run vs the run's reference
# model (jobs/lib/eval/param_diff.py, afai_env). Reads 57 GB per checkpoint from scratch.
# usage: sbatch jobs/eval_param_diff.sbatch RUN_NAME      (RUN_NAME: see run_info in jobs/configs/eval.sh)
set -eo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
NAME=$1
run_info "$NAME"
mapfile -t CKPTS < <(run_ckpts "$NAME" "$RUN")
[ ${#CKPTS[@]} -gt 0 ] || { echo "no checkpoints in $RUN/hf_ckpts"; exit 1; }
echo "run $NAME: $RUN   reference: $REF"; printf '  %s\n' "${CKPTS[@]}"
activate_afai_env
echo "host: $(hostname)  start: $(date)"
OUT=$EVAL_ROOT/param_diff/$NAME${REF_OVERRIDE:+_vs_$(basename "$REF_OVERRIDE")}
python jobs/lib/eval/param_diff.py --base "$REF" --out "$OUT" $(printf -- '--ckpt %s ' "${CKPTS[@]}")
R=experiments/eval/results/param_diff/$(basename "$OUT"); mkdir -p "$R"
cp "$OUT/param_diff_modules.csv" "$R/"
echo "end: $(date)"
