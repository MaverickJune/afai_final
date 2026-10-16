#!/bin/bash
#SBATCH --job-name=eval_specialization
#SBATCH --partition=day
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out
# CPU only: are expert updates concentrated on math-specialized experts? Needs the outputs of
# eval_probe.sbatch and eval_param_diff.sbatch for the same run (jobs/lib/eval/expert_specialization.py).
# usage: sbatch jobs/eval_specialization.sbatch RUN_NAME
set -eo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
NAME=$1${REF_OVERRIDE:+_vs_$(basename "$REF_OVERRIDE")}
activate_afai_env
mkdir -p experiments/eval/results/specialization
python jobs/lib/eval/expert_specialization.py --probe-out "$EVAL_ROOT/probe/$NAME" \
    --param-diff "$EVAL_ROOT/param_diff/$NAME" --out "$EVAL_ROOT/specialization_$NAME.csv"
cp "$EVAL_ROOT/specialization_$NAME.csv" experiments/eval/results/specialization/
