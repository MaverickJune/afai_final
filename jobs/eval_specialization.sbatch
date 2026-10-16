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
run_info "$1"
mapfile -t CKPTS < <(run_ckpts "$1" "$RUN")
NAME=$1${REF_OVERRIDE:+_vs_$(basename "$REF_OVERRIDE")}
activate_afai_env
# merge the probe-analysis parts of all array tasks first
python jobs/lib/eval/probe_analysis.py --combine-only --ref "$REF" --probe-dir "$BENCH_DIR/probes" \
    --out "$EVAL_ROOT/probe/$NAME" $(printf -- '--ckpt %s ' "${CKPTS[@]}")
R=experiments/eval/results/probe/$NAME; mkdir -p "$R"; cp "$EVAL_ROOT/probe/$NAME"/kl.csv "$EVAL_ROOT/probe/$NAME"/routing.csv "$R/"
mkdir -p experiments/eval/results/specialization
python jobs/lib/eval/expert_specialization.py --probe-out "$EVAL_ROOT/probe/$NAME" \
    --param-diff "$EVAL_ROOT/param_diff/$NAME" --out "$EVAL_ROOT/specialization_$NAME.csv"
cp "$EVAL_ROOT/specialization_$NAME.csv" experiments/eval/results/specialization/
