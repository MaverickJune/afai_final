#!/bin/bash
#SBATCH --job-name=eval_probe
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=192G
#SBATCH -t 24:00:00
#SBATCH --output=logs/%x_%j.out
# KL drift + routing drift of every saved HF checkpoint of a run vs the run's reference model, on the probe sets
# (jobs/lib/eval/probe_analysis.py, afai_env; reference + one checkpoint on one H200). Resumable per checkpoint.
# usage: sbatch jobs/eval_probe.sbatch RUN_NAME [LIMIT]       (RUN_NAME: see run_info in jobs/configs/eval.sh)
#        REF_OVERRIDE=<dir> sbatch ...  measures against another reference (e.g. the base model)
set -eo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
NAME=$1; LIMIT=${2:-0}
run_info "$NAME"
mapfile -t CKPTS < <(run_ckpts "$NAME" "$RUN")
[ ${#CKPTS[@]} -gt 0 ] || { echo "no checkpoints in $RUN/hf_ckpts"; exit 1; }
echo "run $NAME: $RUN   reference: $REF"; printf '  %s\n' "${CKPTS[@]}"
activate_afai_env
echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$EVAL_ROOT/probe/$NAME${REF_OVERRIDE:+_vs_$(basename "$REF_OVERRIDE")}
python jobs/lib/eval/probe_analysis.py --ref "$REF" --tokenizer "$BASE_HF" --probe-dir "$BENCH_DIR/probes" \
    --out "$OUT" --limit "$LIMIT" $(printf -- '--ckpt %s ' "${CKPTS[@]}")
R=experiments/eval/results/probe/$(basename "$OUT"); mkdir -p "$R"
cp "$OUT/kl.csv" "$OUT/routing.csv" "$R/"
echo "end: $(date)"
