#!/bin/bash
#SBATCH --job-name=prep_eval_bench
#SBATCH --partition=day
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only: download + normalise the evaluation benchmarks (MATH-500, AIME24, GSM8K, GPQA-Diamond, MMLU, MMLU-Pro)
# + AIME 2026, HMMT Feb 2026, and the probe sets (OpenR1 held-out rows, MATH-500, general) into $BENCH_DIR
# (jobs/lib/eval/prep_bench.py). GPQA is gated on the Hub; if one fails, the rest are still written.
set -euo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
echo "host: $(hostname)  start: $(date)"
slime_exec python jobs/lib/eval/prep_bench.py "$BENCH_DIR" "$@" || echo "WARNING: some benchmarks failed (see above)"
ls -la "$BENCH_DIR"
mkdir -p experiments/eval && cp "$BENCH_DIR/manifest.json" experiments/eval/bench_manifest.json
echo "end: $(date)"
