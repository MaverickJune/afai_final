#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --partition=day
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:20:00
#SBATCH --output=logs/%x_%j.out
# CPU only: figures of the analysis, from experiments/eval/results/ -> experiments/eval/figures/<RUN>/
# (jobs/lib/eval/plot_results.py). Figure 6 also needs $EVAL_ROOT/specialization_<RUN>_experts.csv.gz (from
# eval_specialization); without it figure 6 is skipped and the existing file is kept.
# usage: sbatch jobs/plot_results.sbatch [RUN ...]
#   RUN: sft | grpo | grpo_sft | grpo_sft_vs_Qwen3-30B-A3B-Base   one run vs its reference (figures fig1..fig8)
#        compare                                                  all runs on shared axes  (figures c1..c5)
#   default: every run with results in experiments/eval/results/probe/, then compare
set -eo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
activate_afai_env
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || RUNS=($(ls experiments/eval/results/probe) compare)
for RUN in "${RUNS[@]}"; do
    echo "=== $RUN"
    python jobs/lib/eval/plot_results.py experiments/eval/results experiments/eval/figures "$RUN" \
        "$EVAL_ROOT/specialization_${RUN}_experts.csv.gz"
done
