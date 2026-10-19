#!/bin/bash
#SBATCH --job-name=plot_sft_results
#SBATCH --partition=day
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:20:00
#SBATCH --output=logs/%x_%j.out
# CPU only: figures of the SFT-stage analysis -> experiments/eval/figures/ (jobs/lib/eval/plot_sft_results.py)
set -eo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
activate_afai_env
python jobs/lib/eval/plot_sft_results.py experiments/eval/results "$EVAL_ROOT/specialization_sft_experts.csv.gz" \
    experiments/eval/figures ${PLOT_ONLY:-}
