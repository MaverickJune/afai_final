# Evaluation of the post-trained models (sourced by jobs/*eval*.sbatch after jobs/slime_common.sh, which sets SCR).
# Plan and analysis: experiments/eval/PLAN.md. Code: jobs/lib/eval/.

BASE_HF=$SCR/models/Qwen3-30B-A3B-Base
BENCH_DIR=$SCR/data/eval_bench                       # jobs/prep_eval_bench.sbatch
EVAL_ROOT=$SCR/eval                                  # outputs (summaries are also copied to experiments/eval/results)
SFT_RUN=$SCR/runs/sft_openr1math_idx3k1_qwen3_30b_a3b_base
GRPO_RUN=$SCR/runs/grpo_openr1math_idx3k0_qwen3_30b_a3b_base

# LABEL=DIR for every saved HF checkpoint of a run, in step order: run_ckpts PREFIX RUN_DIR
run_ckpts() {
    local prefix=$1 run=$2 d
    for d in "$run"/hf_ckpts/step_*; do
        [ -d "$d" ] && echo "${prefix}_$(basename "$d")=$d"
    done
}

# afai_env for the analysis jobs (KL, parameter diff). Not for the SGLang jobs, which use the container.
activate_afai_env() {
    module reset && module load miniconda && conda activate ~/envs/afai_env
    export PYTHONNOUSERSITE=1
}
