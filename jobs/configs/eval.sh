# Evaluation of the post-trained models (sourced by jobs/*eval*.sbatch after jobs/slime_common.sh, which sets SCR).
# Plan and analysis: experiments/eval/PLAN.md. Code: jobs/lib/eval/.

BASE_HF=$SCR/models/Qwen3-30B-A3B-Base
BENCH_DIR=$SCR/data/eval_bench                       # jobs/prep_eval_bench.sbatch (benchmarks + probes/)
EVAL_ROOT=$SCR/eval                                  # outputs (summaries are also copied to experiments/eval/results)
SFT_RUN=$SCR/runs/sft_openr1math_idx3k1_qwen3_30b_a3b_base
GRPO_RUN=$SCR/runs/grpo_openr1math_idx3k0_qwen3_30b_a3b_base

# Final (highest-step) HF checkpoint of a run dir.
final_ckpt() { ls -d "$1"/hf_ckpts/step_* | sort | tail -1; }

# Run registry: run_info NAME sets RUN (run dir with hf_ckpts/) and REF (the model the stage started from: changes
# are measured against it). REF_OVERRIDE=<dir> replaces REF, e.g. REF_OVERRIDE=$BASE_HF for a second stage's
# cumulative change since the base model.
run_info() {
    case $1 in
        sft)  RUN=$SFT_RUN;  REF=$BASE_HF ;;           # stage 1 of SFT -> RL
        grpo) RUN=$GRPO_RUN; REF=$BASE_HF ;;           # stage 1 of RL -> SFT
        # Second stages / ablations: add when their runs exist, e.g.
        # sft_grpo) RUN=$SCR/runs/<sft-then-grpo run>; REF=$(final_ckpt "$SFT_RUN") ;;
        # grpo_sft) RUN=$SCR/runs/<grpo-then-sft run>; REF=$(final_ckpt "$GRPO_RUN") ;;
        # sft_frozen) RUN=$SCR/runs/<frozen-router SFT run>; REF=$BASE_HF ;;
        *) echo "unknown run '$1' (see run_info in jobs/configs/eval.sh)"; return 1 ;;
    esac
    REF=${REF_OVERRIDE:-$REF}
}

# LABEL=DIR for every saved HF checkpoint of a run, in step order: run_ckpts PREFIX RUN_DIR
run_ckpts() {
    local prefix=$1 run=$2 d
    for d in "$run"/hf_ckpts/step_*; do
        [ -d "$d" ] && echo "${prefix}_$(basename "$d")=$d"
    done
}

# afai_env for the analysis jobs (probes, parameter diff). Not for the SGLang jobs, which use the container.
activate_afai_env() {
    module reset && module load miniconda && conda activate ~/envs/afai_env
    export PYTHONNOUSERSITE=1
}
