#!/bin/bash
#SBATCH --job-name=smoke_30b
#SBATCH --partition=gpu_h200,gpu_b200
#SBATCH --gpus=8
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=1000G
#SBATCH -t 3:00:00
#SBATCH --output=logs/%x_%j.out
# Full-node smoke test on Qwen3-30B-A3B-Base, router frozen:
#   1) SFT, 2 steps (synthetic data), save torch_dist + HF (--save-hf)
#   2) GRPO, 2 rollouts on dapo-math (short responses), save torch_dist + HF
#   3) convert each saved torch_dist -> HF with the wrapper; check router (mlp.gate) == base, attention changed
# Fewer GPUs: sbatch --gpus=4 jobs/smoke_30b.sbatch  (run_30b.sh scales TP/EP)
set -euo pipefail
source jobs/slime_common.sh
echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

BASE=$SCR/models/Qwen3-30B-A3B-Base
RUNS=$SCR/runs/smoke_30b_${SLURM_JOB_ID}
mkdir -p "$RUNS"
LIB=$PWD/jobs/lib

slime_exec python - "$RUNS/sft_smoke.parquet" <<'EOF'
import sys, pandas as pd
rows = [{"messages": [{"role": "user", "content": f"What is {i + 3} plus {2 * i + 7}?"},
                      {"role": "assistant", "content": f"{i + 3} plus {2 * i + 7} equals {3 * i + 10}."}]}
        for i in range(64)]
pd.DataFrame(rows).to_parquet(sys.argv[1]); print("wrote", len(rows), "rows")
EOF

check() {   # $1 = run name; convert last torch_dist save and compare against base
    local run=$RUNS/$1 last
    last=$(cat "$run/ckpt/latest_checkpointed_iteration.txt")
    slime_exec python "$LIB/convert_torch_dist_to_hf.py" --input-dir "$run/ckpt/$(printf 'iter_%07d' "$last")" \
        --output-dir "$run/hf_from_torch_dist" --origin-hf-dir "$BASE" --vocab-size 151936
    echo "--- $1: torch_dist->HF vs base"
    slime_exec python "$LIB/compare_hf.py" "$BASE" "$run/hf_from_torch_dist" \
        --expect-equal 'mlp\.gate\.weight$' --expect-changed 'self_attn\.q_proj\.weight$' || true
    echo "--- $1: --save-hf output:"; ls "$run"/hf 2>&1 | head
    for d in "$run"/hf/*/; do
        [ -d "$d" ] || continue
        echo "--- $1: $d vs torch_dist->HF (should be bit-identical)"
        slime_exec python "$LIB/compare_hf.py" "$run/hf_from_torch_dist" "$d" || true
    done
}

echo "=== [1] SFT smoke"
# slime_exec uses --cleanenv, so settings are passed with `env` inside the container
slime_exec env FREEZE_ROUTER=1 PROMPT_DATA="$RUNS/sft_smoke.parquet" ROLLOUT_BS=32 GLOBAL_BS=32 NUM_EPOCH=1 \
    SAVE_INTERVAL=2 LR=1e-5 MAX_TOKENS_PER_GPU=4096 RAY_CPUS="$SLURM_CPUS_PER_TASK" \
    bash "$LIB/run_30b.sh" sft "$RUNS/sft" --no-save-optim --save-hf "$RUNS/sft/hf/rollout_{rollout_id}" \
    && check sft || echo "SFT SMOKE FAILED"

echo "=== [2] GRPO smoke"
slime_exec env FREEZE_ROUTER=1 NUM_ROLLOUT=2 ROLLOUT_BS=8 N_SAMPLES=4 GLOBAL_BS=32 MAX_RESP_LEN=1024 \
    SAVE_INTERVAL=2 LR=1e-5 RAY_CPUS="$SLURM_CPUS_PER_TASK" \
    bash "$LIB/run_30b.sh" grpo "$RUNS/grpo" --no-save-optim --save-hf "$RUNS/grpo/hf/rollout_{rollout_id}" \
    --sglang-cuda-graph-bs 1 2 4 8 16 32 \
    && check grpo || echo "GRPO SMOKE FAILED"

du -sh "$RUNS"/*
echo "end: $(date)"
