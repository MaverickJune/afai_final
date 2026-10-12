#!/bin/bash
# Runs INSIDE the slime container (called by jobs/verify_router_freeze.sbatch).
# usage: router_freeze_test.sh WORK_DIR PROJECT_DIR
set -euo pipefail
WORK=$1
PROJ=$2
HF=$WORK/hf_tiny_moe
cd /root/slime

# Same Megatron args as scripts/models/qwen3-30B-A3B.sh, shrunk to match make_tiny_moe.py
MODEL_ARGS=(
   --disable-bias-linear --qk-layernorm --group-query-attention
   --num-attention-heads 8 --num-query-groups 4 --kv-channels 64
   --num-layers 2 --hidden-size 512 --ffn-hidden-size 1536
   --normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6
   --rotary-percent 1.0 --swiglu --untie-embeddings-and-output-weights
   --vocab-size 151936 --rotary-base 1000000
   --moe-ffn-hidden-size 256 --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall --moe-router-topk 2 --moe-layer-freq "[1,1]"
   --num-experts 8 --moe-grouped-gemm --moe-token-drop-policy probs
   --moe-router-dtype fp32 --moe-permute-fusion --moe-aux-loss-coeff 0
)

echo "=== [1] HF -> torch_dist (MoE)"
python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "$HF" --save "$WORK/torch_dist"

echo "=== [2] Megatron parameter names containing 'router'"
python - "$WORK/torch_dist/release" <<'EOF'
import sys
from torch.distributed.checkpoint import FileSystemReader
keys = sorted(FileSystemReader(sys.argv[1]).read_metadata().state_dict_metadata.keys())
print(len(keys), "keys total")
for k in keys:
    if "router" in k or "layers.0." in k:
        print("  ", k)
EOF

echo "=== [3] torch_dist -> HF of the unmodified conversion (MoE round trip)"
python "$PROJ/jobs/lib/convert_torch_dist_to_hf.py" --input-dir "$WORK/torch_dist/release" \
    --output-dir "$WORK/hf_roundtrip" --origin-hf-dir "$HF" --vocab-size 151936
python "$PROJ/jobs/lib/compare_hf.py" "$HF" "$WORK/hf_roundtrip"

RAY_PORT=$((20000 + SLURM_JOB_ID % 5000))
DASH_PORT=$((RAY_PORT + 1))
export no_proxy=127.0.0.1
ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" --dashboard-port "$DASH_PORT" \
    --num-gpus 1 --disable-usage-stats --temp-dir "/tmp/ray_${SLURM_JOB_ID}"
trap 'ray stop --force || true' EXIT

run_sft() {   # $1 = run name, rest = extra args
    local name=$1; shift
    echo "=== [4] SFT run: $name  extra args: $*"
    ray job submit --address="http://127.0.0.1:$DASH_PORT" \
        --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
        -- python3 train.py \
        --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$HF" --ref-load "$WORK/torch_dist" \
        --load "$WORK/$name" --save "$WORK/$name" --save-interval 1 \
        --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
        --prompt-data "$WORK/sft_tiny.parquet" --input-key messages \
        --num-epoch 1 --rollout-batch-size 16 --global-batch-size 16 \
        --loss-type sft_loss --calculate-per-token-loss \
        --disable-compute-advantages-and-returns --debug-train-only \
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
        --use-dynamic-batch-size --max-tokens-per-gpu 4096 \
        --optimizer adam --lr 1e-3 --lr-decay-style constant --weight-decay 0.0 \
        --adam-beta1 0.9 --adam-beta2 0.95 \
        --attention-dropout 0.0 --hidden-dropout 0.0 \
        --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \
        "$@"
    local last
    last=$(cat "$WORK/$name/latest_checkpointed_iteration.txt")
    echo "latest iteration: $last"
    python "$PROJ/jobs/lib/convert_torch_dist_to_hf.py" --input-dir "$WORK/$name/$(printf 'iter_%07d' "$last")" \
        --output-dir "$WORK/${name}_hf" --origin-hf-dir "$HF" --vocab-size 151936
}

run_sft sft_frozen --freeze-params-name-list 'mlp\.router\.'
run_sft sft_control

echo "=== [5] frozen run vs original: router (HF mlp.gate) must be identical, attention must change"
python "$PROJ/jobs/lib/compare_hf.py" "$HF" "$WORK/sft_frozen_hf" \
    --expect-equal 'mlp\.gate\.weight$' --expect-changed 'self_attn\.q_proj\.weight$' && FROZEN=PASS || FROZEN=FAIL
echo "=== [6] control run vs original: router must change"
python "$PROJ/jobs/lib/compare_hf.py" "$HF" "$WORK/sft_control_hf" \
    --expect-changed 'mlp\.gate\.weight$' && CONTROL=PASS || CONTROL=FAIL
echo "ROUTER FREEZE TEST: frozen=$FROZEN control=$CONTROL"
