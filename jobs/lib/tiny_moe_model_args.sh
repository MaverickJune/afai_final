# Megatron args for the tiny random Qwen3-MoE made by jobs/lib/make_tiny_moe.py (2 layers, 8 experts).
# Same flags as scripts/models/qwen3-30B-A3B.sh, shrunk. Used via MODEL_ARGS_FILE for cheap mechanics tests.
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
