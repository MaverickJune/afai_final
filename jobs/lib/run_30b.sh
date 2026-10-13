#!/bin/bash
# Runs INSIDE the slime container. Qwen3-30B-A3B(-Base) SFT or GRPO with slime, single node.
# Adapted from /root/slime/scripts/run-qwen3-30B-A3B.sh and run-qwen3-4B-base-sft.sh
# (without their `pkill -9` preamble, and with Ray sized to the Slurm job).
#
# usage: run_30b.sh {sft|grpo} RUN_DIR [extra train.py args...]
# env:   NGPUS (default: visible GPUs), FREEZE_ROUTER=1, HF_CKPT, REF_LOAD, PROMPT_DATA,
#        NUM_ROLLOUT, NUM_EPOCH, ROLLOUT_BS, N_SAMPLES, GLOBAL_BS, MAX_RESP_LEN, SAVE_INTERVAL, LR, RM_TYPE,
#        RAY_CPUS (default: SLURM_CPUS_PER_TASK), RAY_OBJ_STORE_GB (default 64)
#        MODEL_ARGS_FILE   model config to source (default: slime's scripts/models/qwen3-30B-A3B.sh)
#        TRAIN_ENTRY       driver script (default: slime's train.py), e.g. jobs/lib/sft_ckpt.py
#        PASS_ENV          space-separated env var names forwarded to the Ray driver/actors
#        DRY_RUN=1         only parse+validate the full argument list on CPU (no Ray, no GPU);
#                          DRY_RUN_ARGS_JSON=<file> also writes every resolved argument (incl. defaults)
# jobs/lib is added to PYTHONPATH of the Ray driver and actors, so hook modules there are importable.
set -euo pipefail
MODE=$1
RUN=$2
shift 2
LIB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SCR=${SCR:-$HOME/scratch_pi_lz484/$USER}
NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
HF_CKPT=${HF_CKPT:-$SCR/models/Qwen3-30B-A3B-Base}
REF_LOAD=${REF_LOAD:-$SCR/ckpt/Qwen3-30B-A3B-Base_torch_dist}
mkdir -p "$RUN"

cd /root/slime
source "${MODEL_ARGS_FILE:-scripts/models/qwen3-30B-A3B.sh}"

# Parallelism: the example uses TP4/EP8 on 8 GPUs. Scale EP with the GPU count (128 experts divide 1..8).
TP=$(( NGPUS < 4 ? NGPUS : 4 ))
EP=$NGPUS
PERF_ARGS=(
   --tensor-model-parallel-size "$TP" --sequence-parallel
   --pipeline-model-parallel-size 1 --context-parallel-size 1
   --expert-model-parallel-size "$EP" --expert-tensor-parallel-size 1
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-20480}"
)
CKPT_ARGS=(
   --hf-checkpoint "$HF_CKPT" --ref-load "$REF_LOAD"
   --load "$RUN/ckpt" --save "$RUN/ckpt" --save-interval "${SAVE_INTERVAL:-20}"
)
OPTIMIZER_ARGS=(
   --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
)
MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
)
FREEZE_ARGS=()
if [ "${FREEZE_ROUTER:-0}" = 1 ]; then
   FREEZE_ARGS=(--freeze-params-name-list 'mlp\.router\.')
fi

if [ "$MODE" = sft ]; then
   MODE_ARGS=(
      --actor-num-nodes 1 --actor-num-gpus-per-node "$NGPUS"
      --rollout-function-path slime.rollout.sft_rollout.generate_rollout
      --prompt-data "${PROMPT_DATA:?set PROMPT_DATA to a parquet/jsonl with a messages column}" --input-key messages
      --rollout-shuffle --num-epoch "${NUM_EPOCH:-1}" ${NUM_ROLLOUT:+--num-rollout "$NUM_ROLLOUT"}
      --rollout-batch-size "${ROLLOUT_BS:-128}" --global-batch-size "${GLOBAL_BS:-128}"
      --loss-type sft_loss --calculate-per-token-loss
      --disable-compute-advantages-and-returns --debug-train-only
   )
elif [ "$MODE" = grpo ]; then
   MODE_ARGS=(
      --actor-num-nodes 1 --actor-num-gpus-per-node "$NGPUS" --colocate
      --prompt-data "${PROMPT_DATA:-$SCR/data/dapo-math-17k/dapo-math-17k.jsonl}"
      --input-key prompt --label-key label --apply-chat-template --rollout-shuffle
      --rm-type "${RM_TYPE:-deepscaler}"   # deepscaler scores 0 without </think>; base models: RM_TYPE=math/dapo/boxed_math
      --num-rollout "${NUM_ROLLOUT:-3000}" --rollout-batch-size "${ROLLOUT_BS:-32}"
      --n-samples-per-prompt "${N_SAMPLES:-8}" --rollout-max-response-len "${MAX_RESP_LEN:-8192}"
      --rollout-temperature 1 --global-batch-size "${GLOBAL_BS:-256}" --balance-data
      --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
      --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
      --rollout-num-gpus-per-engine "$NGPUS" --sglang-mem-fraction-static 0.7
   )
else
   echo "unknown mode $MODE"; exit 1
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
HAS_NVLINK=$(( NVLINK_COUNT > 0 ? 1 : 0 ))
echo "MODE=$MODE NGPUS=$NGPUS TP=$TP EP=$EP HAS_NVLINK=$HAS_NVLINK FREEZE_ROUTER=${FREEZE_ROUTER:-0} RUN=$RUN"

TRAIN_ARGS=(
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${MODE_ARGS[@]}" "${OPTIMIZER_ARGS[@]}"
    "${PERF_ARGS[@]}" "${MISC_ARGS[@]}" "${FREEZE_ARGS[@]}" "$@"
)
printf '%s\n' "${TRAIN_ARGS[@]}" > "$RUN/train_args_$(date +%Y%m%d_%H%M%S).txt"

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "=== DRY_RUN: parsing and validating ${#TRAIN_ARGS[@]} arguments (no Ray, no GPU)"
    # Default dump location: the project's logs/ (readable without scratch access).
    export DRY_RUN_ARGS_JSON=${DRY_RUN_ARGS_JSON:-$(cd "$LIB_DIR/../.." && pwd)/logs/resolved_args_dryrun_$(basename "$RUN").json}
    PYTHONPATH="/root/Megatron-LM/:$LIB_DIR" python3 - "${TRAIN_ARGS[@]}" <<'PYEOF'
import sys
import torch
# Megatron's validate_args queries the GPU's compute capability; on a CPU node pretend sm_90 (H100/H200).
if not torch.cuda.is_available():
    torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
from slime.utils.arguments import parse_args
args = parse_args()
keys = ["num_epoch", "num_rollout", "rollout_batch_size", "global_batch_size", "micro_batch_size",
        "lr", "min_lr", "lr_decay_style", "lr_warmup_fraction", "lr_warmup_iters", "weight_decay",
        "adam_beta1", "adam_beta2", "adam_eps", "clip_grad", "optimizer", "loss_type", "loss_mask_type",
        "calculate_per_token_loss", "rollout_shuffle", "rollout_seed", "seed", "max_tokens_per_gpu",
        "tensor_model_parallel_size", "expert_model_parallel_size", "sequence_parallel", "bf16",
        "save", "load", "ref_load", "hf_checkpoint", "save_hf", "no_save_optim", "custom_megatron_init_path",
        "freeze_params_name_list", "seq_length", "max_position_embeddings"]
for k in keys:
    print(f"  {k} = {getattr(args, k, '<unset>')}")
import json, os
out = os.environ.get("DRY_RUN_ARGS_JSON")
if out:
    with open(out, "w") as f:
        json.dump(vars(args), f, indent=1, sort_keys=True, default=str)
    print(f"full resolved arguments ({len(vars(args))} incl. defaults) -> {out}")
print("DRY_RUN OK")
PYEOF
    exit 0
fi

RUNTIME_ENV_JSON=$(LIB_DIR="$LIB_DIR" HAS_NVLINK="$HAS_NVLINK" python3 -c '
import json, os
env = {"PYTHONPATH": "/root/Megatron-LM/:" + os.environ["LIB_DIR"],
       "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": os.environ["HAS_NVLINK"]}
for name in os.environ.get("PASS_ENV", "").split():
    if name in os.environ:
        env[name] = os.environ[name]
print(json.dumps({"env_vars": env}))')
echo "runtime env: $RUNTIME_ENV_JSON"

RAY_PORT=$((20000 + SLURM_JOB_ID % 5000))
DASH_PORT=$((RAY_PORT + 1))
export no_proxy=127.0.0.1
ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" --dashboard-port "$DASH_PORT" \
    --num-gpus "$NGPUS" --num-cpus "${RAY_CPUS:-${SLURM_CPUS_PER_TASK:-8}}" \
    --object-store-memory $(( ${RAY_OBJ_STORE_GB:-64} * 1000000000 )) \
    --disable-usage-stats --temp-dir "/tmp/ray_${SLURM_JOB_ID}"
trap 'ray stop --force || true' EXIT

ray job submit --address="http://127.0.0.1:$DASH_PORT" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 "${TRAIN_ENTRY:-train.py}" "${TRAIN_ARGS[@]}"
