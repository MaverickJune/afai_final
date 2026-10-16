#!/bin/bash
#SBATCH --job-name=eval_bench
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH --output=logs/%x_%j.out
# Benchmarks for ONE model on one H200: SGLang server (bf16, TP 1) + jobs/lib/eval/run_bench.py, inside the container.
# usage: sbatch jobs/eval_bench.sbatch NAME [MODEL_DIR] [BENCHMARKS (comma list)] [LIMIT] [MAX_NEW_CAP]
#   NAME base | sft_final | grpo_final resolve MODEL_DIR from jobs/configs/eval.sh (pass "" as MODEL_DIR);
#   any other NAME needs MODEL_DIR. e.g.  sbatch jobs/eval_bench.sbatch sft_final
# Output: $EVAL_ROOT/bench/NAME/ (samples + summaries); summaries copied to experiments/eval/results/NAME/.
# Resumable: submit the same command again; finished benchmarks are skipped, unfinished ones continue.
set -euo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
NAME=$1; MODEL_DIR=${2:-}; BENCHES=${3:-}; LIMIT=${4:-0}; CAP=${5:-0}
if [ -z "$MODEL_DIR" ]; then
    case $NAME in
        base) MODEL_DIR=$BASE_HF ;;
        sft_final) MODEL_DIR=$(ls -d "$SFT_RUN"/hf_ckpts/step_* | sort | tail -1) ;;
        grpo_final) MODEL_DIR=$(ls -d "$GRPO_RUN"/hf_ckpts/step_* | sort | tail -1) ;;
        *) echo "NAME $NAME needs MODEL_DIR"; exit 1 ;;
    esac
fi
[ -f "$MODEL_DIR/config.json" ] || { echo "no config.json in $MODEL_DIR"; exit 1; }
OUT=$EVAL_ROOT/bench/$NAME
mkdir -p "$OUT"
echo "host: $(hostname)  start: $(date)  job: $SLURM_JOB_ID  model: $MODEL_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
ls "$MODEL_DIR" | head -30

PORT=$((30000 + SLURM_JOB_ID % 1000))
# Tokenizer from the base model for every model (identical prompt tokens); weights from MODEL_DIR.
slime_exec python -m sglang.launch_server --model-path "$MODEL_DIR" --tokenizer-path "$BASE_HF" \
    --host 127.0.0.1 --port "$PORT" --tp-size 1 --mem-fraction-static 0.85 --random-seed 0 \
    > "$OUT/sglang_server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "server healthy after $((i*5))s"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died"; tail -50 "$OUT/sglang_server_${SLURM_JOB_ID}.log"; exit 1; fi
    sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server not healthy after 20 min"; exit 1; }
grep -m1 "max_total_num_tokens" "$OUT/sglang_server_${SLURM_JOB_ID}.log" || true

slime_exec python jobs/lib/eval/run_bench.py --server "http://127.0.0.1:$PORT" --tokenizer "$BASE_HF" \
    --model-dir "$MODEL_DIR" --bench-dir "$BENCH_DIR" --out "$OUT" --limit "$LIMIT" --max-new-cap "$CAP" \
    ${BENCHES:+--benchmarks "$BENCHES"}

mkdir -p "experiments/eval/results/$NAME"
cp "$OUT"/*.summary.json "experiments/eval/results/$NAME/" 2>/dev/null || true
echo "end: $(date)"
