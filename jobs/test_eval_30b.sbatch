#!/bin/bash
#SBATCH --job-name=test_eval_30b
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=192G
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out
# Mechanics test on the real 30B models, 1 H200 (not an accuracy measurement):
#   1) SGLang serves the final SFT checkpoint; run_bench on 3 questions of every benchmark, responses capped at
#      2048 tokens -> summaries written, prompts/stop tokens/grading visible in the log
#   2) kl_math500 with base + final SFT checkpoint on one GPU (memory check), 3 problems
set -euo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
SFT_FINAL=$(ls -d "$SFT_RUN"/hf_ckpts/step_* | sort | tail -1)
OUT=$EVAL_ROOT/test/30b
rm -rf "$OUT"   # outputs of a previous run of this test only
mkdir -p "$OUT"
echo "host: $(hostname)  start: $(date)  model: $SFT_FINAL"
ls "$SFT_FINAL"

echo "=== [1] benchmarks (3 questions each, <= 2048 tokens)"
PORT=$((30000 + SLURM_JOB_ID % 1000))
slime_exec python -m sglang.launch_server --model-path "$SFT_FINAL" --tokenizer-path "$BASE_HF" \
    --host 127.0.0.1 --port "$PORT" --tp-size 1 --mem-fraction-static 0.85 --random-seed 0 \
    > "$OUT/sglang_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "server healthy after $((i*5))s"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died"; tail -50 "$OUT/sglang_server.log"; exit 1; fi
    sleep 5
done
grep -m1 "max_total_num_tokens" "$OUT/sglang_server.log" || true
slime_exec python jobs/lib/eval/run_bench.py --server "http://127.0.0.1:$PORT" --tokenizer "$BASE_HF" \
    --model-dir "$SFT_FINAL" --bench-dir "$BENCH_DIR" --out "$OUT/bench" --limit 3 --max-new-cap 2048
for f in "$OUT"/bench/*.samples.jsonl; do
    echo "--- $(basename "$f"): first sample (first 600 chars of the response, tail 300)"
    head -1 "$f" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); r=d.pop('response'); print(d); print(r[:600]); print('[...]', r[-300:])"
done
kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
sleep 10; nvidia-smi --query-gpu=memory.used --format=csv

echo "=== [2] KL: base + final SFT checkpoint on one GPU, 3 problems"
set +u; activate_afai_env; set -u
python jobs/lib/eval/kl_math500.py --base "$BASE_HF" --data "$BENCH_DIR/math500.jsonl" --out "$OUT/kl" --limit 3 \
    --ckpt sft_final="$SFT_FINAL"
echo "EVAL 30B TEST: finished (check the numbers above)"
echo "end: $(date)"
