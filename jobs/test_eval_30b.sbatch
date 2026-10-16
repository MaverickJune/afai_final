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
#   2) probe_analysis (KL + routing) with base + final SFT checkpoint on one GPU (memory, router hooks on all 48
#      layers), 3 sequences per probe set
# PROBE_ONLY=1 sbatch ... skips part 1.
set -euo pipefail
set -m   # background jobs get their own process group, so the whole SGLang server tree can be stopped
source jobs/slime_common.sh
source jobs/configs/eval.sh
SFT_FINAL=$(ls -d "$SFT_RUN"/hf_ckpts/step_* | sort | tail -1)
OUT=$EVAL_ROOT/test/30b
rm -rf "$OUT"   # outputs of a previous run of this test only
mkdir -p "$OUT"
echo "host: $(hostname)  start: $(date)  model: $SFT_FINAL"
ls "$SFT_FINAL"

if [ "${PROBE_ONLY:-0}" != 1 ]; then
echo "=== [1] benchmarks (3 questions each, <= 2048 tokens)"
PORT=$((30000 + SLURM_JOB_ID % 1000))
slime_exec python -m sglang.launch_server --model-path "$SFT_FINAL" --tokenizer-path "$BASE_HF" \
    --host 127.0.0.1 --port "$PORT" --tp-size 1 --mem-fraction-static 0.85 --random-seed 0 \
    > "$OUT/sglang_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill -- -$SERVER_PID 2>/dev/null || true' EXIT
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
kill -- -$SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
for i in $(seq 1 24); do   # wait until the server's GPU memory is released
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && break; sleep 5
done
nvidia-smi --query-gpu=memory.used --format=csv
fi

echo "=== [2] probes: base + final SFT checkpoint on one GPU, 3 sequences per probe"
set +u; activate_afai_env; set -u
python jobs/lib/eval/probe_analysis.py --ref "$BASE_HF" --probe-dir "$BENCH_DIR/probes" --out "$OUT/probe" --limit 3 \
    --ckpt sft_final="$SFT_FINAL"
python -c "import torch; print('peak GPU memory', torch.cuda.max_memory_allocated() / 1e9)" || true
head -5 "$OUT/probe/routing.csv"
echo "EVAL 30B TEST: finished (check the numbers above)"
echo "end: $(date)"
