#!/bin/bash
#SBATCH --job-name=verify_sglang
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out
# Verification 2: serve Qwen3-0.6B with SGLang, get one completion.
set -euo pipefail
source jobs/slime_common.sh
echo "host: $(hostname)"

MODEL=Qwen/Qwen3-0.6B
PORT=$((30000 + SLURM_JOB_ID % 1000))

slime_exec python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --mem-fraction-static 0.5 > "logs/sglang_server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "server healthy after $((i*5))s"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died"; tail -50 "logs/sglang_server_${SLURM_JOB_ID}.log"; exit 1; fi
    sleep 5
done

curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Answer with just the number.\"}],
  \"max_tokens\": 256, \"temperature\": 0,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}"
echo
curl -s "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}'
echo
nvidia-smi --query-gpu=name,memory.used --format=csv
echo "--- server log tail"; tail -20 "logs/sglang_server_${SLURM_JOB_ID}.log"
