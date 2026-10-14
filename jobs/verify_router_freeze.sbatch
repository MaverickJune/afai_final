#!/bin/bash
#SBATCH --job-name=verify_router_freeze
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out
# Verifications 4 (MoE) + 5: tiny random Qwen3-MoE, HF<->torch_dist round trip,
# then slime SFT with and without --freeze-params-name-list 'mlp\.router\.'.
set -euo pipefail
source jobs/slime_common.sh
echo "host: $(hostname)"

WORK=$SCR/verify/router_freeze
TOK=$SCR/models/Qwen3-0.6B
rm -rf "$WORK"          # outputs of a previous run of this test only
mkdir -p "$WORK"

slime_exec bash -c "
set -euo pipefail
python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='$TOK')\"
python $PWD/jobs/lib/make_tiny_moe.py $WORK $TOK
"
slime_exec bash jobs/lib/router_freeze_test.sh "$WORK" "$PWD"
du -sh "$WORK"
