#!/bin/bash
#SBATCH --job-name=convert_30b
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 3:00:00
#SBATCH --output=logs/%x_%j.out
# Qwen3-30B-A3B-Base: HF -> torch_dist (needed by slime as --ref-load / --load), then torch_dist -> HF
# round-trip check (tensors bit-identical + logits). Model is ~61 GB bf16, fits on one H200.
set -euo pipefail
source jobs/slime_common.sh
echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

HF=$SCR/models/Qwen3-30B-A3B-Base
OUT=$SCR/ckpt/Qwen3-30B-A3B-Base_torch_dist
RT=$SCR/verify/Qwen3-30B-A3B-Base_hf_roundtrip
rm -rf "$OUT" "$RT"   # outputs of a previous run of this job only
mkdir -p "$SCR/ckpt"

slime_exec bash -c "
set -euo pipefail
source /root/slime/scripts/models/qwen3-30B-A3B.sh
cd /root/slime
echo '=== HF -> torch_dist'
python tools/convert_hf_to_torch_dist.py \${MODEL_ARGS[@]} --hf-checkpoint $HF --save $OUT
ls -la $OUT $OUT/release
echo '=== torch_dist -> HF (round trip)'
python $PWD/jobs/lib/convert_torch_dist_to_hf.py --input-dir $OUT/release \
    --output-dir $RT --origin-hf-dir $HF --vocab-size 151936
echo '=== compare'
python $PWD/jobs/lib/compare_hf.py $HF $RT --logits
"
du -sh "$OUT" "$RT"
echo "end: $(date)"
