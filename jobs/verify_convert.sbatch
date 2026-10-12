#!/bin/bash
#SBATCH --job-name=verify_convert
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH --output=logs/%x_%j.out
# Verification 4: Qwen3-0.6B  HF -> torch_dist -> HF, then compare tensors and logits.
set -euo pipefail
source jobs/slime_common.sh
echo "host: $(hostname)"

WORK=$SCR/verify/convert_qwen3_0.6B
HF=$SCR/models/Qwen3-0.6B
mkdir -p "$WORK"
rm -rf "$WORK/torch_dist" "$WORK/hf_roundtrip"   # outputs of a previous run of this test only

slime_exec bash -c "
set -euo pipefail
python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='$HF')\"
source /root/slime/scripts/models/qwen3-0.6B.sh
cd /root/slime
echo '=== HF -> torch_dist'
python tools/convert_hf_to_torch_dist.py \${MODEL_ARGS[@]} --hf-checkpoint $HF --save $WORK/torch_dist
ls $WORK/torch_dist $WORK/torch_dist/release | head
echo '=== torch_dist -> HF'
python tools/convert_torch_dist_to_hf.py --input-dir $WORK/torch_dist/release \
    --output-dir $WORK/hf_roundtrip --origin-hf-dir $HF --vocab-size 151936
ls -la $WORK/hf_roundtrip
echo '=== compare'
python $PWD/jobs/lib/compare_hf.py $HF $WORK/hf_roundtrip --logits
"
du -sh "$WORK"/*
