#!/bin/bash
#SBATCH --job-name=download_30b
#SBATCH --partition=day
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --output=logs/%x_%j.out
# Download Qwen3-30B-A3B-Base (~61 GB) and the GRPO datasets used by slime's example to scratch.
# CPU-only on purpose (GPU jobs that don't use the GPU are auto-cancelled).
# 16G was OOM-killed within 2 min (job 29263670; hf_xet buffers heavily with 8 workers).
set -eo pipefail
module reset && module load miniconda && conda activate ~/envs/afai_env
export PYTHONNOUSERSITE=1
SCR=$HOME/scratch_pi_lz484/wb379
export HF_HOME=$SCR/hf_cache
echo "host: $(hostname)  start: $(date)"

python - "$SCR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
scr = sys.argv[1]
for repo, kind, dest in [
    ("Qwen/Qwen3-30B-A3B-Base", "model", f"{scr}/models/Qwen3-30B-A3B-Base"),
    ("zhuzilin/dapo-math-17k", "dataset", f"{scr}/data/dapo-math-17k"),
    ("zhuzilin/aime-2024", "dataset", f"{scr}/data/aime-2024"),
]:
    path = snapshot_download(repo, repo_type=kind, local_dir=dest, max_workers=4)
    print("done:", repo, "->", path, flush=True)
EOF
du -sh "$SCR"/models/Qwen3-30B-A3B-Base "$SCR"/data/*
ls -la "$SCR"/models/Qwen3-30B-A3B-Base
echo "end: $(date)"
