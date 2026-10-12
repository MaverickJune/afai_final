#!/bin/bash
#SBATCH --job-name=build_afai_env
#SBATCH --partition=day
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=logs/%x_%j.out
# Build the lightweight analysis env by prefix (~/.conda/envs points at the full project dir).
# CUDA comes from pip wheels; do NOT module load CUDA.
set -eo pipefail

ENV=$HOME/envs/afai_env
SCR=$HOME/scratch_pi_lz484/wb379
export PIP_CACHE_DIR=$SCR/pip_cache
export CONDA_PKGS_DIRS=$SCR/conda_pkgs
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HOME/envs"

module reset && module load miniconda
echo "host: $(hostname)  start: $(date)"

if [ ! -d "$ENV" ]; then
    conda create -y --prefix "$ENV" -c conda-forge --override-channels python=3.12 pip
fi
conda activate "$ENV"
export PYTHONNOUSERSITE=1
which python pip

# cu128 wheels include sm_90 (H100/H200) and sm_100 (B200)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers safetensors accelerate numpy pandas matplotlib

pip freeze > logs/afai_env_freeze.txt
python - <<'EOF'
import torch, transformers, safetensors, accelerate, numpy, pandas, matplotlib
for m in [torch, transformers, safetensors, accelerate, numpy, pandas, matplotlib]:
    print(f"{m.__name__:14s} {m.__version__}")
print("torch.version.cuda", torch.version.cuda)
print("arch list", torch.cuda.get_arch_list())
EOF
du -sh "$ENV"
echo "end: $(date)"
