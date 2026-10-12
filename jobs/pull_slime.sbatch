#!/bin/bash
#SBATCH --job-name=pull_slime
#SBATCH --partition=day
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH --output=logs/%x_%j.out
# Pull the official slime image (pinned by digest) into ~/containers as a .sif,
# then run CPU-only sanity checks on its contents.
set -euo pipefail

TAG=v0.4.0-cu129
DIGEST=sha256:626fbff9c062bc3723ee617a6930ff96913750f6847a8df9e2319d84f8169c45
SIF=$HOME/containers/slime_${TAG}.sif

SCR=$HOME/scratch_pi_lz484/wb379
export APPTAINER_CACHEDIR=$SCR/apptainer/cache
export APPTAINER_TMPDIR=$SCR/apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HOME/containers"

echo "host: $(hostname)  start: $(date)"
if [ -f "$SIF" ]; then
    echo "$SIF already exists, skipping pull"
else
    apptainer pull "$SIF" "docker://slimerl/slime@${DIGEST}"
fi
echo "pull done: $(date)"
ls -lh "$SIF"
sha256sum "$SIF"

echo "=== contents check (as normal user) ==="
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 "$SIF" bash -c '
id
ls -ld /root /root/slime /root/Megatron-LM /sgl-workspace/sglang 2>&1
echo "slime commit:    $(git -c safe.directory="*" -C /root/slime rev-parse HEAD 2>&1)"
echo "megatron commit: $(git -c safe.directory="*" -C /root/Megatron-LM rev-parse HEAD 2>&1)"
echo "sglang commit:   $(git -c safe.directory="*" -C /sgl-workspace/sglang rev-parse HEAD 2>&1)"
which python; python --version
python - <<EOF
import importlib
for m in ["torch", "transformer_engine", "megatron.core", "sglang", "slime", "ray", "transformers", "flash_attn"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:20s} OK  {getattr(mod, \"__version__\", \"?\")}  {getattr(mod, \"__file__\", \"\")}")
    except Exception as e:
        print(f"{m:20s} FAIL {type(e).__name__}: {e}")
import torch
print("torch.version.cuda", torch.version.cuda)
print("arch list", torch.cuda.get_arch_list())
EOF
'

echo "=== contents check (--fakeroot) ==="
apptainer exec --fakeroot --cleanenv "$SIF" bash -c 'id; ls -ld /root/slime /root/Megatron-LM' 2>&1 || echo "fakeroot unavailable"
echo "end: $(date)"
