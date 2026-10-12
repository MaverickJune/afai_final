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
# Layer extraction (many small files) is extremely slow on NFS scratch (stalled >50 min in job 29229347),
# so unpack on node-local disk. Still never in home. Removed by the trap on exit.
export APPTAINER_TMPDIR=/tmp/apptainer_tmp_${SLURM_JOB_ID}
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HOME/containers"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT
df -h /tmp

echo "host: $(hostname)  start: $(date)"

# One-off cleanup (user-approved): partial unpack left by the cancelled NFS-tmpdir pull, job 29229347.
OLD_TMP=$SCR/apptainer/tmp
if [ -d "$OLD_TMP" ]; then
    echo "removing stale apptainer temp dirs:"; ls -la "$OLD_TMP"
    for d in "$OLD_TMP"/build-temp-*; do
        [ -d "$d" ] || continue
        chmod -R u+rwX "$d" 2>/dev/null || true
        rm -rf "$d"
    done
    rmdir "$OLD_TMP" 2>/dev/null || true
fi

if [ -f "$SIF" ] && apptainer sif list "$SIF" >/dev/null 2>&1; then
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
'
# Imports, versions and GPU checks: see jobs/verify_container.sbatch
echo "end: $(date)"
