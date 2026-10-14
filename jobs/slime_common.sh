# Source from job scripts: `source jobs/slime_common.sh`, then `slime_exec <cmd...>`.
# Runs <cmd> inside the pinned slime image with GPUs, home and group scratch visible.

SLIME_SIF=${SLIME_SIF:-$HOME/containers/slime_v0.4.0-cu129.sif}
SCR=${SCR:-$HOME/scratch_pi_lz484/$USER}
export HF_HOME=$SCR/hf_cache
mkdir -p "$HF_HOME"

# The image ships an old system cuDNN (9.10.2, /usr/lib/x86_64-linux-gnu) alongside torch's pip cuDNN (9.17.1).
# Once TE etc. are imported the system one can win, and torch then fails its cuDNN version check.
# Put the pip cuDNN first; the rest is the image's own LD_LIBRARY_PATH (verify_container job 29247877).
SLIME_LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs

slime_exec() {
    # --cleanenv: keep host module env (LD_LIBRARY_PATH etc.) out of the container.
    # $HOME is bound by default; ~/scratch_pi_lz484 is a symlink into /nfs/roberts/scratch.
    apptainer exec --nv --cleanenv --writable-tmpfs \
        --bind /nfs/roberts/scratch \
        --env HF_HOME="$HF_HOME" \
        --env SCR="$SCR" \
        --env PYTHONNOUSERSITE=1 \
        --env PYTHONUNBUFFERED=1 \
        --env PYTHONPATH=/root/Megatron-LM \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        --env LD_LIBRARY_PATH="$SLIME_LD_LIBRARY_PATH" \
        --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
        --env SLURM_JOB_ID="${SLURM_JOB_ID:-0}" \
        "$SLIME_SIF" "$@"
}
