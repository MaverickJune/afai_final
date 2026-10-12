# Source from job scripts: `source jobs/slime_common.sh`, then `slime_exec <cmd...>`.
# Runs <cmd> inside the pinned slime image with GPUs, home and group scratch visible.

SLIME_SIF=${SLIME_SIF:-$HOME/containers/slime_v0.4.0-cu129.sif}
SCR=$HOME/scratch_pi_lz484/wb379
export HF_HOME=$SCR/hf_cache
mkdir -p "$HF_HOME"

slime_exec() {
    # --cleanenv: keep host module env (LD_LIBRARY_PATH etc.) out of the container.
    # $HOME is bound by default; ~/scratch_pi_lz484 is a symlink into /nfs/roberts/scratch.
    apptainer exec --nv --cleanenv --writable-tmpfs \
        --bind /nfs/roberts/scratch \
        --env HF_HOME="$HF_HOME" \
        --env PYTHONNOUSERSITE=1 \
        --env PYTHONUNBUFFERED=1 \
        --env PYTHONPATH=/root/Megatron-LM \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
        --env SLURM_JOB_ID="${SLURM_JOB_ID:-0}" \
        "$SLIME_SIF" "$@"
}
