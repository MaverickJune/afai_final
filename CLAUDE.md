# Cluster rules (Yale Bouchet)

## Where you are
- You run on a 1-CPU, 8GB `agent` node. Never run heavy compute, large builds, or GPU work here.
- Never use `salloc`. It opens an interactive shell and will hang.

## Running work
- Quick GPU test (< 30 min): `srun -p gpu_devel --gpus=h200:1 -c 8 --mem=64G -t 0:30:00 <cmd>`
- Experiments: write a script in `jobs/`, submit with `sbatch`, monitor with `squeue --me`, read logs in `logs/`.
- Training/rollout jobs (slime, Megatron, SGLang) run inside the Apptainer image: `source jobs/slime_common.sh` then `slime_exec <cmd>` (see SETUP.md). Do not load conda in these jobs.
- Analysis/eval jobs: `module reset && module load miniconda && conda activate ~/envs/afai_env` (prefix env; never `conda create -n`, since ~/.conda/envs points at the full project dir). Never `module load CUDA` with conda.
- Never copy the `pkill -9 python/ray/sglang` preamble from slime's example scripts (shared nodes).
- Use `#SBATCH --output=logs/%x_%j.out`. No space between `#` and `SBATCH`.
- Ask before running `scancel`.

## Storage
- Code lives here (home, backed up). Commit to git often.
- Large files (model weights, datasets, checkpoints) go in `~/scratch_pi_lz484/wb379/`. Files there are deleted after 30 days.
- Do NOT write to `~/project_pi_lz484/` (group quota is full).
- HuggingFace cache: `HF_HOME=~/scratch_pi_lz484/wb379/hf_cache`
