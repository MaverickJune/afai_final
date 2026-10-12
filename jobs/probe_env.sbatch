#!/bin/bash
#SBATCH --job-name=probe_env
#SBATCH --partition=day
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 0:05:00
#SBATCH --output=logs/%x_%j.out
# Read-only probe of the host environment on a compute node. Installs nothing.

echo "host: $(hostname)"
echo "--- apptainer"
which apptainer singularity 2>&1
apptainer --version 2>&1
echo "--- home / scratch"
ls -la ~ 2>&1
ls -ld ~/scratch_pi_lz484 ~/scratch_pi_lz484/wb379 ~/project_pi_lz484 ~/containers ~/envs ~/.conda ~/.conda/envs 2>&1
readlink -f ~/scratch_pi_lz484 ~/.conda/envs 2>&1
ls -la ~/scratch_pi_lz484/wb379 2>&1 | head -30
echo "--- quotas"
getquota 2>&1 || quota -s 2>&1
df -h ~ ~/scratch_pi_lz484/wb379 /tmp 2>&1
echo "--- /nfs/roberts"
ls /nfs/roberts 2>&1 | head
echo "--- miniconda module"
module reset >/dev/null 2>&1
module load miniconda 2>&1
which conda; conda --version
conda config --show envs_dirs pkgs_dirs 2>&1
echo "--- network"
curl -sS -o /dev/null -w "dockerhub %{http_code}\n" https://registry-1.docker.io/v2/ 2>&1
curl -sS -o /dev/null -w "pypi %{http_code}\n" https://pypi.org/simple/ 2>&1
curl -sS -o /dev/null -w "hf %{http_code}\n" https://huggingface.co 2>&1
