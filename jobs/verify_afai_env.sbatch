#!/bin/bash
#SBATCH --job-name=verify_afai_env
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0:15:00
#SBATCH --output=logs/%x_%j.out
# afai_env on GPU: arch list, and load the slime-converted HF checkpoint (from verify_convert)
# plus the converted tiny MoE (from verify_router_freeze) with this env's transformers.
set -eo pipefail
module reset && module load miniconda && conda activate ~/envs/afai_env
export PYTHONNOUSERSITE=1
SCR=${SCR:-$HOME/scratch_pi_lz484/$USER}
export HF_HOME=$SCR/hf_cache
nvidia-smi --query-gpu=name,driver_version --format=csv

python - "$SCR/verify/convert_qwen3_0.6B/hf_roundtrip" "$SCR/verify/router_freeze/sft_frozen_hf" <<'EOF'
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
archs = torch.cuda.get_arch_list()
print("arch list", archs, "| sm_90:", "sm_90" in archs, "| sm_100:", "sm_100" in archs)
for d in sys.argv[1:]:
    if not os.path.isdir(d):
        print("skip (missing):", d); continue
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    out = m.generate(ids, max_new_tokens=8, do_sample=False)
    print(f"{d}: {type(m).__name__} -> {tok.decode(out[0])!r}")
EOF
