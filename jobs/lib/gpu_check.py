"""GPU sanity checks run inside the slime container (see jobs/verify_container.sbatch)."""

import importlib, subprocess
for m in ["torch", "transformer_engine", "megatron.core", "sglang", "slime", "ray", "transformers", "flash_attn"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:20s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:20s} FAIL {type(e).__name__}: {e}")
for repo in ["/root/slime", "/root/Megatron-LM", "/sgl-workspace/sglang"]:
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True)
    print(f"{repo:24s} {r.stdout.strip() or r.stderr.strip()}")

import torch
print("torch.version.cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())  # raises if the image's old system cuDNN got loaded
print("cuda available", torch.cuda.is_available(), "| devices", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
archs = torch.cuda.get_arch_list()
print("arch list", archs)
print("sm_90 in list:", "sm_90" in archs, "| sm_100 in list:", "sm_100" in archs)

# real GPU work: bf16 matmul, TE linear, flash-attn
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
import time; t = time.time()
for _ in range(20):
    b = a @ a
torch.cuda.synchronize()
print(f"bf16 matmul TFLOP/s ~ {20 * 2 * 8192**3 / (time.time() - t) / 1e12:.0f}")

import transformer_engine.pytorch as te
lin = te.Linear(4096, 4096, params_dtype=torch.bfloat16).cuda()
print("TE linear out", lin(torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)).shape)

from flash_attn import flash_attn_func
q = torch.randn(2, 1024, 16, 128, device="cuda", dtype=torch.bfloat16)
print("flash_attn out", flash_attn_func(q, q, q, causal=True).shape)
x = torch.randn(8, 64, 128, 128, device="cuda", dtype=torch.bfloat16)
conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda().to(torch.bfloat16)
print("cudnn conv2d out", conv(x).shape)
print("max mem GB", torch.cuda.max_memory_allocated() / 1e9)
