"""Wrapper around slime's tools/convert_torch_dist_to_hf.py for the Megatron in the slime v0.4.0 image.

Megatron 4b4acac stores the checkpoint's common (non-tensor) state, including `args`, as a
"common_state" ShardedObject inside the .distcp files instead of a separate common.pt.
slime's converter (v0.4.0 and main as of 2026-10-12) still does
`torch.load(<input_dir>/common.pt)["args"]`, which fails with FileNotFoundError.

This wrapper makes that one torch.load fall back to the embedded common_state when common.pt
is missing, then runs the unmodified slime tool. Same CLI as the original:

    python jobs/lib/convert_torch_dist_to_hf.py --input-dir CKPT/iter_xxxxxxx \
        --output-dir OUT_HF --origin-hf-dir ORIG_HF [--vocab-size N]
"""

import io
import os
import runpy
import sys

import torch
import torch.distributed.checkpoint as dist_cp

TOOL = "/root/slime/tools/convert_torch_dist_to_hf.py"
_torch_load = torch.load


def _load_embedded_common_state(ckpt_dir):
    # Mirrors megatron.core.dist_checkpointing.serialization.load_common_state_dict (current format).
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    key = ShardedObject("common_state", None, (1,), (0,)).unique_key
    state = {key: io.BytesIO()}
    dist_cp.load(state, storage_reader=dist_cp.FileSystemReader(ckpt_dir), no_dist=True)
    loaded = state[key]
    if isinstance(loaded, io.BytesIO):
        loaded.seek(0)
        loaded = _torch_load(loaded, weights_only=False)
    return loaded[0]


def _patched_load(f, *args, **kwargs):
    if isinstance(f, (str, os.PathLike)) and os.path.basename(f) == "common.pt" and not os.path.exists(f):
        print(f"[wrapper] {f} not found; reading embedded common_state instead")
        return _load_embedded_common_state(os.path.dirname(f))
    return _torch_load(f, *args, **kwargs)


torch.load = _patched_load
sys.argv[0] = TOOL
runpy.run_path(TOOL, run_name="__main__")
