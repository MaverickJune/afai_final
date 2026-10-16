"""Parameter-level change of fine-tuned checkpoints vs the base model ("Scalpel vs. Hammer" Sec. 4.2, extended to MoE).

For every weight tensor W and checkpoint C:  d = ||W_C - W_base||_F  (computed in fp32), b = ||W_base||_F, numel.
Reported:  rel = d / b   (relative change; the paper's "normalized Frobenius norm")
           rms = d / sqrt(numel)   (root-mean-square change per weight entry)
Tensors are streamed shard by shard (one base shard in memory at a time). CPU only; runs in afai_env.

usage: python param_diff.py --base BASE_HF_DIR --ckpt LABEL=DIR [--ckpt LABEL=DIR ...] --out OUT_DIR
writes: OUT_DIR/param_diff_tensors.csv.gz   one row per (checkpoint, tensor): d, b, numel, layer, module, expert
        OUT_DIR/param_diff_modules.csv      per (checkpoint, layer, module): experts pooled as
                                            rel = sqrt(sum d^2) / sqrt(sum b^2), rms = sqrt(sum d^2 / sum numel)
"""

import argparse
import csv
import gzip
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

# Qwen3-MoE HF names. Experts are stored per expert (model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight).
PATTERNS = [
    (re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight$"), None),
    (re.compile(r"model\.layers\.(\d+)\.mlp\.gate\.weight$"), "router"),
    (re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"), "expert"),
    (re.compile(r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$"), None),
]


def classify(name):
    """(layer, module, expert) for a tensor name; layer -1 for non-layer tensors."""
    for pat, kind in PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        if kind == "router":
            return int(m.group(1)), "router", -1
        if kind == "expert":
            return int(m.group(1)), "experts." + m.group(3), int(m.group(2))
        return int(m.group(1)), m.group(2), -1
    return -1, name.replace("model.", "").replace(".weight", ""), -1


def weight_map(d):
    """tensor name -> shard file of an HF checkpoint dir (sharded or single-file)."""
    idx = Path(d) / "model.safetensors.index.json"
    if idx.exists():
        return {k: str(Path(d) / v) for k, v in json.loads(idx.read_text())["weight_map"].items()}
    files = sorted(Path(d).glob("*.safetensors"))
    out = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            out.update({k: str(f) for k in h.keys()})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--ckpt", action="append", required=True, help="LABEL=DIR")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    base_map = weight_map(a.base)
    ckpts = [(lbl, weight_map(d)) for lbl, d in (c.split("=", 1) for c in a.ckpt)]
    for lbl, wm in ckpts:
        missing, extra = set(base_map) - set(wm), set(wm) - set(base_map)
        print(f"{lbl}: {len(wm)} tensors; missing vs base {len(missing)} {sorted(missing)[:5]}; extra {len(extra)} {sorted(extra)[:5]}")
        assert not missing, f"{lbl} lacks base tensors"

    by_shard = defaultdict(list)
    for name, f in base_map.items():
        by_shard[f].append(name)
    rows = []
    t0 = time.time()
    for si, (shard, names) in enumerate(sorted(by_shard.items())):
        with safe_open(shard, "pt") as h:
            base = {n: h.get_tensor(n) for n in names}
        bnorm = {n: torch.linalg.vector_norm(t.float()).item() for n, t in base.items()}
        for lbl, wm in ckpts:
            need = defaultdict(list)
            for n in names:
                need[wm[n]].append(n)
            for f, ns in need.items():
                with safe_open(f, "pt") as h:
                    for n in ns:
                        c = h.get_tensor(n)
                        assert c.shape == base[n].shape, f"{lbl} {n}: shape {tuple(c.shape)} vs {tuple(base[n].shape)}"
                        d = torch.linalg.vector_norm(c.float() - base[n].float()).item()
                        layer, module, expert = classify(n)
                        rows.append({"ckpt": lbl, "tensor": n, "layer": layer, "module": module, "expert": expert,
                                     "numel": base[n].numel(), "diff_fro": d, "base_fro": bnorm[n]})
        print(f"shard {si + 1}/{len(by_shard)} ({len(names)} tensors) done, {(time.time() - t0) / 60:.1f} min", flush=True)
        del base

    with gzip.open(out / "param_diff_tensors.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    agg = defaultdict(lambda: [0.0, 0.0, 0, 0])  # sum d^2, sum b^2, sum numel, n tensors
    for r in rows:
        g = agg[(r["ckpt"], r["layer"], r["module"])]
        g[0] += r["diff_fro"] ** 2
        g[1] += r["base_fro"] ** 2
        g[2] += r["numel"]
        g[3] += 1
    with open(out / "param_diff_modules.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "layer", "module", "n_tensors", "numel", "rel", "rms"])
        for (lbl, layer, module), (d2, b2, numel, nt) in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            w.writerow([lbl, layer, module, nt, numel, math.sqrt(d2) / math.sqrt(b2) if b2 > 0 else float("nan"),
                        math.sqrt(d2 / numel)])

    # Short report: per checkpoint and module type, pooled over layers.
    tot = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        t = tot[(r["ckpt"], r["module"] if r["layer"] >= 0 else "non-layer")]
        t[0] += r["diff_fro"] ** 2
        t[1] += r["base_fro"] ** 2
    for lbl, _ in ckpts:
        print(f"--- {lbl}: relative change pooled over layers")
        for (l2, mod), (d2, b2) in sorted(tot.items()):
            if l2 == lbl:
                print(f"  {mod:28s} {math.sqrt(d2) / math.sqrt(b2) if b2 else float('nan'):.6f}")
    print(f"wrote {out}/param_diff_tensors.csv.gz ({len(rows)} rows) and param_diff_modules.csv")


if __name__ == "__main__":
    main()
