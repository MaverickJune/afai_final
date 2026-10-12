"""Compare two HF checkpoints: per-tensor equality and (optionally) logits on GPU.

usage: python compare_hf.py ORIG_DIR NEW_DIR [--logits] [--expect-changed REGEX] [--expect-equal REGEX]

--expect-equal REGEX     tensors matching REGEX must be bit-identical (fail otherwise)
--expect-changed REGEX   tensors matching REGEX must differ (fail otherwise)
Without either flag, all tensors must be bit-identical.
"""

import argparse
import glob
import os
import re
import sys

import torch
from safetensors.torch import load_file


def load_dir(d):
    sd = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        sd.update(load_file(f))
    return sd


p = argparse.ArgumentParser()
p.add_argument("orig")
p.add_argument("new")
p.add_argument("--logits", action="store_true")
p.add_argument("--expect-equal", default=None)
p.add_argument("--expect-changed", default=None)
a = p.parse_args()

A, B = load_dir(a.orig), load_dir(a.new)
print(f"orig: {len(A)} tensors, new: {len(B)} tensors")
missing, extra = sorted(set(A) - set(B)), sorted(set(B) - set(A))
print(f"missing in new: {len(missing)} {missing[:10]}")
print(f"extra in new:   {len(extra)} {extra[:10]}")

ok = not missing and not extra
n_equal = n_diff = 0
for k in sorted(set(A) & set(B)):
    x, y = A[k], B[k]
    if x.shape != y.shape:
        print(f"SHAPE MISMATCH {k}: {tuple(x.shape)} vs {tuple(y.shape)}")
        ok = False
        continue
    equal = x.dtype == y.dtype and torch.equal(x, y)
    n_equal += equal
    n_diff += not equal
    maxdiff = (x.float() - y.float()).abs().max().item()
    if a.expect_equal and re.search(a.expect_equal, k) and not equal:
        print(f"FAIL expected equal: {k} maxdiff={maxdiff:.3e} dtypes {x.dtype}/{y.dtype}")
        ok = False
    if a.expect_changed and re.search(a.expect_changed, k) and equal:
        print(f"FAIL expected changed: {k}")
        ok = False
    if not a.expect_equal and not a.expect_changed and not equal:
        print(f"DIFF {k}: maxdiff={maxdiff:.3e} dtypes {x.dtype}/{y.dtype}")
        ok = False
print(f"bit-identical tensors: {n_equal}, differing: {n_diff}")
if a.expect_equal:
    m = [k for k in A if re.search(a.expect_equal, k)]
    print(f"--expect-equal {a.expect_equal!r} matched {len(m)} tensors, e.g. {m[:3]}")
    ok &= len(m) > 0
if a.expect_changed:
    m = [k for k in A if re.search(a.expect_changed, k)]
    print(f"--expect-changed {a.expect_changed!r} matched {len(m)} tensors, e.g. {m[:3]}")
    ok &= len(m) > 0

if a.logits:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.orig)
    ids = tok("The quick brown fox jumps over the lazy dog. In 1905, Einstein", return_tensors="pt").input_ids.cuda()
    outs = []
    for d in (a.orig, a.new):
        m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).cuda().eval()
        with torch.no_grad():
            outs.append(m(ids).logits.float())
        del m
    print(f"logits max abs diff: {(outs[0] - outs[1]).abs().max().item():.3e}")
    print("argmax tokens equal:", torch.equal(outs[0].argmax(-1), outs[1].argmax(-1)))

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
