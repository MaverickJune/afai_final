"""Prepare the SFT set: first N rows of open-r1/OpenR1-Math-220k (config "default", split "train"), and
validate it against slime's SFT loss masking with the Qwen3-30B-A3B-Base tokenizer.

usage: python prep_openr1_sft.py OUT_DIR TOKENIZER_DIR N
writes: OUT_DIR/openr1_math_default_train_first{N}.jsonl   ({"messages": [...], "uuid": ...} per line)
        OUT_DIR/manifest.json                             (source revision, file hash, token statistics)
Runs inside the slime container (needs slime.utils.mask_utils).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from slime.utils.mask_utils import MultiTurnLossMaskGenerator

REPO = "open-r1/OpenR1-Math-220k"
SHARD = "data/train-00000-of-00010.parquet"  # config "default" = data/train-*; rows are in dataset order

out_dir, tok_dir, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

revision = HfApi().dataset_info(REPO).sha
shard_path = hf_hub_download(REPO, SHARD, repo_type="dataset", revision=revision)
sha256 = hashlib.sha256(Path(shard_path).read_bytes()).hexdigest()
df = pd.read_parquet(shard_path)
print(f"{REPO}@{revision} {SHARD}: {len(df)} rows, sha256 {sha256}")
assert len(df) >= n, f"first shard has only {len(df)} rows"
df = df.iloc[:n]


def to_messages(m):
    return [{"role": str(x["role"]), "content": str(x["content"])} for x in m]


rows = [{"messages": to_messages(m), "uuid": u} for m, u in zip(df["messages"], df["uuid"])]
roles = {tuple(x["role"] for x in r["messages"]) for r in rows}
print("role patterns:", roles)

out_file = out_dir / f"openr1_math_default_train_first{n}.jsonl"
with open(out_file, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", out_file)

tok = AutoTokenizer.from_pretrained(tok_dir)
assert tok.chat_template, "tokenizer has no chat template"
gen3 = MultiTurnLossMaskGenerator(tok, tokenizer_type="qwen3")
gen_q = MultiTurnLossMaskGenerator(tok, tokenizer_type="qwen")

# Loss-mask check on the first sample: "qwen3" must keep the <think> reasoning, "qwen" is shown for contrast.
for name, gen in (("qwen3", gen3), ("qwen", gen_q)):
    ids, mask = gen.get_loss_mask(rows[0]["messages"])
    trained = tok.decode([t for t, m in zip(ids, mask) if m])
    print(f"--- loss-mask-type {name}: {len(ids)} tokens, {sum(mask)} trained; trained text starts/ends:")
    print(repr(trained[:200]), "...", repr(trained[-120:]))
    print(f"    trained text contains '<think>': {'<think>' in trained}, '</think>': {'</think>' in trained}")

has_think = sum("<think>" in r["messages"][-1]["content"] for r in rows)
print(f"assistant messages containing '<think>': {has_think}/{n}")

lengths, trained = [], []
for r in rows:
    ids, mask = gen3.get_loss_mask(r["messages"])
    lengths.append(len(ids))
    trained.append(sum(mask))
lengths, trained = np.array(lengths), np.array(trained)
stats = {
    "n": int(n),
    "total_tokens": int(lengths.sum()),
    "trained_tokens": int(trained.sum()),
    "mean": float(lengths.mean()),
    "p50": float(np.percentile(lengths, 50)),
    "p95": float(np.percentile(lengths, 95)),
    "p99": float(np.percentile(lengths, 99)),
    "max": int(lengths.max()),
    "over_16384": int((lengths > 16384).sum()),
    "over_20480": int((lengths > 20480).sum()),
    "over_32768": int((lengths > 32768).sum()),
}
print("token stats (loss-mask-type qwen3):", json.dumps(stats))
manifest = {
    "dataset": REPO,
    "config": "default",
    "split": "train",
    "revision": revision,
    "source_file": SHARD,
    "source_sha256": sha256,
    "selection": f"first {n} rows in dataset order (no filtering, no shuffling)",
    "first_uuid": rows[0]["uuid"],
    "last_uuid": rows[-1]["uuid"],
    "output": str(out_file),
    "output_sha256": hashlib.sha256(out_file.read_bytes()).hexdigest(),
    "tokenizer": tok_dir,
    "token_stats": stats,
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
