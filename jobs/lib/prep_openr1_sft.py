"""Prepare the SFT set: rows START, START+STEP, START+2*STEP, ... (0-based, dataset order) of
open-r1/OpenR1-Math-220k (config "default", split "train"), i.e. pandas `df.iloc[START::STEP]`, and validate it
against slime's SFT loss masking with the Qwen3-30B-A3B-Base tokenizer.

usage: python prep_openr1_sft.py OUT_DIR TOKENIZER_DIR START STEP
writes: OUT_DIR/openr1_math_default_train_idx{STEP}k{START}.jsonl  ({"messages", "uuid", "row_index"} per line)
        OUT_DIR/manifest.json                                   (source revision, file hashes, token statistics)
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
SHARDS = [f"data/train-{i:05d}-of-00010.parquet" for i in range(10)]  # config "default" = data/train-*, in order
EXPECTED_ROWS = 93733

out_dir, tok_dir, start, step = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)

revision = HfApi().dataset_info(REPO).sha
parts, shard_hashes = [], {}
for shard in SHARDS:
    path = hf_hub_download(REPO, shard, repo_type="dataset", revision=revision)
    shard_hashes[shard] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    parts.append(pd.read_parquet(path, columns=["messages", "uuid"]))
    print(f"{shard}: {len(parts[-1])} rows, sha256 {shard_hashes[shard]}", flush=True)
full = pd.concat(parts, ignore_index=True)
print(f"{REPO}@{revision}: {len(full)} rows in train")
assert len(full) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(full)}"
df = full.iloc[start::step]
n = len(df)
print(f"selected iloc[{start}::{step}]: {n} rows, indices {df.index[0]}, {df.index[1]}, {df.index[2]}, ..., {df.index[-1]}")


def to_messages(m):
    return [{"role": str(x["role"]), "content": str(x["content"])} for x in m]


rows = [{"messages": to_messages(m), "uuid": u, "row_index": int(i)} for i, m, u in zip(df.index, df["messages"], df["uuid"])]
roles = {tuple(x["role"] for x in r["messages"]) for r in rows}
print("role patterns:", roles)

out_file = out_dir / f"openr1_math_default_train_idx{step}k{start}.jsonl"
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
    "source_files_sha256": shard_hashes,
    "total_rows": len(full),
    "selection": f"0-based rows {start}, {start + step}, {start + 2 * step}, ... of the train split in dataset order "
                 f"(pandas iloc[{start}::{step}] over the 10 shards concatenated in order); no filtering, no shuffling",
    "n_selected": n,
    "first_row_index": rows[0]["row_index"],
    "last_row_index": rows[-1]["row_index"],
    "first_uuid": rows[0]["uuid"],
    "last_uuid": rows[-1]["uuid"],
    "output": str(out_file),
    "output_sha256": hashlib.sha256(out_file.read_bytes()).hexdigest(),
    "tokenizer": tok_dir,
    "token_stats": stats,
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
