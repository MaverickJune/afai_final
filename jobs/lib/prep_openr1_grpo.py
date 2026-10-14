"""Prepare the GRPO prompt set: rows START, START+STEP, START+2*STEP, ... (0-based, dataset order) of
open-r1/OpenR1-Math-220k (config "default", split "train"), i.e. pandas `df.iloc[START::STEP]`, no filtering.

Prompt = INSTRUCTION + "\n\n" + the row's user message, verbatim (the user turn the SFT run trains on). The dataset
card says R1 generated the solutions with this instruction prepended; `messages` stores the problem without it.
Label = the row's `answer`. slime wraps the prompt list in the tokenizer's chat template (--apply-chat-template) and
rewards with --rm-type math (last \\boxed{} in the response vs label). Nothing is filtered; the reward-coverage numbers
below are for information only.

usage: python prep_openr1_grpo.py OUT_DIR TOKENIZER_DIR START STEP
writes: OUT_DIR/openr1_math_default_train_idx{STEP}k{START}_grpo.jsonl
            {"prompt": [{"role": "user", "content": ...}], "label", "uuid", "row_index", "question_type"} per line
        OUT_DIR/manifest.json             (source revision and hashes, counts, reward coverage, token statistics)
Runs inside the slime container (needs slime.rollout.rm_hub).
"""

import hashlib
import json
import signal
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from slime.rollout.rm_hub.math_utils import grade_answer_verl

REPO = "open-r1/OpenR1-Math-220k"
REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"  # same snapshot as the SFT data (SETTINGS.md of the SFT run)
SHARDS = [f"data/train-{i:05d}-of-00010.parquet" for i in range(10)]  # config "default" = data/train-*, in order
EXPECTED_ROWS = 93733
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."  # from the dataset card
GRADE_TIMEOUT_S = 10


def with_instruction(text):
    return f"{INSTRUCTION}\n\n{text}"


def _timeout(signum, frame):
    raise TimeoutError


def grade(args):
    """slime's --rm-type math reward of `response` against `label` (1, 0, or None on timeout)."""
    response, label = args
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(GRADE_TIMEOUT_S)
    try:
        return int(bool(grade_answer_verl(response, label)))
    except TimeoutError:
        return None
    except Exception:
        return 0
    finally:
        signal.alarm(0)


def main():
    out_dir, tok_dir = Path(sys.argv[1]), sys.argv[2]
    start, step = int(sys.argv[3]), int(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["messages", "answer", "uuid", "question_type", "correctness_math_verify"]
    parts, shard_hashes = [], {}
    for shard in SHARDS:
        path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REVISION)
        shard_hashes[shard] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        parts.append(pd.read_parquet(path, columns=cols))
        print(f"{shard}: {len(parts[-1])} rows, sha256 {shard_hashes[shard]}", flush=True)
    full = pd.concat(parts, ignore_index=True)
    print(f"{REPO}@{REVISION}: {len(full)} rows in train")
    assert len(full) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(full)}"
    df = full.iloc[start::step]
    n = len(df)
    print(f"selected iloc[{start}::{step}]: {n} rows, indices {df.index[0]}, {df.index[1]}, {df.index[2]}, ..., {df.index[-1]}")
    roles = {tuple(str(x["role"]) for x in m) for m in df["messages"]}
    print("role patterns:", roles)
    assert roles == {("user", "assistant")}, "expected one user and one assistant message per row"

    rows = [
        {
            "prompt": [{"role": "user", "content": with_instruction(str(m[0]["content"]))}],
            "label": str(a),
            "uuid": u,
            "row_index": int(i),
            "question_type": str(q),
        }
        for i, m, a, u, q in zip(df.index, df["messages"], df["answer"], df["uuid"], df["question_type"])
    ]
    out_file = out_dir / f"openr1_math_default_train_idx{step}k{start}_grpo.jsonl"
    with open(out_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out_file)

    # What the policy sees: the chat template of the base model's tokenizer, as slime applies it.
    tok = AutoTokenizer.from_pretrained(tok_dir)
    assert tok.chat_template, "tokenizer has no chat template"
    def render(prompt):
        return tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    print("--- rendered prompt of the first row:\n" + render(rows[0]["prompt"]) + "<END>")
    prompt_lens = np.array([len(tok(render(r["prompt"]), add_special_tokens=False).input_ids) for r in rows])

    # Reward coverage (information only, nothing is filtered): the training reward of each row's R1 reference solution
    # (the SFT data's assistant message). Rows where even that gets 0 rarely give a learning signal, so this fraction
    # is a rough ceiling for the mean training reward.
    with Pool(8) as pool:
        r1_ref = pool.map(grade, [(m[-1]["content"], str(a)) for m, a in zip(df["messages"], df["answer"])], chunksize=64)
    by_type = {}
    for qt in sorted(set(df["question_type"])):
        idx = [k for k, r in enumerate(rows) if r["question_type"] == qt]
        by_type[qt] = {"n": len(idx), "r1_ref_reward_1": sum(r1_ref[k] == 1 for k in idx)}
    coverage = {
        "r1_ref_reward_1": sum(x == 1 for x in r1_ref),
        "r1_ref_timeouts": sum(x is None for x in r1_ref),
        "math_verify_any_correct": int(sum(any(c) for c in df["correctness_math_verify"])),
        "by_question_type": by_type,
    }
    print("reward coverage (rm-type math):", json.dumps(coverage))
    stats = {
        "n": int(n),
        "prompt_tokens_mean": float(prompt_lens.mean()),
        "prompt_tokens_p99": float(np.percentile(prompt_lens, 99)),
        "prompt_tokens_max": int(prompt_lens.max()),
    }
    print("prompt token stats:", json.dumps(stats))

    manifest = {
        "dataset": REPO,
        "config": "default",
        "split": "train",
        "revision": REVISION,
        "source_files_sha256": shard_hashes,
        "total_rows": len(full),
        "selection": f"0-based rows {start}, {start + step}, {start + 2 * step}, ... of the train split in dataset order "
                     f"(pandas iloc[{start}::{step}] over the 10 shards concatenated in order); no filtering, no shuffling",
        "n_selected": n,
        "question_types": dict(Counter(r["question_type"] for r in rows)),
        "first_row_index": rows[0]["row_index"],
        "last_row_index": rows[-1]["row_index"],
        "first_uuid": rows[0]["uuid"],
        "last_uuid": rows[-1]["uuid"],
        "prompt_format": f"user message = {INSTRUCTION!r} + '\\n\\n' + messages[0].content (verbatim); label = answer",
        "output": str(out_file),
        "output_sha256": hashlib.sha256(out_file.read_bytes()).hexdigest(),
        "tokenizer": tok_dir,
        "reward_coverage": coverage,
        "prompt_token_stats": stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
