"""Download the evaluation benchmarks of "Scalpel vs. Hammer" (Rajani et al., 2025, Sec. 3.3) and normalise them to
one jsonl per benchmark. Runs inside the slime container (pandas + pyarrow + huggingface_hub), CPU only.

usage: python prep_bench.py OUT_DIR [BENCH ...]          (default: all)
writes: OUT_DIR/<bench>.jsonl   {"id", "bench", "kind": "math"|"mc", "question", "gold", "n_choices", "group",
                                 "solution" (MATH-500 only, for the KL analysis)}
        OUT_DIR/manifest.json   (repo, resolved revision, source file sha256, row counts, output sha256)

Benchmarks: math500, aime24, gsm8k (graded with slime's math grader), gpqa_diamond, mmlu, mmlu_pro (multiple choice).
IFEval is not included (needs the IFEval instruction checkers, not installed in either environment).
GPQA is gated on the Hub: it needs a token of an account that accepted its terms (saved by `hf auth login` in
$HF_HOME); without one it fails here and is reported, the others still run.
"""

import fnmatch
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

LETTERS = "ABCDEFGHIJ"


def fetch(repo, pattern):
    """Download all files of dataset `repo` matching glob `pattern` at the current revision."""
    info = HfApi().dataset_info(repo)
    files = sorted(f for f in (s.rfilename for s in info.siblings) if fnmatch.fnmatch(f, pattern))
    assert files, f"{repo}: no file matches {pattern}"
    paths, hashes = [], {}
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", revision=info.sha)
        hashes[f] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        paths.append(p)
    return info.sha, paths, hashes


def read(paths):
    parts = []
    for p in paths:
        if p.endswith(".parquet"):
            parts.append(pd.read_parquet(p))
        elif p.endswith(".jsonl"):
            parts.append(pd.read_json(p, lines=True, dtype=False))  # keep answers as strings ("025")
        elif p.endswith(".csv"):
            parts.append(pd.read_csv(p))
        else:
            raise ValueError(p)
    return pd.concat(parts, ignore_index=True)


def mc_question(stem, options):
    return stem.strip() + "\n\n" + "\n".join(f"{LETTERS[i]}) {str(o).strip()}" for i, o in enumerate(options))


def math500(df):
    return [{"id": r.unique_id, "kind": "math", "question": r.problem, "gold": str(r.answer), "n_choices": 0,
             "group": r.subject, "solution": r.solution} for r in df.itertuples()]


def aime24(df):
    return [{"id": str(r.id), "kind": "math", "question": r.problem, "gold": str(r.answer), "n_choices": 0,
             "group": "aime2024"} for r in df.itertuples()]


def gsm8k(df):
    out = []
    for i, r in enumerate(df.itertuples()):
        gold = r.answer.split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k_test_{i}", "kind": "math", "question": r.question, "gold": gold, "n_choices": 0,
                    "group": "gsm8k"})
    return out


def mmlu(df):
    return [{"id": f"mmlu_test_{i}", "kind": "mc", "question": mc_question(r.question, list(r.choices)),
             "gold": LETTERS[int(r.answer)], "n_choices": len(r.choices), "group": r.subject}
            for i, r in enumerate(df.itertuples())]


def mmlu_pro(df):
    out = []
    for r in df.itertuples():
        opts = list(r.options)
        assert LETTERS[int(r.answer_index)] == r.answer
        out.append({"id": f"mmlu_pro_{r.question_id}", "kind": "mc", "question": mc_question(r.question, opts),
                    "gold": r.answer, "n_choices": len(opts), "group": r.category})
    return out


def gpqa_diamond(df):
    out = []
    for r in df.to_dict("records"):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        # Fixed option order per question, independent of row order: seeded by the record id.
        random.Random(int(hashlib.sha256(r["Record ID"].encode()).hexdigest(), 16)).shuffle(order)
        out.append({"id": r["Record ID"], "kind": "mc", "question": mc_question(r["Question"], [opts[j] for j in order]),
                    "gold": LETTERS[order.index(0)], "n_choices": 4, "group": r.get("High-level domain", "gpqa")})
    return out


# name: (repo, file glob, converter, expected rows)
SPECS = {
    "math500": ("HuggingFaceH4/MATH-500", "test.jsonl", math500, 500),
    "aime24": ("HuggingFaceH4/aime_2024", "data/train-*.parquet", aime24, 30),
    "gsm8k": ("openai/gsm8k", "main/test-*.parquet", gsm8k, 1319),
    "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond.csv", gpqa_diamond, 198),
    "mmlu": ("cais/mmlu", "all/test-*.parquet", mmlu, 14042),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "data/test-*.parquet", mmlu_pro, 12032),
}


def main():
    out_dir = Path(sys.argv[1])
    names = sys.argv[2:] or list(SPECS)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    failed = []
    for name in names:
        repo, pattern, conv, expected = SPECS[name]
        try:
            rev, paths, hashes = fetch(repo, pattern)
            rows = conv(read(paths))
            assert len(rows) == expected, f"{name}: expected {expected} rows, got {len(rows)}"
            assert len({r["id"] for r in rows}) == len(rows), f"{name}: duplicate ids"
        except Exception as e:  # report and continue (GPQA is gated)
            print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)
            failed.append(name)
            continue
        out = out_dir / f"{name}.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps({"bench": name, **r}, ensure_ascii=False) + "\n")
        manifest[name] = {"repo": repo, "revision": rev, "files_sha256": hashes, "n": len(rows),
                          "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        print(f"{name}: {len(rows)} rows from {repo}@{rev} -> {out}")
        print("  first question:", json.dumps(rows[0], ensure_ascii=False)[:600], flush=True)
    man_path.write_text(json.dumps(manifest, indent=2))
    if failed:
        print("FAILED benchmarks:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
