"""Download the evaluation benchmarks of "Scalpel vs. Hammer" (Rajani et al., 2025, Sec. 3.3) and normalise them to
one jsonl per benchmark. Runs inside the slime container (pandas + pyarrow + huggingface_hub), CPU only.

usage: python prep_bench.py OUT_DIR [NAME ...]          (default: all benchmarks, then all probe sets)
writes: OUT_DIR/<bench>.jsonl   {"id", "bench", "kind": "math"|"mc", "question", "gold", "n_choices", "group",
                                 "solution" (MATH-500 only)}
        OUT_DIR/probes/<probe>.jsonl   teacher-forcing text for probe_analysis.py (KL + routing):
                                 {"id", "probe", "mode": "answer"|"text", "question", "target"}
        OUT_DIR/manifest.json   (repo, resolved revision, source file sha256, row counts, output sha256)

Benchmarks: math500, aime24, aime26, hmmt_feb26, gsm8k (graded with slime's math grader); gpqa_diamond, mmlu,
mmlu_pro (multiple choice). AIME 2026 / HMMT Feb 2026 postdate the Qwen3 release (no contamination); they are taken
from MathArena's Hub copies (if a repo name changes, the job reports it and the rest still run).
Probe sets (built after the benchmarks they depend on):
  openr1_heldout  300 rows of OpenR1-Math-220k default/train iloc[2::3] (the third never trained on by the SFT
                  iloc[1::3] or GRPO iloc[0::3] runs), random.Random(0).sample; target = the R1 solution (<think> trace
                  + answer), i.e. held-out text from the training distribution
  math500         the 500 MATH-500 problems, target = the human reference solution (the paper's KL probe)
  general         300 MMLU test questions (random.Random(0).sample), mode "text": target = the question text itself
                  (general-knowledge text, for expert usage outside math)
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


def matharena(df):
    """MathArena competition sets (columns problem_idx / problem / answer)."""
    idcol = next((c for c in ("problem_idx", "id", "idx") if c in df.columns), None)
    return [{"id": str(getattr(r, idcol) if idcol else i), "kind": "math", "question": r.problem,
             "gold": str(r.answer), "n_choices": 0, "group": str(getattr(r, "problem_type", "competition"))}
            for i, r in enumerate(df.itertuples())]


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


# name: (repo, file glob, converter, expected rows or None)
SPECS = {
    "math500": ("HuggingFaceH4/MATH-500", "test.jsonl", math500, 500),
    "aime24": ("HuggingFaceH4/aime_2024", "data/train-*.parquet", aime24, 30),
    "aime26": ("MathArena/aime_2026", "*.parquet", matharena, 30),
    "hmmt_feb26": ("MathArena/hmmt_feb_2026", "*.parquet", matharena, None),
    "gsm8k": ("openai/gsm8k", "main/test-*.parquet", gsm8k, 1319),
    "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond.csv", gpqa_diamond, 198),
    "mmlu": ("cais/mmlu", "all/test-*.parquet", mmlu, 14042),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "data/test-*.parquet", mmlu_pro, 12032),
}


OPENR1 = "open-r1/OpenR1-Math-220k"
OPENR1_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"  # the SFT / GRPO data snapshot
OPENR1_SHARDS = [f"data/train-{i:05d}-of-00010.parquet" for i in range(10)]
N_PROBE = 300


def write_probe(out_dir, name, rows):
    pdir = out_dir / "probes"
    pdir.mkdir(exist_ok=True)
    out = pdir / f"{name}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({"probe": name, **r}, ensure_ascii=False) + "\n")
    print(f"probe {name}: {len(rows)} rows -> {out}", flush=True)
    return {"n": len(rows), "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def build_probes(out_dir):
    man = {}
    parts, hashes = [], {}
    for shard in OPENR1_SHARDS:
        p = hf_hub_download(OPENR1, shard, repo_type="dataset", revision=OPENR1_REVISION)
        hashes[shard] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        parts.append(pd.read_parquet(p, columns=["messages", "uuid"]))
    full = pd.concat(parts, ignore_index=True)
    assert len(full) == 93733, len(full)
    held = full.iloc[2::3]
    pick = sorted(random.Random(0).sample(list(held.index), N_PROBE))
    rows = []
    for i in pick:
        m = full.at[i, "messages"]
        assert [str(x["role"]) for x in m] == ["user", "assistant"]
        rows.append({"id": f"openr1_row{i}", "mode": "answer", "question": str(m[0]["content"]),
                     "target": str(m[1]["content"])})
    man["openr1_heldout"] = {"repo": OPENR1, "revision": OPENR1_REVISION, "files_sha256": hashes,
                             "selection": f"random.Random(0).sample of iloc[2::3] ({len(held)} rows), {N_PROBE}",
                             **write_probe(out_dir, "openr1_heldout", rows)}

    m5 = [json.loads(l) for l in open(out_dir / "math500.jsonl")]
    man["math500"] = {"source": "math500.jsonl", **write_probe(out_dir, "math500", [
        {"id": r["id"], "mode": "answer", "question": r["question"], "target": r["solution"]} for r in m5])}

    mm = [json.loads(l) for l in open(out_dir / "mmlu.jsonl")]
    pick = sorted(random.Random(0).sample(range(len(mm)), N_PROBE))
    man["general"] = {"source": "mmlu.jsonl", "selection": f"random.Random(0).sample, {N_PROBE}",
                      **write_probe(out_dir, "general", [
                          {"id": mm[i]["id"], "mode": "text", "question": "", "target": mm[i]["question"]}
                          for i in pick])}
    return man


def main():
    out_dir = Path(sys.argv[1])
    names = sys.argv[2:] or list(SPECS) + ["probes"]
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    failed = []
    for name in names:
        if name == "probes":
            try:
                manifest["probes"] = build_probes(out_dir)
            except Exception as e:
                print(f"FAILED probes: {type(e).__name__}: {e}", flush=True)
                failed.append(name)
            continue
        repo, pattern, conv, expected = SPECS[name]
        try:
            rev, paths, hashes = fetch(repo, pattern)
            rows = conv(read(paths))
            assert expected is None or len(rows) == expected, f"{name}: expected {expected} rows, got {len(rows)}"
            assert rows, f"{name}: no rows"
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
