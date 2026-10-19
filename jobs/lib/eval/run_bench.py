"""Generate and grade benchmark answers for one model served by a running SGLang server (native /generate API).

Runs inside the slime container (needs transformers for the chat template and slime's math grader).

usage: python run_bench.py --server http://127.0.0.1:PORT --tokenizer BASE_HF_DIR --model-dir MODEL_DIR
                           --bench-dir DIR --out DIR [--benchmarks a,b,..] [--limit N] [--max-new-cap T]
writes, per benchmark, into --out:
    <bench>.samples.jsonl   one line per (question, sample): response, finish reason, tokens, extracted answer, correct
    <bench>.summary.json    accuracy (mean over all samples = avg@n) with 95% CI over questions, pass@k,
                            truncation / no-answer rates, mean response tokens, per-group accuracy
A benchmark whose summary exists is skipped. Finished request chunks are appended to <bench>.partial.jsonl, so a
killed job resumes where it stopped.

Prompts: one user turn rendered with the BASE model's chat template (identical prompt tokens for every model):
    math: "Please reason step by step, and put your final answer within \\boxed{}.\\n\\n" + problem
          (= the GRPO run's prompt and the instruction R1 used for the SFT data, see prep_openr1_grpo.py)
    mc:   the same instruction asking for the option letter in \\boxed{}, then question + "A) ..." options
"""

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import LETTERS, last_boxed, mc_letter, mean_ci95, pass_at_k  # noqa: E402

MATH_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
MC_INSTR = ("Answer the following multiple-choice question. Please reason step by step, and put the letter of the "
            "correct option ({letters}) within \\boxed{{}}.")

# Samples per question and generation budget. Small benchmarks get several samples (avg@n has lower variance, and
# pass@n shows whether a correct answer is reachable at all, cf. Yue et al. 2025 cited in the paper).
BENCH_CFG = {
    "math500":      {"n": 4,  "max_new": 30000},
    "aime24":       {"n": 16, "max_new": 30000},
    "aime26":       {"n": 16, "max_new": 30000},
    "hmmt_feb26":   {"n": 16, "max_new": 30000},
    "gsm8k":        {"n": 1,  "max_new": 16384},
    "gpqa_diamond": {"n": 4,  "max_new": 30000},
    "mmlu":         {"n": 1,  "max_new": 16384},
    "mmlu_pro":     {"n": 1,  "max_new": 16384},
}
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": -1}  # DeepSeek-R1 / Qwen3 thinking-mode evaluation setting
CHUNK = 32          # prompts per /generate request
WORKERS = 16        # concurrent requests -> up to 512 sequences in flight; SGLang schedules them
GRADE_TIMEOUT_S = 10


def user_text(rec):
    if rec["kind"] == "math":
        return f"{MATH_INSTR}\n\n{rec['question']}"
    letters = ", ".join(LETTERS[:rec["n_choices"]])
    return f"{MC_INSTR.format(letters=letters)}\n\n{rec['question']}"


def post(url, payload, timeout=6 * 3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _timeout(signum, frame):
    raise TimeoutError


def grade_math(args):
    """slime's `math` reward (last \\boxed{} vs gold, MATH normalisation or sympy equivalence): 1, 0, or None on timeout."""
    from slime.rollout.rm_hub.math_utils import grade_answer_verl
    response, gold = args
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(GRADE_TIMEOUT_S)
    try:
        return int(bool(grade_answer_verl(response, gold)))
    except TimeoutError:
        return None
    except Exception:
        return 0
    finally:
        signal.alarm(0)


def run_one(name, a, tok, ctx_len, stop_ids):
    out_dir = Path(a.out)
    summary_path = out_dir / f"{name}.summary.json"
    if summary_path.exists():
        print(f"[{name}] summary exists, skipping")
        return
    recs = [json.loads(l) for l in open(Path(a.bench_dir) / f"{name}.jsonl")]
    if a.limit:
        recs = recs[:a.limit]
    cfg = BENCH_CFG[name]
    n, max_new = cfg["n"], min(cfg["max_new"], a.max_new_cap or 10**9)

    prompts = [tok.apply_chat_template([{"role": "user", "content": user_text(r)}], tokenize=False,
                                       add_generation_prompt=True) for r in recs]
    plens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    jobs = [(qi, k) for qi in range(len(recs)) for k in range(n)]

    partial_path = out_dir / f"{name}.partial.jsonl"
    done = {}
    if partial_path.exists():
        bad = 0
        for l in open(partial_path):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:  # line cut off by an interrupted write (e.g. disk quota)
                bad += 1
                continue
            done[(d["qi"], d["k"])] = d
        # Rewrite with only complete lines, so new lines are never appended to a cut-off one.
        print(f"[{name}] resuming: {len(done)} samples kept, {bad} unreadable line(s) dropped")
        # Written to a temp file and renamed: a failed write (e.g. disk quota) leaves the old file intact.
        tmp_path = partial_path.with_suffix(".tmp")
        with open(tmp_path, "w") as pf:
            for d in done.values():
                pf.write(json.dumps(d, ensure_ascii=False) + "\n")
        os.replace(tmp_path, partial_path)
    todo = [j for j in jobs if j not in done]
    print(f"[{name}] {len(recs)} questions x {n} samples = {len(jobs)} generations; {len(done)} already done; "
          f"max_new {max_new}; prompt tokens max {max(plens)}", flush=True)

    def request(chunk):
        sps = [{**SAMPLING, "max_new_tokens": max(1, min(max_new, ctx_len - plens[qi] - 8)), "stop_token_ids": stop_ids}
               for qi, _ in chunk]
        res = post(f"{a.server}/generate", {"text": [prompts[qi] for qi, _ in chunk], "sampling_params": sps})
        assert len(res) == len(chunk)
        return [{"qi": qi, "k": k, "response": r["text"], "finish": r["meta_info"]["finish_reason"]["type"],
                 "tokens": r["meta_info"]["completion_tokens"]} for (qi, k), r in zip(chunk, res)]

    t0, n_new = time.time(), 0
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    with open(partial_path, "a") as pf, ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(request, c) for c in chunks]
        for f in as_completed(futs):
            for d in f.result():
                done[(d["qi"], d["k"])] = d
                pf.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_new += 1
            pf.flush()
            el = time.time() - t0
            print(f"[{name}] {len(done)}/{len(jobs)} generations, {el / 60:.1f} min, "
                  f"{sum(done[j]['tokens'] for j in done) / max(len(done), 1):.0f} mean tokens", flush=True)

    samples = [done[j] for j in jobs]
    if recs[0]["kind"] == "math":
        with Pool(8) as pool:
            scores = pool.map(grade_math, [(s["response"], recs[s["qi"]]["gold"]) for s in samples], chunksize=16)
        for s, c in zip(samples, scores):
            s["extracted"], s["correct"], s["grade_timeout"] = last_boxed(s["response"]), int(c or 0), c is None
    else:
        for s in samples:
            r = recs[s["qi"]]
            s["extracted"] = mc_letter(s["response"], r["n_choices"])
            s["correct"], s["grade_timeout"] = int(s["extracted"] == r["gold"]), False

    with open(out_dir / f"{name}.samples.jsonl", "w") as f:
        for s in samples:
            r = recs[s["qi"]]
            f.write(json.dumps({"id": r["id"], "group": r["group"], "gold": r["gold"], **s}, ensure_ascii=False) + "\n")

    per_q = defaultdict(list)
    for s in samples:
        per_q[s["qi"]].append(s["correct"])
    q_acc = [sum(v) / len(v) for v in per_q.values()]
    acc, ci = mean_ci95(q_acc)
    by_group = defaultdict(list)
    for qi, v in per_q.items():
        by_group[recs[qi]["group"]].append(sum(v) / len(v))
    ks = sorted({1, n} | ({4} if n >= 4 else set()))
    summary = {
        "bench": name, "model_dir": a.model_dir, "n_questions": len(recs), "samples_per_question": n,
        "accuracy": acc, "ci95": ci,
        **{f"pass@{k}": sum(pass_at_k(len(v), sum(v), k) for v in per_q.values()) / len(per_q) for k in ks},
        "truncated_rate": sum(s["finish"] == "length" for s in samples) / len(samples),
        "no_answer_rate": sum(s["extracted"] is None for s in samples) / len(samples),
        "grade_timeouts": sum(s["grade_timeout"] for s in samples),
        "mean_response_tokens": sum(s["tokens"] for s in samples) / len(samples),
        "max_new_tokens": max_new, "sampling": SAMPLING, "limit": a.limit,
        "by_group": {g: {"n": len(v), "accuracy": sum(v) / len(v)} for g, v in sorted(by_group.items())},
        "generation_minutes_this_job": (time.time() - t0) / 60,
    }
    summary_path.write_text(json.dumps(summary, indent=1))
    partial_path.unlink()
    print(f"[{name}] accuracy {acc:.4f} +- {ci:.4f}  " + "  ".join(f"pass@{k} {summary[f'pass@{k}']:.4f}" for k in ks)
          + f"  truncated {summary['truncated_rate']:.3f}  no-answer {summary['no_answer_rate']:.3f}"
          + f"  mean tokens {summary['mean_response_tokens']:.0f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--benchmarks", default=",".join(BENCH_CFG))
    p.add_argument("--limit", type=int, default=0, help="first N questions only (testing)")
    p.add_argument("--max-new-cap", type=int, default=0, help="cap on max_new_tokens (testing)")
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ctx_len = json.load(open(Path(a.model_dir) / "config.json"))["max_position_embeddings"]
    # Stop at end of turn as well as end of text: the base tokenizer's EOS is <|endoftext|>, while the chat-trained
    # model ends its turn with <|im_end|>; without both, generation would run on into an invented next turn.
    stop_ids = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")})
    print(f"model {a.model_dir}  context {ctx_len}  stop ids {stop_ids}")
    print("--- rendered example prompt:\n" + tok.apply_chat_template(
        [{"role": "user", "content": user_text({"kind": "math", "question": "What is 1+1?"})}],
        tokenize=False, add_generation_prompt=True) + "<END>")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name in a.benchmarks.split(","):
        if not (Path(a.bench_dir) / f"{name}.jsonl").exists():
            print(f"[{name}] no data in {a.bench_dir}, skipping")
            continue
        run_one(name, a, tok, ctx_len, stop_ids)


if __name__ == "__main__":
    main()
