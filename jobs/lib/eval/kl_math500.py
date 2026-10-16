"""Per-token KL divergence between the base model and fine-tuned checkpoints on MATH-500 ("Scalpel vs. Hammer" Sec. 4.1).

Teacher forcing on held-out text: each MATH-500 problem with its reference solution, formatted as one chat turn
(the evaluation prompt rendered with the base chat template, then solution + <|im_end|>). On the solution tokens:
    KL(base || ckpt) = sum_v p_base(v) [log p_base(v) - log p_ckpt(v)]     (the paper's direction)
    KL(ckpt || base)                                                          (reverse, also reported)
averaged over all solution tokens (token-weighted) and per question. Full-vocabulary distributions, fp32 softmax.

The base model stays on the GPU; checkpoints are loaded one at a time next to it (2 x 57 GB bf16 for the 30B model
fits on one 141 GB H200). Runs in afai_env.

usage: python kl_math500.py --base BASE_HF_DIR --ckpt LABEL=DIR [...] --data math500.jsonl --out OUT_DIR [--limit N]
writes: OUT_DIR/kl_math500.csv               per checkpoint: token-weighted and per-question mean KL (both directions)
        OUT_DIR/kl_math500_per_question.jsonl
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MATH_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."  # as in run_bench.py
POS_CHUNK = 1024  # positions per softmax chunk (bounds fp32 memory: 1024 x 151936 x 4 B = 0.6 GB per tensor)


def load(d):
    return AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cuda").eval()


@torch.no_grad()
def logits_on(model, ids, start):
    """Logits predicting tokens ids[start:], i.e. positions start-1 .. len-2."""
    out = model(input_ids=ids.unsqueeze(0).cuda()).logits[0]
    return out[start - 1:-1]


@torch.no_grad()
def kl_both(lb, lc):
    """Per-position KL(b||c) and KL(c||b) from two [T, V] logit tensors, in fp32, chunked over positions."""
    fwd, rev = [], []
    for i in range(0, lb.shape[0], POS_CHUNK):
        a = torch.log_softmax(lb[i:i + POS_CHUNK].float(), -1)
        b = torch.log_softmax(lc[i:i + POS_CHUNK].float(), -1)
        fwd.append((a.exp() * (a - b)).sum(-1))
        rev.append((b.exp() * (b - a)).sum(-1))
    return torch.cat(fwd).cpu(), torch.cat(rev).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--ckpt", action="append", required=True, help="LABEL=DIR")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    recs = [json.loads(l) for l in open(a.data)]
    if a.limit:
        recs = recs[:a.limit]
    seqs = []
    for r in recs:
        prompt = tok.apply_chat_template([{"role": "user", "content": f"{MATH_INSTR}\n\n{r['question']}"}],
                                         tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        s_ids = tok(r["solution"] + "<|im_end|>", add_special_tokens=False).input_ids  # separate: no boundary merge
        seqs.append((r["id"], torch.tensor(p_ids + s_ids), len(p_ids)))
    n_tok = sum(len(s) - st for _, s, st in seqs)
    print(f"{len(seqs)} sequences, {n_tok} solution tokens, longest sequence {max(len(s) for _, s, _ in seqs)}")

    t0 = time.time()
    base = load(a.base)
    print(f"base loaded in {(time.time() - t0) / 60:.1f} min; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    summary = []
    with open(out / "kl_math500_per_question.jsonl", "w") as fq:
        for lbl, d in (c.split("=", 1) for c in a.ckpt):
            t1 = time.time()
            same = Path(d).resolve() == Path(a.base).resolve()
            model = base if same else load(d)
            print(f"[{lbl}] loaded in {(time.time() - t1) / 60:.1f} min; GPU memory "
                  f"{torch.cuda.memory_allocated() / 1e9:.1f} GB (peak {torch.cuda.max_memory_allocated() / 1e9:.1f})", flush=True)
            sum_f = sum_r = 0.0
            per_q_f, per_q_r = [], []
            for qid, ids, st in seqs:
                lb = logits_on(base, ids, st)
                lc = lb if same else logits_on(model, ids, st)
                f, r = kl_both(lb, lc)
                sum_f += f.sum().item()
                sum_r += r.sum().item()
                per_q_f.append(f.mean().item())
                per_q_r.append(r.mean().item())
                fq.write(json.dumps({"ckpt": lbl, "id": qid, "n_tokens": len(f), "kl_base_ckpt": per_q_f[-1],
                                     "kl_ckpt_base": per_q_r[-1]}) + "\n")
                del lb, lc
            row = {"ckpt": lbl, "n_questions": len(seqs), "n_tokens": n_tok,
                   "kl_base_ckpt_token_mean": sum_f / n_tok, "kl_ckpt_base_token_mean": sum_r / n_tok,
                   "kl_base_ckpt_question_mean": sum(per_q_f) / len(per_q_f),
                   "kl_ckpt_base_question_mean": sum(per_q_r) / len(per_q_r),
                   "minutes": (time.time() - t1) / 60}
            summary.append(row)
            print(f"[{lbl}] KL(base||ckpt) {row['kl_base_ckpt_token_mean']:.5f}  KL(ckpt||base) "
                  f"{row['kl_ckpt_base_token_mean']:.5f} nats/token  ({row['minutes']:.1f} min)", flush=True)
            if not same:
                del model
                torch.cuda.empty_cache()
    with open(out / "kl_math500.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {out}/kl_math500.csv")


if __name__ == "__main__":
    main()
