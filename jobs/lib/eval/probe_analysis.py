"""Output drift (KL) and routing drift of checkpoints relative to a reference model, on fixed probe text.

Teacher forcing on probe sets written by prep_bench.py (OUT_DIR/probes/<probe>.jsonl). Per sequence, the reference
model and the checkpoint see the same tokens; analysed are the target tokens (truncated to --max-target):
  mode "answer": prefix = eval prompt (instruction + question) in the base chat template, target = solution + <|im_end|>
  mode "text":   prefix = "<|im_start|>user\\n",                                         target = text + <|im_end|>

1. KL ("Scalpel vs. Hammer" Sec. 4.1): full-vocabulary KL(ref || ckpt) and KL(ckpt || ref) of the next-token
   distributions predicting each target token, fp32, token-weighted and per-sequence means.
2. Routing (MoE): a forward pre-hook on every layer's router (`mlp.gate`) takes the router input h and recomputes the
   top-k expert set (fp32 logits, k = num_experts_per_tok) for each target token:
     R = ref router on ref inputs        (reference routing)
     C = ckpt router on ckpt inputs      (actual routing after training)
     F = REF router on ckpt inputs       (counterfactual: what the old router would choose now)
   overlap(X, Y) = |X ∩ Y| / k averaged over tokens, per layer:
     overlap_total  = overlap(C, R)   how much routing changed
     overlap_inputs = overlap(F, R)   change due to drifted router inputs alone (router weights held at ref)
     overlap_router = overlap(C, F)   change due to router weights alone (same inputs); exactly 1 if the router is frozen
   plus expert load (tokens per expert, per layer) for ref and ckpt and the Jensen-Shannon divergence between them.

Checkpoints are processed one at a time next to the reference model on one GPU (2 x 57 GB bf16 for the 30B model).
Resumable per checkpoint. Runs in afai_env.

usage: python probe_analysis.py --ref REF_HF_DIR --ckpt LABEL=DIR [...] --probe-dir DIR --out OUT_DIR
                                [--probes a,b] [--tokenizer DIR] [--limit N] [--max-target 4096]
writes: OUT_DIR/kl.csv, routing.csv, expert_load.csv.gz (all checkpoints; per-checkpoint parts in OUT_DIR/ckpts/LABEL/)
"""

import argparse
import csv
import gzip
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MATH_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."  # as in run_bench.py
POS_CHUNK = 1024  # positions per fp32 softmax chunk (1024 x 151936 x 4 B = 0.6 GB per tensor)
GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")


class RouterTap:
    """Forward pre-hooks on all routers: per layer, top-k expert indices of positions >= self.start (and the
    counterfactual top-k under `cf_weights`, the reference router weights, if given)."""

    def __init__(self, model, k, cf_weights=None):
        self.k, self.cf_weights, self.start, self.seq_len = k, cf_weights, 0, 0
        self.top, self.cf = {}, {}
        self.gates = {}
        for name, mod in model.named_modules():
            m = GATE_RE.search(name)
            if m and hasattr(mod, "weight"):
                self.gates[int(m.group(1))] = mod
                mod.register_forward_pre_hook(self._hook(int(m.group(1))))
        assert self.gates, "no router modules (layers.N.mlp.gate) found"

    def weights_fp32(self):
        return {l: g.weight.detach().float().clone() for l, g in self.gates.items()}

    def _hook(self, layer):
        def hook(module, args):
            h = args[0].reshape(-1, args[0].shape[-1])
            assert h.shape[0] == self.seq_len, f"router input has {h.shape[0]} rows, sequence {self.seq_len}"
            h = h[self.start:].float()
            self.top[layer] = (h @ module.weight.float().T).topk(self.k, dim=-1).indices
            if self.cf_weights is not None:
                self.cf[layer] = (h @ self.cf_weights[layer].T).topk(self.k, dim=-1).indices
        return hook


def load(d):
    return AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cuda").eval()


@torch.no_grad()
def run(model, tap, ids, start):
    tap.start, tap.seq_len = start, len(ids)
    logits = model(input_ids=ids.unsqueeze(0).cuda()).logits[0]
    return logits[start - 1:-1]  # predicting ids[start:]


@torch.no_grad()
def kl_both(la, lb):
    fwd, rev = [], []
    for i in range(0, la.shape[0], POS_CHUNK):
        a = torch.log_softmax(la[i:i + POS_CHUNK].float(), -1)
        b = torch.log_softmax(lb[i:i + POS_CHUNK].float(), -1)
        fwd.append((a.exp() * (a - b)).sum(-1))
        rev.append((b.exp() * (b - a)).sum(-1))
    return torch.cat(fwd), torch.cat(rev)


def overlap_sum(a, b, n_exp, k):
    ma = torch.zeros(a.shape[0], n_exp, device=a.device).scatter_(1, a, 1.0)
    mb = torch.zeros(b.shape[0], n_exp, device=b.device).scatter_(1, b, 1.0)
    return ((ma * mb).sum(1) / k).sum().item()


def js_div(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(x, y):
        nz = x > 0
        return (x[nz] * (x[nz] / y[nz]).log()).sum().item()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def build_sequences(tok, path, limit, max_target, max_len):
    recs = [json.loads(l) for l in open(path)]
    if limit:
        recs = recs[:limit]
    end_ids = tok("<|im_end|>", add_special_tokens=False).input_ids
    seqs = []
    for r in recs:
        if r["mode"] == "answer":
            prefix = tok.apply_chat_template([{"role": "user", "content": f"{MATH_INSTR}\n\n{r['question']}"}],
                                             tokenize=False, add_generation_prompt=True)
        else:
            prefix = "<|im_start|>user\n"
        p_ids = tok(prefix, add_special_tokens=False).input_ids
        t_ids = tok(r["target"], add_special_tokens=False).input_ids + end_ids  # separate: no boundary merge
        t_ids = t_ids[:max(1, min(max_target, max_len - len(p_ids)))]
        seqs.append((r["id"], torch.tensor(p_ids + t_ids), len(p_ids)))
    return seqs


def write_csv(path, rows, gz=False):
    opener = gzip.open if gz else open
    with opener(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--ckpt", action="append", required=True, help="LABEL=DIR")
    p.add_argument("--probe-dir", required=True)
    p.add_argument("--probes", default="openr1_heldout,math500,general")
    p.add_argument("--tokenizer", default=None, help="default: --ref")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-target", type=int, default=4096)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.ref)
    cfg = json.load(open(Path(a.ref) / "config.json"))
    k, n_exp, max_len = cfg["num_experts_per_tok"], cfg["num_experts"], cfg["max_position_embeddings"]
    probes = {}
    for name in a.probes.split(","):
        probes[name] = build_sequences(tok, Path(a.probe_dir) / f"{name}.jsonl", a.limit, a.max_target, max_len)
        print(f"probe {name}: {len(probes[name])} sequences, {sum(len(s) - st for _, s, st in probes[name])} target "
              f"tokens, longest {max(len(s) for _, s, _ in probes[name])}", flush=True)

    t0 = time.time()
    ref = load(a.ref)
    ref_tap = RouterTap(ref, k)
    ref_w = ref_tap.weights_fp32()
    print(f"ref loaded in {(time.time() - t0) / 60:.1f} min; {len(ref_w)} routers, top-{k} of {n_exp}; "
          f"GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    labels = []
    for lbl, d in (c.split("=", 1) for c in a.ckpt):
        labels.append(lbl)
        cdir = out / "ckpts" / lbl
        if (cdir / "done.json").exists():
            print(f"[{lbl}] done earlier, skipping")
            continue
        assert Path(d).resolve() != Path(a.ref).resolve(), "checkpoint is the reference model"
        t1 = time.time()
        model = load(d)
        tap = RouterTap(model, k, cf_weights=ref_w)
        assert set(tap.gates) == set(ref_w)
        print(f"[{lbl}] loaded in {(time.time() - t1) / 60:.1f} min; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB",
              flush=True)
        kl_rows, rt_rows, load_rows = [], [], []
        for pname, seqs in probes.items():
            n_tok = 0
            s_fwd = s_rev = 0.0
            q_fwd, q_rev = [], []
            L = sorted(ref_w)
            ov = {l: [0.0, 0.0, 0.0] for l in L}
            ld_ref = {l: torch.zeros(n_exp, device="cuda") for l in L}
            ld_ck = {l: torch.zeros(n_exp, device="cuda") for l in L}
            for _, ids, st in seqs:
                la = run(ref, ref_tap, ids, st)
                lb = run(model, tap, ids, st)
                f, r = kl_both(la, lb)
                n = f.shape[0]
                n_tok += n
                s_fwd += f.sum().item()
                s_rev += r.sum().item()
                q_fwd.append(f.mean().item())
                q_rev.append(r.mean().item())
                for l in L:
                    R, C, F = ref_tap.top[l], tap.top[l], tap.cf[l]
                    assert R.shape[0] == n
                    ov[l][0] += overlap_sum(C, R, n_exp, k)
                    ov[l][1] += overlap_sum(F, R, n_exp, k)
                    ov[l][2] += overlap_sum(C, F, n_exp, k)
                    ld_ref[l] += torch.bincount(R.flatten(), minlength=n_exp)
                    ld_ck[l] += torch.bincount(C.flatten(), minlength=n_exp)
                del la, lb
            kl_rows.append({"ckpt": lbl, "probe": pname, "n_seqs": len(seqs), "n_tokens": n_tok,
                            "kl_ref_ckpt_token_mean": s_fwd / n_tok, "kl_ckpt_ref_token_mean": s_rev / n_tok,
                            "kl_ref_ckpt_seq_mean": sum(q_fwd) / len(q_fwd),
                            "kl_ckpt_ref_seq_mean": sum(q_rev) / len(q_rev)})
            for l in L:
                rt_rows.append({"ckpt": lbl, "probe": pname, "layer": l, "n_tokens": n_tok,
                                "overlap_total": ov[l][0] / n_tok, "overlap_inputs": ov[l][1] / n_tok,
                                "overlap_router": ov[l][2] / n_tok, "load_js": js_div(ld_ref[l], ld_ck[l])})
                for e in range(n_exp):
                    load_rows.append({"model": lbl, "probe": pname, "layer": l, "expert": e,
                                      "tokens": int(ld_ck[l][e].item()), "ref_tokens": int(ld_ref[l][e].item())})
            mean = lambda key: sum(x[key] for x in rt_rows if x["probe"] == pname) / len(L)  # noqa: E731
            print(f"[{lbl}] {pname}: KL(ref||ckpt) {kl_rows[-1]['kl_ref_ckpt_token_mean']:.5f}  "
                  f"KL(ckpt||ref) {kl_rows[-1]['kl_ckpt_ref_token_mean']:.5f} nats/token; routing overlap "
                  f"total {mean('overlap_total'):.4f} inputs-only {mean('overlap_inputs'):.4f} "
                  f"router-only {mean('overlap_router'):.4f} (mean over layers)", flush=True)
        cdir.mkdir(parents=True, exist_ok=True)
        write_csv(cdir / "kl.csv", kl_rows)
        write_csv(cdir / "routing.csv", rt_rows)
        write_csv(cdir / "expert_load.csv.gz", load_rows, gz=True)
        (cdir / "done.json").write_text(json.dumps({"dir": d, "ref": a.ref, "minutes": (time.time() - t1) / 60}))
        print(f"[{lbl}] finished in {(time.time() - t1) / 60:.1f} min", flush=True)
        del model, tap
        torch.cuda.empty_cache()

    for fname, gz in (("kl.csv", False), ("routing.csv", False), ("expert_load.csv.gz", True)):
        rows = [r for lbl in labels if (out / "ckpts" / lbl / fname).exists() for r in read_csv(out / "ckpts" / lbl / fname)]
        if rows:
            write_csv(out / fname, rows, gz=gz)
    print(f"wrote {out}/kl.csv, routing.csv, expert_load.csv.gz")


if __name__ == "__main__":
    main()
