#!/bin/bash
#SBATCH --job-name=test_eval_tiny
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0:15:00
#SBATCH --output=logs/%x_%j.out
# Known-answer tests of the analysis code on the tiny Qwen3-MoE from verify_router_freeze (afai_env):
#   param_diff: round trip (bit-identical copy) -> all changes 0; frozen-router SFT -> router change exactly 0,
#               attention changed; control SFT -> router changed
#   probe_analysis: round trip (separately loaded, identical weights) -> KL 0, routing overlaps all exactly 1;
#               frozen router -> router-only overlap exactly 1 in every layer (routing can only move through the
#               inputs), KL > 0; control -> KL > 0
#   expert_specialization: runs on the outputs above
# Needs $BENCH_DIR/probes/ (jobs/prep_eval_bench.sbatch).
set -eo pipefail
source jobs/slime_common.sh
source jobs/configs/eval.sh
TINY=$SCR/verify/router_freeze
OUT=$EVAL_ROOT/test/tiny
rm -rf "$OUT"   # outputs of a previous run of this test only
activate_afai_env
echo "host: $(hostname)  start: $(date)"
python jobs/lib/eval/test_scoring.py

python jobs/lib/eval/param_diff.py --base "$TINY/hf_tiny_moe" --out "$OUT/param_diff" \
    --ckpt roundtrip="$TINY/hf_roundtrip" --ckpt frozen="$TINY/sft_frozen_hf" --ckpt control="$TINY/sft_control_hf"
python jobs/lib/eval/probe_analysis.py --ref "$TINY/hf_tiny_moe" --tokenizer "$BASE_HF" --probe-dir "$BENCH_DIR/probes" --out "$OUT/probe" \
    --limit 10 --max-target 512 \
    --ckpt roundtrip="$TINY/hf_roundtrip" --ckpt frozen="$TINY/sft_frozen_hf" --ckpt control="$TINY/sft_control_hf"
python jobs/lib/eval/expert_specialization.py --probe-out "$OUT/probe" --param-diff "$OUT/param_diff" \
    --out "$OUT/specialization.csv"

python - "$OUT" <<'PY'
import csv, math, sys
out = sys.argv[1]
rows = list(csv.DictReader(open(f"{out}/param_diff/param_diff_modules.csv")))
def rel(ck, mod):
    return [float(r["rel"]) for r in rows if r["ckpt"] == ck and r["module"] == mod]
fail = []
if any(float(r["rel"]) != 0 for r in rows if r["ckpt"] == "roundtrip"): fail.append("roundtrip has nonzero change")
if any(v != 0 for v in rel("frozen", "router")) or not rel("frozen", "router"): fail.append("frozen router changed")
if not all(v > 0 for v in rel("frozen", "q_proj")): fail.append("frozen run: q_proj unchanged")
if not all(v > 0 for v in rel("control", "router")): fail.append("control run: router unchanged")
kl = list(csv.DictReader(open(f"{out}/probe/kl.csv")))
rt = list(csv.DictReader(open(f"{out}/probe/routing.csv")))
for r in kl:
    v = float(r["kl_ref_ckpt_token_mean"])
    print(r["ckpt"], r["probe"], "KL", v, float(r["kl_ckpt_ref_token_mean"]))
    if r["ckpt"] == "roundtrip" and abs(v) > 1e-6: fail.append(f"roundtrip KL not 0 ({r['probe']})")
    if r["ckpt"] != "roundtrip" and not (v > 0 and math.isfinite(v)): fail.append(f"{r['ckpt']} KL {v} ({r['probe']})")
for r in rt:
    print(r["ckpt"], r["probe"], "layer", r["layer"], "overlap total/inputs/router",
          r["overlap_total"], r["overlap_inputs"], r["overlap_router"], "load JS", r["load_js"])
    if r["ckpt"] == "roundtrip" and any(float(r[c]) != 1.0 for c in ("overlap_total", "overlap_inputs", "overlap_router")):
        fail.append(f"roundtrip routing changed (layer {r['layer']}, {r['probe']})")
    if r["ckpt"] == "frozen" and float(r["overlap_router"]) != 1.0:
        fail.append(f"frozen router-only overlap != 1 (layer {r['layer']}, {r['probe']})")
if len(rt) != 3 * 3 * 2: fail.append(f"expected 18 routing rows (3 ckpts x 3 probes x 2 layers), got {len(rt)}")
print("EVAL TINY TEST:", "PASS" if not fail else "FAIL " + "; ".join(fail))
sys.exit(1 if fail else 0)
PY
echo "end: $(date)"
