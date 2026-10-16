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
#   kl_math500: round trip (separately loaded, identical weights) -> KL 0; trained models -> KL > 0, finite
# Needs $BENCH_DIR/math500.jsonl (jobs/prep_eval_bench.sbatch).
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
python jobs/lib/eval/kl_math500.py --base "$TINY/hf_tiny_moe" --data "$BENCH_DIR/math500.jsonl" --out "$OUT/kl" \
    --limit 20 --ckpt roundtrip="$TINY/hf_roundtrip" --ckpt frozen="$TINY/sft_frozen_hf" --ckpt control="$TINY/sft_control_hf"

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
kl = {r["ckpt"]: r for r in csv.DictReader(open(f"{out}/kl/kl_math500.csv"))}
if abs(float(kl["roundtrip"]["kl_base_ckpt_token_mean"])) > 1e-6: fail.append("roundtrip KL not 0")
for ck in ("frozen", "control"):
    v = float(kl[ck]["kl_base_ckpt_token_mean"])
    if not (v > 0 and math.isfinite(v)): fail.append(f"{ck} KL not positive/finite: {v}")
print({ck: (float(r["kl_base_ckpt_token_mean"]), float(r["kl_ckpt_base_token_mean"])) for ck, r in kl.items()})
print("EVAL TINY TEST:", "PASS" if not fail else "FAIL " + "; ".join(fail))
sys.exit(1 if fail else 0)
PY
echo "end: $(date)"
