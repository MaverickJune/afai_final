# SFT stage: what the results show, and how to read them

Stage analysed: Qwen3-30B-A3B-Base → full SFT on OpenR1-Math-220k (rows `iloc[1::3]`, 31,244 R1 solutions, 1 epoch,
976 steps, router trainable; settings in `experiments/sft_openr1math_idx3k1/SETTINGS.md`). Every comparison is
**SFT checkpoint vs. the base model**. Ten checkpoints were analysed at steps 1, 2, 5, 10, 21, 46, 98, 211, 454
and 976 (log-spaced, so early training is dense).

Figures: `experiments/eval/figures/`. Numbers: `experiments/eval/results/`. Method details: `experiments/eval/PLAN.md`.

**Status (Oct 19):** analyses 1–4 below are complete. Benchmarks are complete for the base model (math + MMLU +
MMLU-Pro) and for the SFT model's math set. **SFT MMLU / MMLU-Pro is still running (job 30040147)**; it was interrupted
twice by an unexplained scratch "Disk quota exceeded" error (scratch is at 16% of quota). GPQA needs a HuggingFace
login. Nothing from the GRPO side exists yet.

---

## 0. Two things to know before reading any number

- **Step 1 equals the base model.** At step 1 the warm-up learning rate is 2×10⁻⁷. That update is smaller than bf16
  can represent, so the saved weights are bit-identical to the base, and every metric at step 1 is exactly 0.
  This is correct, not a bug.
- **Routing has a noise floor of about 2.5%.** At step 2 the weights moved by only ~10⁻⁸ per entry, yet ~2.5% of
  expert choices already flipped. For many tokens the 8th and 9th experts have nearly equal router scores, so a
  tiny perturbation swaps them. Treat routing differences below ~3% as noise (grey band in Fig. 2).

---

## 1. Output drift: KL divergence (Fig. 1, `results/probe/sft/kl.csv`)

**What is measured.** Fixed text is fed through both models (teacher forcing, so neither model generates anything).
At every token we compare the two next-token distributions with KL(base ‖ SFT), in nats per token. 0 means
identical predictions; larger means the SFT model predicts differently. Three probe texts are used:
- *held-out OpenR1 math*: 300 R1 solutions from the third of OpenR1 that neither SFT nor GRPO trains on, so
  in-distribution but unseen;
- *MATH-500*: problems with human-written solutions, the probe the Rajani et al. paper uses;
- *general text*: 300 MMLU questions, i.e. non-math knowledge text.

| Probe | step 10 | step 21 | step 46 | step 98 | step 211 | step 976 |
|---|---|---|---|---|---|---|
| Held-out OpenR1 math | 0.005 | 0.083 | 0.150 | 0.192 | 0.212 | 0.252 |
| MATH-500 | 0.002 | 0.027 | 0.176 | 0.209 | 0.186 | 0.216 |
| General text | 0.004 | 0.024 | 0.053 | 0.089 | 0.223 | 0.305 |

**How to read it.**
- Almost nothing happens until step 10 (learning-rate warm-up). The big jump on math text is between steps 10 and 46,
  and math KL is largely saturated by step ~100. This matches the paper's observation that SFT drifts early.
- **General text drifts later but further.** It lags math at first, overtakes it around step 211, and is still rising
  at the end (0.30 vs 0.22–0.25). SFT on math changes predictions on non-math text more than on math text. That fits
  the "SFT replaces" view and the forgetting concern; benchmarks will show whether it costs accuracy.
- Absolute size: 0.25 nats/token is a large shift. The SFT model is far more confident on R1-style reasoning, since
  that is what it was trained on.

---

## 2. Routing drift (Figs. 2–3, `results/probe/sft/routing.csv`)

**What is measured.** For every token and every one of the 48 MoE layers, we compare which 8 of the 128 experts
the router picks. "Changed %" = the share of those 8 choices that differ from the base model, averaged over tokens.

To find *why* routing changed, a third, counterfactual routing is computed: the **old (base) router weights applied
to the SFT model's hidden states**. This splits the change:
- **inputs only** = old router on new hidden states vs. base routing. The router did not change, but the
  representation arriving at it did;
- **router weights only** = new router vs. old router, on the *same* new hidden states. This is the change due to
  the router's own weights; it is exactly 0 if the router is frozen.

| Probe (step 976) | changed overall | from inputs only | from router weights only |
|---|---|---|---|
| Held-out OpenR1 math | 14.4% | 14.0% | 2.3% |
| MATH-500 | 16.1% | 16.0% | 1.5% |
| General text | 14.9% | 14.9% | 0.8% |

The two parts don't add up exactly: they are overlaps of sets, not additive shares.

**How to read it.**
- About **1 in 7 expert choices changed** by the end of SFT, and routing is still slowly changing at step 976 (Fig. 2).
- **Almost all of the routing change comes from changed inputs, not from the router.** Holding the router at its
  base weights reproduces nearly the whole change. The router weights themselves move routing by only 1–2%.
  This matters for the router-freezing ablation: **freezing the router would not freeze routing.** Routing would still
  shift by ~14% through the hidden states.
- **By depth (Fig. 3):** layer 0 barely changes (its input is the token embedding). Change is ~13–15% across the middle
  and peaks at ~19% in layers 44–46. The router-weight part grows with depth, from ~1% to ~7% in layers 44–46. The last
  layers are where the router itself learns most.
- Expert *load* (how many tokens each expert receives) barely changes: Jensen-Shannon divergence ≈ 0.005. SFT
  re-assigns tokens among experts while keeping overall usage balanced.
- The timing mirrors the KL curve. Math text routing moves first (steps 10–46), general text catches up by step 211.

---

## 3. Parameter change (Figs. 4–5, `results/param_diff/sft/param_diff_modules.csv`)

**What is measured.** For every weight matrix: the relative change ‖W_SFT − W_base‖ / ‖W_base‖, the paper's
"normalised Frobenius norm". For experts, the 128 experts of a layer are pooled.

**How to read it.**
- **Overall size:** ~0.7–0.9% relative change for attention and expert weights after 976 steps (Fig. 4). The RMS change
  per entry is ~2×10⁻⁴ (Fig. 5).
- **Timing (Fig. 5):** the change grows ~10× per log-spaced checkpoint until step ~46, then slowly. All module types
  move at nearly the same rate per entry.
- **Where (Fig. 4):**
  - attention q/k change somewhat more in the last third of the network (k up to ~1.1%);
  - **expert MLP changes are flat across depth** (~0.75% in every layer group). This differs from the paper's dense
    model, where SFT changed *mid-layer* MLPs most. In this MoE the update is spread evenly over depth;
  - **the router changes little early and much more late:** 0.44% in layers 0–11, 1.2% in layers 36–47, up to 1.6% at
    layer 46. This agrees with the routing result: late-layer routers are the ones that learn;
  - layer 0 attention (k, v) is an outlier with the largest change in its column.

---

## 4. Expert specialization: are SFT updates localized to "math experts"? (Figs. 6–7, `results/specialization/`)

**What is measured.**
- *Math affinity* of each expert = log(its share of tokens on the math probe) − log(its share on the general probe),
  measured on the **base** model. Positive means used more on math.
- *Update size* = the expert's relative weight change, divided by the median of its layer, so that layers are comparable.
  1.0 means a typical update for that layer.

**Results (step 976).**
- The rank correlation between affinity and update size is **+0.35** on average over layers. It is positive in 47 of 48
  layers and strongest in the middle layers (+0.43 to +0.45 in layers 12–35).
- **But the 10% most math-affine experts receive only 11.2% of the update energy**, where 10% would mean no preference.
  Gini over experts ≈ 0.2, so the updates are spread fairly evenly.
- **The shape of the relationship (Fig. 6) is the important part.**

| Math affinity | experts | mean update / layer median |
|---|---|---|
| < −2 (used mostly on general text) | 1,316 | 0.71 |
| −2 to 0 | 2,061 | 0.99 |
| 0 to 2 | 1,950 | 1.02 |
| > 2 (used mostly on math) | 817 | 1.00 |

**How to read it.** The positive correlation does **not** mean "math experts get bigger updates". Experts with
math affinity above about −1 all get the same, typical update. The correlation comes entirely from the low end:
**experts that the base model rarely uses on math get ~30% smaller updates.** The likely explanation: an expert only
receives gradient for the tokens routed to it, and SFT data is all math. So SFT updates are **distributed across every
expert that math text reaches, not concentrated in a specialized subset**. Only the general-text-only experts are
partly left alone.

- Over training (Fig. 7), this pattern settles after step ~100. Before step 46 the update sizes are tiny and noisy, so the
  early points there carry little meaning.
- Caveat: the general probe is short (31k tokens vs. 1.03M for the math probe), so affinities are noisy. The vertical
  stripes in Fig. 6 near affinity −3.5 are experts that received no math tokens at all; their affinity comes only from
  smoothing.

---

## 5. Benchmarks: base vs. SFT (Fig. 8, `results/base/`, `results/sft_final/`)

Zero-shot, base chat template, "reason step by step … `\boxed{}`" prompt, temperature 0.6, top-p 0.95, up to 30k
response tokens (16k for GSM8K / MMLU / MMLU-Pro). Accuracy is the mean over samples (avg@n) with a 95% confidence
interval over questions. pass@n = probability that at least one of the n samples is correct.

| Benchmark | questions × samples | Base | SFT (step 976) | pass@n: base → SFT |
|---|---|---|---|---|
| MATH-500 | 500 × 4 | 75.4 ± 3.0% | **91.3 ± 2.1%** | 88.6 → 95.6% |
| GSM8K | 1,319 × 1 | 86.9 ± 1.8% | **94.2 ± 1.3%** | — |
| AIME 2024 | 30 × 16 | 16.7 ± 8.1% | **50.0 ± 13.5%** | 50.0 → 83.3% |
| AIME 2026 | 30 × 16 | 10.8 ± 8.8% | **45.8 ± 13.6%** | 33.3 → 76.7% |
| HMMT Feb 2026 | 33 × 16 | 5.1 ± 5.2% | **26.3 ± 13.2%** | 15.2 → 45.5% |
| MMLU | 14,042 × 1 | 66.1 ± 0.8% | *running* | — |
| MMLU-Pro | 12,032 × 1 | 53.0 ± 0.9% | *running* | — |

Behaviour of the responses (base → SFT):

| Benchmark | mean response tokens | hit the token limit | no boxed answer |
|---|---|---|---|
| MATH-500 | 1,053 → 3,909 | 1.6 → 2.5% | 3.2 → 2.1% |
| GSM8K | 402 → 1,326 | 0.6 → 1.2% | 3.6 → 1.1% |
| AIME 2024 | 2,909 → 14,027 | 6.5 → 17.1% | 11.9 → 16.5% |
| AIME 2026 | 2,429 → 13,813 | 4.2 → 17.3% | 12.7 → 17.1% |
| HMMT Feb 2026 | 2,988 → 17,947 | 6.8 → 29.0% | 13.6 → 28.0% |
| MMLU | 696 → *running* | 2.2% (base) | 13.0% (base) |
| MMLU-Pro | 917 → *running* | 2.5% (base) | 15.9% (base) |

**How to read it.**
- **SFT strongly improves math on every benchmark**, far outside the confidence intervals: +16 points on MATH-500,
  +33 to +35 on the AIME sets, +21 on HMMT. The gains hold on AIME 2026 and HMMT Feb 2026, which postdate the model's
  release, so they are not memorisation of test problems.
- **pass@n rises as well** (AIME 2026: 33 → 77% at n = 16). SFT makes problems solvable that the base model almost never
  solves in 16 tries, rather than only making already-reachable answers more frequent. This is the "SFT adds
  capability" side of the SFT-vs-RL question; the GRPO run will show whether RL behaves differently (Rajani et al. and
  Yue et al. would predict mainly higher accuracy at similar pass@n).
- **Responses get 4–6× longer**: R1-style long reasoning, learned from the SFT data. On the hard contests, 17–29% of
  SFT responses hit the 30k-token limit, and the no-answer rate is about the same size, consistent with cut-off
  responses never reaching a final answer. So the SFT scores on AIME/HMMT are slightly conservative.
- **Base MMLU / MMLU-Pro come with high no-answer rates (13–16%).** The base model often does not put a letter in
  `\boxed{}`, which counts as wrong. Part of any SFT-vs-base difference on these two benchmarks will be answer formatting,
  not knowledge. Compare the no-answer rates before drawing a forgetting conclusion.
- AIME/HMMT have only 30–33 questions, so their intervals are wide (±13 points). MATH-500 and GSM8K give the precise
  in-domain comparison.

---

## 6. Summary of the SFT stage (as facts, for your own write-up)

1. SFT changes the output distribution early (steps 10–46 on math) and keeps changing it on general text until the end.
   At the end the drift on general text is larger than on math (KL 0.30 vs 0.22–0.25).
2. ~14–16% of expert-routing decisions change. ~95%+ of that is caused by changed hidden states, not by the router
   weights, which matter only in the last layers (up to ~7%).
3. Weights change ~0.7–0.9% (relative). Expert MLPs change evenly across depth (unlike the dense-model finding of
   mid-layer emphasis); the router changes mostly in late layers.
4. Expert updates are distributed over all experts that math text uses. They are not localized to the most
   math-specialized experts; only general-text-only experts get smaller updates.
5. Math accuracy rises sharply (MATH-500 75 → 91%, AIME 2026 11 → 46%), pass@16 rises too (33 → 77% on AIME 2026),
   and responses become 4–6× longer. The out-of-domain check (MMLU / MMLU-Pro) is pending.

## 7. Limitations to state

- One run, one seed; no repeat to estimate run-to-run variation. Routing changes below ~3% are within near-tie noise.
- Analyses use probe text (teacher forcing), not the model's own generations.
- Math affinity is measured with a small general-text probe.
- Benchmarks: one sampling seed; small contests (30–33 questions) give wide intervals; 17–29% of SFT responses on the
  hardest contests are cut off at 30k tokens; base-model MMLU scores include formatting failures.
- Comparisons with GRPO (the main question) need the GRPO run. Every analysis here re-runs unchanged on it
  (`jobs/eval_*.sbatch grpo`), with the same checkpoint steps, so the curves will be directly comparable.
