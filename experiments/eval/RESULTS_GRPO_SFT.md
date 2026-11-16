# GRPO and GRPO → SFT: what the results show, and how to read them

Stages analysed, in the RL → SFT order:
1. **GRPO**: Qwen3-30B-A3B-Base → GRPO on OpenR1-Math-220k prompts (rows `iloc[0::3]`, 31,245 prompts, 1 epoch,
   976 steps, 32 prompts × 8 responses per step, constant LR 1e-6, router trainable; settings in
   `experiments/grpo_openr1math_idx3k0/SETTINGS.md`).
2. **SFT after GRPO**: the final GRPO model → the same SFT recipe as base → SFT (rows `iloc[1::3]`, 976 steps,
   LR 1e-5 cosine to 1e-6; settings in `experiments/sft_openr1math_idx3k1/SETTINGS.md`, section "GRPO → SFT").

Ten checkpoints per stage, at steps 1, 2, 5, 10, 21, 46, 98, 211, 454 and 976, as in the SFT analysis.
The measures are defined in `RESULTS_SFT.md` (sections 1–5) and `PLAN.md`; they are not repeated here.

| Comparison | Checkpoints | Measured against | Result / figure folder |
|---|---|---|---|
| GRPO | GRPO steps | base model | `grpo` |
| SFT after GRPO | GRPO → SFT steps | GRPO final (step 976) | `grpo_sft` |
| GRPO → SFT total | GRPO → SFT steps | base model | `grpo_sft_vs_Qwen3-30B-A3B-Base` |
| SFT (reference, colleague's run) | SFT steps | base model | `sft` |

Figures: `experiments/eval/figures/<folder>/` (fig1–fig8 per comparison) and `experiments/eval/figures/compare/`
(c1–c5, all comparisons on shared axes). Numbers: `experiments/eval/results/`.

**Status (Nov 16):** all analyses are complete for both stages. Benchmarks are complete for GRPO final and
GRPO → SFT final (math + MMLU + MMLU-Pro). GPQA was not run (gated dataset). SFT-from-base MMLU / MMLU-Pro is still
missing, so those two benchmarks cannot compare the two SFT models.

---

## 0. Things to know before reading any number

- **GRPO step 1 is not identical to the base model.** GRPO has no warm-up (constant LR 1e-6), so its first update is
  real but tiny: 1–2×10⁻⁷ per weight entry, KL 0.001–0.003 nats/token. Its routing change at step 1 (1.4–2.6%) is the
  near-tie noise floor described in `RESULTS_SFT.md`: treat routing changes below ~2.5% on math text and ~1.5% on
  general text as noise.
- **SFT-after-GRPO step 1 is identical to GRPO final** (same warm-up as base → SFT), so every metric is 0 there.
  Consistency check, passed exactly: the GRPO → SFT total at step 1 equals GRPO's values at step 976 for weights, KL
  and routing.
- **The size of GRPO vs SFT is partly a learning-rate choice.** GRPO used slime's example LR (1e-6, constant); SFT
  used 1e-5 with cosine decay to 1e-6, about 5× more per step on average. "GRPO changes the model less" holds for
  these two runs; it is not a pure effect of the algorithm. Patterns (where and how the model changes) are less
  affected by this than magnitudes.

---

## 1. Output drift: KL divergence (`figures/compare/c1`, per-run fig1, `results/probe/*/kl.csv`)

KL(reference ‖ model), nats per token:

| GRPO vs base | step 1 | step 10 | step 46 | step 98 | step 211 | step 454 | step 976 |
|---|---|---|---|---|---|---|---|
| Held-out OpenR1 math | 0.001 | 0.002 | 0.003 | 0.005 | 0.009 | 0.023 | 0.045 |
| MATH-500 | 0.002 | 0.003 | 0.011 | 0.019 | 0.032 | 0.056 | 0.138 |
| General text | 0.003 | 0.004 | 0.005 | 0.006 | 0.008 | 0.020 | 0.036 |

| Step 976 | SFT vs base | GRPO vs base | SFT after GRPO vs GRPO final | GRPO → SFT total vs base |
|---|---|---|---|---|
| Held-out OpenR1 math | 0.252 | 0.045 | 0.248 | 0.252 |
| MATH-500 | 0.216 | 0.138 | 0.276 | 0.232 |
| General text | 0.305 | 0.036 | 0.282 | 0.284 |

**How to read it.**
- **GRPO drifts slowly and steadily, and is still accelerating at step 976** (MATH-500: 0.056 → 0.138 between steps
  454 and 976). There is no early jump as in SFT. More GRPO steps would move the model further.
- **GRPO's drift is largest on MATH-500 and small elsewhere.** On general text (0.036) it is 8× smaller than SFT's
  (0.305): GRPO leaves non-math predictions nearly unchanged, the opposite of SFT, whose largest drift is on general
  text. On held-out R1 solutions it is 5.5× smaller than SFT's: GRPO does not learn to imitate R1's long reasoning,
  which it never sees.
- **SFT after GRPO drifts like SFT from base.** Measured from GRPO final, its curves nearly coincide with SFT from
  base on held-out math and general text (c1). On MATH-500 it ends higher (0.276 vs 0.216).
- **The final GRPO → SFT model is about as far from base as the SFT-only model** (0.252 / 0.232 / 0.284 vs
  0.252 / 0.216 / 0.305).
- **The first SFT steps partly reverse GRPO's drift.** Measured from base, KL on held-out math falls from 0.045 to
  0.024 by SFT step 10, and on general text from 0.036 to 0.024 by step 21, before SFT's own drift takes over
  (`figures/grpo_sft_vs_Qwen3-30B-A3B-Base/fig1`). By step 10, when the held-out-math KL has already halved, SFT has
  moved the weights by only ~10% of GRPO's change (section 3), so GRPO's effect on these predictions is easy to undo.
  MATH-500 shows no such reversal.

---

## 2. Routing drift (`figures/compare/c2`, per-run fig2–3, `results/probe/*/routing.csv`)

Share of top-8 expert choices that changed at step 976, mean over layers ("inputs" and "router" are the
counterfactual split defined in `RESULTS_SFT.md` section 2):

| Step 976 | SFT vs base | GRPO vs base | SFT after GRPO vs GRPO final | GRPO → SFT total vs base |
|---|---|---|---|---|
| Held-out OpenR1 math: overall / inputs / router | 14.4 / 14.0 / 2.3% | 6.4 / 6.4 / 0.2% | 14.0 / 13.7 / 2.3% | 14.1 / 13.7 / 2.3% |
| MATH-500 | 16.1 / 16.0 / 1.5% | 9.1 / 9.1 / 0.2% | 16.4 / 16.3 / 1.5% | 15.5 / 15.4 / 1.5% |
| General text | 14.9 / 14.9 / 0.8% | 3.7 / 3.7 / 0.1% | 13.1 / 13.1 / 0.8% | 13.4 / 13.4 / 0.8% |

**How to read it.**
- **GRPO changes 4–9% of expert choices, 2–4× less than SFT.** On general text (3.7%) this is only ~2 points above
  the noise floor.
- **GRPO's routing change comes entirely from changed hidden states.** The router weights themselves change choices
  by ≤ 0.2%, even though they moved by 0.1% (section 3). The SFT conclusion holds even more strongly for GRPO:
  freezing the router would not freeze routing.
- **By depth (`figures/grpo/fig3`):** GRPO's routing change grows steadily with depth, from 3.8% (layers 1–11) to
  9.3% (layers 36–47), peaking at 12.2% in layer 46. Expert load barely changes (Jensen-Shannon divergence ≤ 0.002).
- **SFT after GRPO routes like SFT from base:** same overall size, the same router-weight part (2.3 / 1.5 / 0.8%),
  the same depth profile (peak 20.9% in layer 46; router-weight part up to 7.1% in layer 45). General text changes
  slightly less (13.1% vs 14.9%).

---

## 3. Parameter change (`figures/compare/c3`, per-run fig4–5, `results/param_diff/*/param_diff_modules.csv`)

Relative change ‖ΔW‖ / ‖W_ref‖ at step 976, mean over the 48 layers (%):

| Module | SFT vs base | GRPO vs base | SFT after GRPO vs GRPO final | GRPO → SFT total vs base |
|---|---|---|---|---|
| Attention q / k / v / o | 0.80 / 0.83 / 0.70 / 0.76 | 0.10 / 0.11 / 0.10 / 0.11 | 0.81 / 0.84 / 0.70 / 0.77 | 0.81 / 0.85 / 0.71 / 0.77 |
| Router | 0.74 | 0.10 | 0.74 | 0.74 |
| Experts gate / up / down | 0.76 / 0.77 / 0.76 | 0.09 / 0.09 / 0.09 | 0.76 / 0.77 / 0.76 | 0.76 / 0.77 / 0.76 |
| Embedding / LM head | 0.27 / 0.78 | 0.026 / 0.027 | 0.27 / 0.79 | 0.27 / 0.78 |

**How to read it.**
- **GRPO moves the weights ~8× less than SFT** (RMS per entry 2.2–2.8×10⁻⁵ vs 1.8–2.0×10⁻⁴). Roughly the LR ratio;
  see section 0.
- **Timing (c3):** GRPO's change grows steadily, about as steps^0.7 from step 98 on, with no sign of saturating.
  SFT's growth slows from step ~200 and nearly stops after ~450 as its LR decays.
- **Where GRPO writes (`figures/grpo/fig4`):**
  - the **router changes mostly in late layers**, as in SFT: 0.06% in layers 0–11, 0.17% in layers 36–47, up to
    0.22% in layer 46;
  - expert MLPs are uniform across depth, slightly smaller in layers 36–47 (0.08% vs 0.09–0.10%);
  - **the LM head barely changes** (0.027%, the same as the embedding). SFT changes the LM head 3× more than its
    embedding (0.78%); the SFT/GRPO ratio is ~29× for the LM head vs 8–10× elsewhere. SFT rewrites the output layer,
    GRPO does not.
- **SFT after GRPO changes the weights like SFT from base:** at step 976 every module agrees within ~0.01
  percentage points, and the trajectories coincide from step ~21 on (the blue and green lines in c3 overlap; before
  that, SFT after GRPO moves slightly faster).
- **GRPO's weight change survives SFT: SFT neither undoes nor reinforces it.** From the squared norms
  ‖total‖² = ‖ΔGRPO‖² + ‖ΔSFT‖² + 2‖ΔGRPO‖‖ΔSFT‖·cos, the cosine between GRPO's update and the following SFT update
  is −0.004 over all layer weights, and |cos| ≤ 0.01 for every attention, router and expert module type. A strongly
  negative value would have meant SFT reverts GRPO; it does not. (In this many dimensions unrelated updates are
  near-orthogonal too, so cos ≈ 0 means "not reverted", not "deliberately independent".)
- Together with section 1: **GRPO's change stays in the weights, but its effect on the outputs is largely covered by
  SFT's 8× larger update.**

---

## 4. Expert specialization (`figures/compare/c4`, per-run fig6–7, `results/specialization/`)

Math affinity is measured on each comparison's reference model (base, or GRPO final for SFT after GRPO).

| Step 976 | SFT vs base | GRPO vs base | SFT after GRPO vs GRPO final | GRPO → SFT total vs base |
|---|---|---|---|---|
| Rank correlation, affinity vs update size (mean over layers) | +0.35 | **+0.43** | +0.36 | +0.36 |
| … in layers 12–35 | +0.44 | **+0.53** | +0.45 | +0.45 |
| Layers with a positive correlation | 47 / 48 | 47 / 48 | 47 / 48 | 47 / 48 |
| Update energy in the 10% most math-affine experts (10% = no preference) | 11.2% | 11.8% | 11.3% | 11.2% |
| Gini of update energy over experts | 0.20 | 0.24 | 0.20 | 0.20 |

**How to read it.**
- **The SFT pattern holds for GRPO and for SFT after GRPO** (fig6): experts with affinity above about −1 get a
  typical update (binned means ≈ 1.0–1.07 × the layer median), and the correlation comes from the low end. Experts
  the reference model rarely uses on math get about half the typical update (lowest 5% bin ≈ 0.50–0.55). In no
  comparison are the updates concentrated on a math-specialized subset.
- **GRPO's updates follow math usage slightly more closely** (+0.43 vs +0.35, top-10% share 11.8% vs 11.2%, Gini 0.24
  vs 0.20). One plausible reason: GRPO trains only on its own math generations, and an expert receives gradient only
  for tokens routed to it.
- **Over training (c4):** GRPO's correlation is +0.32 from step 1, dips to +0.16 around steps 21–46, and climbs to
  +0.43. Before step ~100 the updates are tiny, so the early points are noisy. SFT after GRPO follows SFT from base
  step by step.

---

## 5. Benchmarks (`figures/compare/c5`, per-run fig8, `results/<model>/`)

Same protocol as `RESULTS_SFT.md` section 5 (temperature 0.6, top-p 0.95, up to 30k response tokens; 16k for GSM8K /
MMLU / MMLU-Pro). Accuracy = mean over samples, ± 95% CI over questions; pass@n in brackets.

| Benchmark | Base | SFT | GRPO | GRPO → SFT |
|---|---|---|---|---|
| MATH-500 (500 × 4) | 75.4 ± 3.0 [88.6] | 91.3 ± 2.1 [95.6] | 85.7 ± 2.7 [91.6] | **91.4 ± 2.1** [96.2] |
| GSM8K (1,319 × 1) | 86.9 ± 1.8 | 94.2 ± 1.3 | **95.1 ± 1.2** | 94.6 ± 1.2 |
| AIME 2024 (30 × 16) | 16.7 ± 8.1 [50.0] | 50.0 ± 13.5 [83.3] | 26.9 ± 13.4 [46.7] | **50.4 ± 13.8** [83.3] |
| AIME 2026 (30 × 16) | 10.8 ± 8.8 [33.3] | 45.8 ± 13.6 [76.7] | 18.5 ± 10.8 [46.7] | **47.9 ± 14.1** [76.7] |
| HMMT Feb 2026 (33 × 16) | 5.1 ± 5.2 [15.2] | 26.3 ± 13.2 [45.5] | 12.3 ± 8.5 [27.3] | **26.7 ± 12.9** [60.6] |
| MMLU (14,042 × 1) | 66.1 ± 0.8 | not run | 78.9 ± 0.7 | **84.3 ± 0.6** |
| MMLU-Pro (12,032 × 1) | 53.0 ± 0.9 | not run | 58.2 ± 0.9 | **74.7 ± 0.8** |

Behaviour of the responses (base and SFT: `RESULTS_SFT.md`):

| Benchmark | mean response tokens: GRPO / GRPO → SFT | hit the token limit | no boxed answer |
|---|---|---|---|
| MATH-500 | 1,120 / 3,856 | 1.2 / 2.1% | 1.3 / 2.0% |
| GSM8K | 348 / 1,325 | 0.0 / 1.0% | 0.0 / 1.0% |
| AIME 2024 | 6,991 / 14,014 | 16.0 / 15.4% | 16.7 / 15.0% |
| AIME 2026 | 5,538 / 14,183 | 12.1 / 17.5% | 12.3 / 16.7% |
| HMMT Feb 2026 | 4,868 / 18,276 | 10.2 / 28.8% | 10.4 / 27.5% |
| MMLU | 450 / 1,788 | 0.1 / 3.1% | 4.2 / 0.6% |
| MMLU-Pro | 941 / 3,912 | 1.5 / 8.4% | 17.0 / 3.2% |

**How to read it.**
- **GRPO improves every math benchmark, by less than SFT on hard problems.** It gains +10 points on MATH-500, +8 on
  GSM8K (the best GSM8K score of the four models) and +7 to +10 on AIME / HMMT. On the contests that is a fifth to a
  third of SFT's gain; on MATH-500 about two thirds.
- **GRPO raises pass@16 much less than accuracy.** AIME 2024 pass@16 does not improve (50.0 → 46.7%); AIME 2026 and
  HMMT rise modestly (33 → 47%, 15 → 27%). SFT reaches 77–83% on AIME. GRPO mainly makes answers the base model can
  already reach more frequent, while SFT makes new problems solvable. This matches Yue et al.'s observation for RL;
  with 30–33 questions, pass@16 is coarse.
- **GRPO keeps responses short:** MATH-500 1,120 tokens (base 1,053), AIME and HMMT 1.6–2.4× the base length, against
  4–6× for SFT. It learned the answer format: no missing boxed answers on GSM8K, 1.3% on MATH-500.
- **GRPO does not forget; it gains on general benchmarks** (MMLU 66.1 → 78.9, MMLU-Pro 53.0 → 58.2). Part of this is
  formatting: on MMLU the no-answer rate drops from 13.0% to 4.2%. The gain remains among answered questions
  (accuracy ÷ (1 − no-answer rate)): MMLU 76.0 → 82.4%, MMLU-Pro 63.0 → 70.2%. On MMLU-Pro GRPO still misses the
  boxed letter 17% of the time.
- **GRPO → SFT matches SFT from base on every math benchmark.** All differences are ≤ 2.1 points, far inside the
  intervals: 91.4 vs 91.3 on MATH-500, 47.9 vs 45.8 on AIME 2026. pass@16 is equal on both AIME sets. It is higher on
  HMMT (60.6 vs 45.5%), from 33 questions and a different benchmark version (section 7). Response lengths are the
  same as SFT-only: **SFT sets the final style, and GRPO before it brings no measurable math benefit.**
- **GRPO → SFT has the best MMLU / MMLU-Pro of the models measured** (84.3 / 74.7, with almost no format failures),
  but SFT-only was not measured on them, so this cannot be attributed to the GRPO stage yet.
- On GSM8K, SFT after GRPO adds nothing (95.1 → 94.6) while making responses 4× longer.

---

## 6. Summary (as facts, for your own write-up)

1. GRPO (LR 1e-6, 976 steps) changes the model much less than SFT: ~8× less in weights, 1.6–8× less in KL, 2–4× less
   in routing. Its drift is concentrated on math text; general-text predictions and routing barely change. Its
   routing change is entirely caused by changed hidden states, not by the router weights. Part of the size gap is the
   10× smaller learning rate.
2. GRPO's change is still growing at step 976. It raises math accuracy by 7–10 points with short responses and
   modest pass@16 gains, and raises MMLU / MMLU-Pro (no forgetting).
3. SFT after GRPO is nearly indistinguishable from SFT from base: same weight-change size, location and timing, same
   KL and routing drift, same specialization pattern, same benchmark scores and response lengths.
4. GRPO's weight change is not reverted by SFT (cosine ≈ 0 with the SFT update), but its effect on the outputs is
   covered by SFT's 8× larger update. The final model is as far from base as SFT-only, and the first 10–20 SFT steps
   undo a third to a half of GRPO's KL on held-out math and general text.
5. Neither GRPO nor SFT concentrates its updates on math-specialized experts. Both give smaller updates only to
   experts rarely used on math; GRPO follows math usage slightly more closely (+0.43 vs +0.35).

## 7. Limitations to state

- One run per stage, one seed; no estimate of run-to-run variation. Routing changes below ~2.5% are noise.
- GRPO and SFT differ in more than the algorithm: LR (1e-6 constant vs 1e-5 cosine), Adam β2 (0.98 vs 0.95), batch
  (256 sampled responses vs 32 R1 solutions per step), and data (GRPO sees prompts and its own generations; SFT sees
  R1 solutions). Magnitude comparisons reflect these choices too.
- GRPO trained with responses capped at 8,192 tokens; evaluation allows 30k.
- The SFT-only and GRPO → SFT final models were not compared directly (the SFT-only checkpoint is on another
  account's scratch); their similarity is inferred from each one's distance to base and from the benchmarks.
- Benchmark data versions: AIME 2026 and HMMT Feb 2026 for GRPO / GRPO → SFT were downloaded on Nov 15 (MathArena
  revisions `d61f486` / `589a53f`); base and SFT used the earlier revisions (`d2de22f` / `02fba4f`). The HMMT upload
  changed some problem or answer text (one question ≈ 3 points); the AIME 2026 file was re-saved with the same
  content size.
- MMLU / MMLU-Pro for SFT-only are missing; GPQA was not run. Probe analyses use teacher-forced text, not the models'
  own generations.
