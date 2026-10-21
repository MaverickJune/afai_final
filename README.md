# Disentangling SFT and RL effects in an MoE model

CPSC 4880/5880 final project (Shuang Song, Wonjun Bang). We post-train **Qwen3-30B-A3B-Base** on
OpenR1-Math-220k with SFT and with GRPO, and analyse how each changes the model's weights, experts and routing.

**Current state:** SFT (1 epoch, 976 steps) is done and analysed; GRPO and the SFT↔RL orderings come next.
Results: [`experiments/eval/RESULTS_SFT.md`](experiments/eval/RESULTS_SFT.md) (figures in `experiments/eval/figures/`).

## Layout

| Path | Contents |
|---|---|
| `jobs/*.sbatch` | Slurm jobs (training, tests, evaluation) |
| `jobs/configs/` | Run settings (SFT, GRPO, evaluation paths) |
| `jobs/lib/` | Training launcher (`run_30b.sh`, `sft_ckpt.py`) and data prep |
| `jobs/lib/eval/` | Analysis: benchmarks, KL + routing drift, parameter change, expert specialization |
| `experiments/<run>/SETTINGS.md` | All settings of each training run |
| `experiments/eval/` | Evaluation plan, results, figures |

## Run

Requires the Yale Bouchet cluster: training uses the [slime](https://github.com/THUDM/slime) container,
analysis uses a conda env. Setup is in `env_guide.txt` / `SETUP.md`. Submit from the repo root:

```bash
# training (8 GPUs): data prep -> small mechanics test -> full run
sbatch jobs/prep_openr1_sft.sbatch;  sbatch jobs/test_sft_ckpt_tiny.sbatch;  sbatch jobs/sft_openr1_30b.sbatch
sbatch jobs/prep_openr1_grpo.sbatch; sbatch jobs/test_grpo_ckpt_tiny.sbatch; sbatch jobs/grpo_openr1_30b.sbatch

# evaluation (1 GPU or CPU per job), for run "sft" (or "grpo")
sbatch jobs/prep_eval_bench.sbatch                      # benchmarks + probe texts
sbatch jobs/test_eval_tiny.sbatch                       # known-answer test of the analysis code
sbatch jobs/eval_param_diff.sbatch sft                  # weight change per layer / module / expert
sbatch jobs/eval_probe.sbatch sft                       # KL and routing drift per checkpoint
sbatch jobs/eval_specialization.sbatch sft              # are updates concentrated on math experts?
sbatch jobs/eval_bench.sbatch sft_final                 # benchmarks (also: base, grpo_final)
sbatch jobs/plot_sft_results.sbatch                     # figures
```

Each step should wait for the one before it (e.g. `sbatch --dependency=afterok:<jobid> ...`).

## Inspect without a cluster

The results are plain files: `experiments/eval/results/` (CSV / JSON summaries) and `experiments/eval/figures/`.
The scoring helpers can be tested anywhere: `python jobs/lib/eval/test_scoring.py`.
