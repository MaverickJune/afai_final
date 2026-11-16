"""Figures of the post-training analysis (afai_env, CPU). Reads the CSVs / summaries in experiments/eval/results/
(written by the eval_* jobs) and, for figure 6 only, the per-expert table $SCR/eval/specialization_RUN_experts.csv.gz.

usage: python plot_results.py RESULTS_DIR FIG_DIR RUN [EXPERTS_CSV_GZ]
  RUN = sft | grpo | grpo_sft | grpo_sft_vs_Qwen3-30B-A3B-Base   one run vs its reference -> FIG_DIR/RUN/fig1..fig8
        compare                                                  all runs on shared axes  -> FIG_DIR/compare/c1..c5
  Figure 6 is skipped when EXPERTS_CSV_GZ does not exist (e.g. the base -> SFT table is on another account's scratch).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Reference palette (dataviz skill, light mode): first three categorical slots, ink, grid.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
PROBES = [("openr1_heldout", "Held-out OpenR1 math", BLUE), ("math500", "MATH-500", ORANGE),
          ("general", "General text (MMLU)", AQUA)]
STEPS = [1, 2, 5, 10, 21, 46, 98, 211, 454, 976]
BENCHES = [("math500", "MATH-500"), ("gsm8k", "GSM8K"), ("aime24", "AIME 2024"), ("aime26", "AIME 2026"),
           ("hmmt_feb26", "HMMT Feb 2026"), ("mmlu", "MMLU"), ("mmlu_pro", "MMLU-Pro")]
MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "router", "experts.gate_proj", "experts.up_proj", "experts.down_proj"]
MOD_NAMES = ["q", "k", "v", "o", "router", "expert\ngate", "expert\nup", "expert\ndown"]
GROUPS = [("Attention (q, k, v, o)", ["q_proj", "k_proj", "v_proj", "o_proj"]),
          ("Experts (gate, up, down)", ["experts.gate_proj", "experts.up_proj", "experts.down_proj"]),
          ("Router", ["router"])]

# One entry per analysed comparison (the result folder names). Wording used in the figure text:
#   model   the trained model          ref   what it is compared with      stage  the training whose steps are on x
#   final   benchmark folder of the model at step 976, bench_ref the benchmark folder of the reference
RUNS = {
    "sft": dict(model="SFT", ref="base", stage="SFT", final="sft_final", bench_ref="base",
                drift="from the base model during SFT", after="after SFT"),
    "grpo": dict(model="GRPO", ref="base", stage="GRPO", final="grpo_final", bench_ref="base",
                 drift="from the base model during GRPO", after="after GRPO"),
    "grpo_sft": dict(model="GRPO → SFT", ref="GRPO final", stage="SFT", final="grpo_sft_final", bench_ref="grpo_final",
                     drift="from the GRPO model during SFT", after="by SFT after GRPO (vs GRPO final)"),
    "grpo_sft_vs_Qwen3-30B-A3B-Base": dict(model="GRPO → SFT", ref="base", stage="SFT (after GRPO)",
                                           final="grpo_sft_final", bench_ref="base",
                                           drift="from the base model, GRPO then SFT",
                                           after="by GRPO then SFT (vs base)"),
}
BENCH_MODELS = {"base": "Base", "sft_final": "SFT", "grpo_final": "GRPO", "grpo_sft_final": "GRPO → SFT"}
# compare figures: colour = final model, dashed = measured from the base model across both stages
COMPARE = [("sft", "SFT (vs base)", BLUE, "-"), ("grpo", "GRPO (vs base)", ORANGE, "-"),
           ("grpo_sft", "SFT after GRPO (vs GRPO final)", AQUA, "-"),
           ("grpo_sft_vs_Qwen3-30B-A3B-Base", "GRPO → SFT total (vs base)", AQUA, "--")]
BENCH_COLORS = {"base": MUTED, "sft_final": BLUE, "grpo_final": ORANGE, "grpo_sft_final": AQUA}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def step_of(label):
    return int(label.rsplit("_", 1)[1])


def with_step(df):
    df = df.copy()
    df["step"] = df["ckpt"].map(step_of)
    return df


def final_ckpt(df):
    return df.loc[df["ckpt"].map(step_of).idxmax(), "ckpt"]


def step_axis(ax, steps, xlabel, right_room=False):
    steps = sorted(set(int(s) for s in steps))
    ax.set_xscale("log")
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.minorticks_off()
    ax.set_xlim(steps[0] * 0.8, steps[-1] * (3.6 if right_room else 1.3))
    ax.set_xlabel(xlabel)


def line(ax, x, y, color, ls="-"):
    ax.plot(x, y, ls, lw=2, color=color, zorder=3)
    ax.plot(x, y, "o", ms=6, color=color, mec="white", mew=1.5, zorder=4)


def end_labels(ax, items, min_gap=11):
    """Direct labels at the last point of each line, pushed apart vertically (min_gap points) so they never overlap.
    items: (x, y, text, color). Call after the axis scales and limits are final."""
    ax.get_xlim(), ax.get_ylim()  # force autoscaling
    pts = [ax.transData.transform((x, y))[1] * 72 / ax.figure.dpi for x, y, _, _ in items]
    placed, prev = {}, -1e9
    for i in sorted(range(len(items)), key=lambda i: pts[i]):
        placed[i] = prev = max(pts[i], prev + min_gap)
    for i, (x, y, text, color) in enumerate(items):
        ax.annotate(text, (x, y), xytext=(8, placed[i] - pts[i]), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)
        ax.plot([x], [y], "o", ms=8, color=color, mec="white", mew=2, zorder=5)


def noise_band(ax, x_text):
    ax.axhspan(0, 2.6, color="#f0efec", zorder=0)
    ax.text(x_text, 0.9, "near-tie noise floor (~2.5%)", fontsize=8, color=MUTED, ha="right")


def rms_by_group(pdm, mods):
    p = pdm[(pdm["layer"] >= 0) & pdm["module"].isin(mods)].copy()
    p["d2"] = p["rms"] ** 2 * p["numel"]
    s = p.groupby("step")[["d2", "numel"]].sum()
    return np.sqrt(s["d2"] / s["numel"])


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- one run vs its reference (fig1..fig8)

def fig_kl(res, out, name, R):
    kl = with_step(pd.read_csv(res / f"probe/{name}/kl.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = []
    for key, pname, c in PROBES:
        d = kl[kl["probe"] == key].sort_values("step")
        line(ax, d["step"], d["kl_ref_ckpt_token_mean"], c)
        labels.append((d["step"].iloc[-1], d["kl_ref_ckpt_token_mean"].iloc[-1], pname, c))
    step_axis(ax, kl["step"], f"{R['stage']} optimizer step (log scale)", right_room=True)
    ax.set_ylabel(f"KL({R['ref']} ‖ {R['model']}), nats per token")
    ax.set_title(f"Output drift {R['drift']}")
    end_labels(ax, labels)
    save(fig, out / "fig1_kl_drift.png")


def fig_routing_steps(res, out, name, R):
    rt = with_step(pd.read_csv(res / f"probe/{name}/routing.csv"))
    g = rt.groupby(["probe", "step"])[["overlap_total"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = []
    for key, pname, c in PROBES:
        d = g[g["probe"] == key].sort_values("step")
        y = 100 * (1 - d["overlap_total"])
        line(ax, d["step"], y, c)
        labels.append((d["step"].iloc[-1], y.iloc[-1], pname, c))
    step_axis(ax, g["step"], f"{R['stage']} optimizer step (log scale)", right_room=True)
    noise_band(ax, ax.get_xlim()[1] * 0.95)
    ax.set_ylim(0, None)
    ax.set_ylabel("Top-8 expert choices changed (%)\nmean over 48 layers")
    ax.set_title(f"Routing drift {R['drift']}")
    end_labels(ax, labels)
    save(fig, out / "fig2_routing_drift.png")


def fig_routing_layers(res, out, name, R):
    rt = pd.read_csv(res / f"probe/{name}/routing.csv")
    last = final_ckpt(rt)
    d = rt[(rt["ckpt"] == last) & (rt["probe"] == "openr1_heldout")].sort_values("layer")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    series = [("overlap_total", "Total change", BLUE), ("overlap_inputs", "Inputs only (old router, new hidden states)", ORANGE),
              ("overlap_router", "Router weights only (same hidden states)", AQUA)]
    for col, sname, c in series:
        ax.plot(d["layer"], 100 * (1 - d[col]), "-", lw=2, color=c, label=sname)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Top-8 expert choices changed (%)\n{R['model']} vs {R['ref']}")
    ax.set_ylim(0, None)
    ax.set_xlim(0, 47)
    ax.set_title(f"Where routing changed, and why (step {step_of(last)}, held-out math)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out / "fig3_routing_by_layer.png")


def fig_param_heatmap(res, out, name, R):
    pdm = pd.read_csv(res / f"param_diff/{name}/param_diff_modules.csv")
    last = final_ckpt(pdm)
    d = pdm[(pdm["ckpt"] == last) & (pdm["layer"] >= 0)]
    m = d.pivot(index="layer", columns="module", values="rel")[MODS] * 100
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    im = ax.imshow(m.values, aspect="auto", cmap=SEQ, origin="lower", vmin=0)
    ax.set_xticks(range(len(MODS)), MOD_NAMES, fontsize=8.5)
    ax.set_yticks(range(0, 48, 4))
    ax.set_ylabel("Layer")
    ax.grid(False)
    ax.set_title(f"Relative weight change {R['after']}\n(step {step_of(last)})")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(f"‖ΔW‖ / ‖W_{R['ref'].replace(' ', '_')}‖ (%)", color=INK2)
    cb.outline.set_visible(False)
    save(fig, out / "fig4_param_change_heatmap.png")


def fig_param_steps(res, out, name, R):
    pdm = with_step(pd.read_csv(res / f"param_diff/{name}/param_diff_modules.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    shown = set()
    for (gname, mods), c in zip(GROUPS, (BLUE, ORANGE, AQUA)):
        y = rms_by_group(pdm, mods)
        y = y[y > 0]  # a step with no change (bit-identical to the reference) has no place on a log axis
        shown |= set(y.index)
        line(ax, y.index, y, c)
        ax.plot([], [], "-o", lw=2, ms=6, color=c, mec="white", label=gname)
    ax.legend(loc="lower right", fontsize=9)
    step_axis(ax, shown, f"{R['stage']} optimizer step (log scale)")
    ax.set_yscale("log")
    ax.set_ylabel(f"RMS weight change per entry vs {R['ref']}\n(log scale)")
    ax.set_title(f"How fast the weights move {R['drift']}")
    save(fig, out / "fig5_param_change_over_steps.png")


def fig_specialization(experts_gz, res, out, name, R):
    ex = pd.read_csv(experts_gz)
    last = final_ckpt(ex)
    d = ex[ex["ckpt"] == last].copy()
    d["rel_norm"] = d["rel"] / d.groupby("layer")["rel"].transform("median")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(d["affinity"], d["rel_norm"], s=6, color=BLUE, alpha=0.25, linewidths=0)
    bins = np.quantile(d["affinity"], np.linspace(0, 1, 21))  # binned mean, to show the trend
    d["bin"] = pd.cut(d["affinity"], bins, include_lowest=True)
    b = d.groupby("bin", observed=True).agg(x=("affinity", "mean"), y=("rel_norm", "mean"))
    print(f"[{name}] specialization bins (mean affinity, mean update/layer median):")
    print(b.round(3).to_string(index=False))
    ax.plot(b["x"], b["y"], "-", lw=2, color=ORANGE)
    ax.plot(b["x"], b["y"], "o", ms=6, color=ORANGE, mec="white", mew=1.5)
    ax.annotate("mean of each 5% affinity bin", (b["x"].iloc[1], b["y"].iloc[1]), xytext=(12, -4),
                textcoords="offset points", ha="left", va="top", fontsize=9, color=INK2)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel(f"Math affinity of the expert in the {R['ref']} model\n"
                  "(log share on math − log share on general text)")
    ax.set_ylabel("Expert update size / layer median")
    ax.set_title(f"Do math-specialized experts get larger {R['stage']} updates?")
    ax.set_ylim(min(d["rel_norm"].quantile(0.002), b["y"].min()) * 0.95, d["rel_norm"].quantile(0.998) * 1.12)
    ax.set_xlim(d["affinity"].quantile(0.002) - 0.5, d["affinity"].quantile(0.998) + 0.5)
    sp = pd.read_csv(res / f"specialization/specialization_{name}.csv")
    sp = sp[sp["ckpt"] == last]
    ax.text(0.02, 0.97, f"6,144 experts (48 layers × 128), step {step_of(last)}\n"
            f"rank correlation, mean over layers: {sp['spearman'].mean():+.2f}\n"
            f"top-10% math experts' share of update energy: {100 * sp['top10_share'].mean():.1f}% (10% = no preference)",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    save(fig, out / "fig6_expert_specialization.png")


def fig_specialization_steps(res, out, name, R):
    sp = with_step(pd.read_csv(res / f"specialization/specialization_{name}.csv"))
    g = sp.dropna(subset=["spearman", "top10_share"]).groupby("step")[["spearman", "top10_share", "gini"]].mean()
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    line(axs[0], g.index, g["spearman"], BLUE)
    axs[0].axhline(0, color=MUTED, lw=1, ls="--")
    axs[0].set_title("Affinity vs update size\n(rank correlation, mean over layers)", fontsize=10)
    line(axs[1], g.index, 100 * g["top10_share"], BLUE)
    axs[1].axhline(10, color=MUTED, lw=1, ls="--")
    axs[1].set_title("Share of update energy in the 10% most\nmath-affine experts (%; dashed = no preference)",
                     fontsize=10)
    for ax in axs:
        step_axis(ax, [s for s in g.index if s in (1, 2, 10, 46, 211, 976)] + [g.index.min(), g.index.max()],
                  f"{R['stage']} step (log scale)")
    fig.tight_layout()
    panel_title(fig, f"Expert specialization of the update, {R['model']} vs {R['ref']}", 1.0)
    save(fig, out / "fig7_specialization_over_steps.png")


def bench_table(res, models):
    """{model: {bench: summary}} for the benchmark folders that exist."""
    out = {}
    for m in models:
        out[m] = {b: json.load(open(res / m / f"{b}.summary.json")) for b, _ in BENCHES
                  if (res / m / f"{b}.summary.json").exists()}
    return out


def bars(ax, models, names, S, width):
    x = np.arange(len(names))
    n = len(models)
    for i, m in enumerate(models):
        xs = x + (i - (n - 1) / 2) * (width + 0.02)
        for xi, (b, _) in zip(xs, names):
            s = S[m].get(b)
            if s is None:
                ax.text(xi, 1.5, "n/a", ha="center", va="bottom", fontsize=7, color=MUTED, rotation=90)
                continue
            a, e = 100 * s["accuracy"], 100 * s["ci95"]
            ax.bar(xi, a, width, color=BENCH_COLORS[m], zorder=3)
            ax.errorbar(xi, a, yerr=e, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
            if n <= 2:
                ax.text(xi, a + e + 1.5, f"{a:.0f}", ha="center", va="bottom", fontsize=8, color=INK2, zorder=5)
    ax.legend([Patch(color=BENCH_COLORS[m]) for m in models], [BENCH_MODELS[m] for m in models], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=len(models))
    ax.set_xticks(x, [nm for _, nm in names])
    ax.set_ylabel("Accuracy (%), mean over samples\nerror bars: 95% CI over questions")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="x", visible=False)


def fig_bench(res, out, name, R):
    """Reference model vs this run's final model, on the benchmarks both have."""
    models = [R["bench_ref"], R["final"]]
    S = bench_table(res, models)
    names = [(b, n) for b, n in BENCHES if all(b in S[m] for m in models)]
    if not names:
        print(f"[{name}] no benchmark summaries for {models}; fig8 skipped")
        return
    fig, ax = plt.subplots(figsize=(max(7.2, 1.3 * len(names) + 1.6), 3.8))
    bars(ax, models, names, S, 0.38)
    ax.set_title(f"Benchmark accuracy: {BENCH_MODELS[models[0]]} vs {BENCH_MODELS[models[1]]} (step 976)", pad=24)
    save(fig, out / "fig8_benchmarks.png")


# ---------------------------------------------------------------- all runs on shared axes (c1..c5)

def compare_series(res, kind):
    """(name, label, colour, linestyle, dataframe-with-step) for every comparison whose results exist."""
    out = []
    for name, label, c, ls in COMPARE:
        f = res / (f"probe/{name}/{kind}.csv" if kind in ("kl", "routing") else
                   f"param_diff/{name}/param_diff_modules.csv" if kind == "param" else
                   f"specialization/specialization_{name}.csv")
        if f.exists():
            out.append((name, label, c, ls, with_step(pd.read_csv(f))))
    return out


def legend_top(fig, series, y=1.0):
    handles = [Line2D([], [], color=c, ls=ls, lw=2.5) for _, _, c, ls, _ in series]
    fig.legend(handles, [lbl for _, lbl, _, _, _ in series], loc="lower center", ncol=len(series),
               bbox_to_anchor=(0.5, y), fontsize=9, handlelength=3.2)


def panel_title(fig, text, y):
    fig.suptitle(text, fontsize=11, fontweight="bold", color=INK, y=y, va="bottom")


def compare_kl(res, out):
    series = compare_series(res, "kl")
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (key, pname, _) in zip(axs, PROBES):
        for _, _, c, ls, kl in series:
            d = kl[(kl["probe"] == key) & (kl["kl_ref_ckpt_token_mean"] > 0)].sort_values("step")
            line(ax, d["step"], d["kl_ref_ckpt_token_mean"], c, ls)
        ax.set_yscale("log")
        step_axis(ax, [1, 5, 21, 98, 454, 976], "Training step (log scale)")
        ax.set_title(pname, fontsize=10)
    axs[0].set_ylabel("KL(reference ‖ model)\nnats per token (log scale)")
    fig.tight_layout()
    legend_top(fig, series, 1.0)
    panel_title(fig, "Output drift from each run's reference model", 1.08)
    save(fig, out / "c1_kl_drift.png")


def compare_routing(res, out):
    series = compare_series(res, "routing")
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    top = 0
    for ax, (key, pname, _) in zip(axs, PROBES):
        for _, _, c, ls, rt in series:
            g = rt[rt["probe"] == key].groupby("step")["overlap_total"].mean()
            line(ax, g.index, 100 * (1 - g), c, ls)
            top = max(top, 100 * (1 - g).max())
        step_axis(ax, [1, 5, 21, 98, 454, 976], "Training step (log scale)")
        noise_band(ax, ax.get_xlim()[1] * 0.9)
        ax.set_title(pname, fontsize=10)
    axs[0].set_ylim(0, top * 1.06)
    axs[0].set_ylabel("Top-8 expert choices changed (%)\nmean over 48 layers")
    fig.tight_layout()
    legend_top(fig, series, 1.0)
    panel_title(fig, "Routing drift from each run's reference model", 1.08)
    save(fig, out / "c2_routing_drift.png")


def compare_param(res, out):
    series = compare_series(res, "param")
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (gname, mods) in zip(axs, GROUPS):
        for _, _, c, ls, pdm in series:
            y = rms_by_group(pdm, mods)
            y = y[y > 0]
            line(ax, y.index, y, c, ls)
        ax.set_yscale("log")
        step_axis(ax, [1, 5, 21, 98, 454, 976], "Training step (log scale)")
        ax.set_title(gname, fontsize=10)
    axs[0].set_ylabel("RMS weight change per entry\nvs reference (log scale)")
    fig.tight_layout()
    legend_top(fig, series, 1.0)
    panel_title(fig, "How far the weights move from each run's reference model", 1.08)
    save(fig, out / "c3_param_change_over_steps.png")


def compare_specialization(res, out):
    series = compare_series(res, "spec")
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for _, _, c, ls, sp in series:
        g = sp.dropna(subset=["spearman", "top10_share"]).groupby("step")[["spearman", "top10_share"]].mean()
        line(axs[0], g.index, g["spearman"], c, ls)
        line(axs[1], g.index, 100 * g["top10_share"], c, ls)
    axs[0].axhline(0, color=MUTED, lw=1, ls="--")
    axs[0].set_title("Affinity vs update size\n(rank correlation, mean over layers)", fontsize=10)
    axs[1].axhline(10, color=MUTED, lw=1, ls="--")
    axs[1].set_title("Share of update energy in the 10% most\nmath-affine experts (%; dashed = no preference)",
                     fontsize=10)
    for ax in axs:
        step_axis(ax, [1, 5, 21, 98, 454, 976], "Training step (log scale)")
    fig.tight_layout()
    legend_top(fig, series, 1.0)
    panel_title(fig, "Are the updates concentrated on math-specialized experts?", 1.08)
    save(fig, out / "c4_specialization_over_steps.png")


def compare_bench(res, out):
    models = [m for m in BENCH_MODELS if (res / m).is_dir()]
    S = bench_table(res, models)
    names = [(b, n) for b, n in BENCHES if any(b in S[m] for m in models)]
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    bars(ax, models, names, S, 0.19)
    ax.set_title("Benchmark accuracy of the four models (step 976; n/a = not evaluated)", pad=24)
    save(fig, out / "c5_benchmarks.png")
    print("accuracy (%):")
    print(pd.DataFrame({BENCH_MODELS[m]: {nm: round(100 * S[m][b]["accuracy"], 1) if b in S[m] else None
                                          for b, nm in names} for m in models}).to_string())


def summary(res):
    """Final-step numbers of every comparison (printed to the log, for the write-up)."""
    rows = []
    for name, label, _, _, kl in compare_series(res, "kl"):
        rt = pd.read_csv(res / f"probe/{name}/routing.csv")
        pdm = with_step(pd.read_csv(res / f"param_diff/{name}/param_diff_modules.csv"))
        last = kl["step"].max()
        r = {"comparison": label}
        for key, _, _ in PROBES:
            r[f"KL {key}"] = kl[(kl["step"] == last) & (kl["probe"] == key)]["kl_ref_ckpt_token_mean"].iloc[0]
            r[f"routing% {key}"] = 100 * (1 - rt[(rt["ckpt"].map(step_of) == last) & (rt["probe"] == key)]
                                          ["overlap_total"].mean())
        for gname, mods in GROUPS:
            r[f"rms {gname.split()[0]}"] = rms_by_group(pdm, mods).loc[last]
        rows.append(r)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
        print(pd.DataFrame(rows).set_index("comparison").to_string())


def main():
    res, figs, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    experts_gz = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    out = figs / name
    out.mkdir(parents=True, exist_ok=True)
    if name == "compare":
        compare_kl(res, out)
        compare_routing(res, out)
        compare_param(res, out)
        compare_specialization(res, out)
        compare_bench(res, out)
        summary(res)
        return
    if name not in RUNS:
        sys.exit(f"unknown run {name!r}: one of {', '.join(RUNS)} or compare")
    R = RUNS[name]
    fig_kl(res, out, name, R)
    fig_routing_steps(res, out, name, R)
    fig_routing_layers(res, out, name, R)
    fig_param_heatmap(res, out, name, R)
    fig_param_steps(res, out, name, R)
    if experts_gz is not None and experts_gz.exists():
        fig_specialization(experts_gz, res, out, name, R)
    else:
        print(f"[{name}] no per-expert table {experts_gz}; fig6 skipped (existing file kept)")
    fig_specialization_steps(res, out, name, R)
    fig_bench(res, out, name, R)


if __name__ == "__main__":
    main()
