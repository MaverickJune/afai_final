"""Figures for the SFT-stage analysis (afai_env, CPU). Reads the CSVs in experiments/eval/results/ and the per-expert
table from $SCR/eval/specialization_sft_experts.csv.gz; writes PNGs to experiments/eval/figures/.

usage: python plot_sft_results.py RESULTS_DIR EXPERTS_CSV_GZ OUT_DIR
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Reference palette (dataviz skill, light mode): first three categorical slots, ink, grid.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
PROBES = [("openr1_heldout", "Held-out OpenR1 math", BLUE), ("math500", "MATH-500", ORANGE),
          ("general", "General text (MMLU)", AQUA)]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def step_of(label):
    return int(label.rsplit("_", 1)[1])


def direct_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.plot([x], [y], "o", ms=8, color=color, mec="white", mew=2, zorder=5)


def line(ax, x, y, color):
    ax.plot(x, y, "-", lw=2, color=color, zorder=3)
    ax.plot(x, y, "o", ms=6, color=color, mec="white", mew=1.5, zorder=4)


def fig_kl(res, out):
    kl = pd.read_csv(res / "probe/sft/kl.csv")
    kl["step"] = kl["ckpt"].map(step_of)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, name, c in PROBES:
        d = kl[kl["probe"] == key].sort_values("step")
        line(ax, d["step"], d["kl_ref_ckpt_token_mean"], c)
        direct_label(ax, d["step"].iloc[-1], d["kl_ref_ckpt_token_mean"].iloc[-1], name, c)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 21, 46, 98, 211, 454, 976], ["1", "2", "5", "10", "21", "46", "98", "211", "454", "976"])
    ax.set_xlim(0.8, 3500)
    ax.set_xlabel("SFT optimizer step (log scale)")
    ax.set_ylabel("KL(base ‖ SFT), nats per token")
    ax.set_title("Output drift from the base model during SFT")
    fig.savefig(out / "fig1_kl_drift.png")
    plt.close(fig)


def fig_routing_steps(res, out):
    rt = pd.read_csv(res / "probe/sft/routing.csv")
    rt["step"] = rt["ckpt"].map(step_of)
    g = rt.groupby(["probe", "step"])[["overlap_total"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    nudge = {"openr1_heldout": -7, "general": 5, "math500": 0}
    for key, name, c in PROBES:
        d = g[g["probe"] == key].sort_values("step")
        y = 100 * (1 - d["overlap_total"])
        line(ax, d["step"], y, c)
        direct_label(ax, d["step"].iloc[-1], y.iloc[-1], name, c, dy=nudge[key])
    ax.axhspan(0, 2.6, color="#f0efec", zorder=0)
    ax.text(3300, 0.9, "near-tie noise floor (~2.5%, reached at step 2)", fontsize=8, color=MUTED, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 21, 46, 98, 211, 454, 976], ["1", "2", "5", "10", "21", "46", "98", "211", "454", "976"])
    ax.set_xlim(0.8, 3500)
    ax.set_ylim(0, None)
    ax.set_xlabel("SFT optimizer step (log scale)")
    ax.set_ylabel("Top-8 expert choices changed (%)\nmean over 48 layers")
    ax.set_title("Routing drift from the base model during SFT")
    fig.savefig(out / "fig2_routing_drift.png")
    plt.close(fig)


def fig_routing_layers(res, out):
    rt = pd.read_csv(res / "probe/sft/routing.csv")
    d = rt[(rt["ckpt"] == "sft_step_00976") & (rt["probe"] == "openr1_heldout")].sort_values("layer")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    series = [("overlap_total", "Total change", BLUE), ("overlap_inputs", "Inputs only (old router, new hidden states)", ORANGE),
              ("overlap_router", "Router weights only (same hidden states)", AQUA)]
    for col, name, c in series:
        y = 100 * (1 - d[col])
        ax.plot(d["layer"], y, "-", lw=2, color=c, label=name)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-8 expert choices changed (%)")
    ax.set_ylim(0, None)
    ax.set_xlim(0, 47)
    ax.set_title("Where routing changed, and why (step 976, held-out math)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(out / "fig3_routing_by_layer.png")
    plt.close(fig)


MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "router", "experts.gate_proj", "experts.up_proj", "experts.down_proj"]
MOD_NAMES = ["q", "k", "v", "o", "router", "expert\ngate", "expert\nup", "expert\ndown"]


def fig_param_heatmap(res, out):
    pdm = pd.read_csv(res / "param_diff/sft/param_diff_modules.csv")
    d = pdm[(pdm["ckpt"] == "sft_step_00976") & (pdm["layer"] >= 0)]
    m = d.pivot(index="layer", columns="module", values="rel")[MODS] * 100
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    im = ax.imshow(m.values, aspect="auto", cmap=SEQ, origin="lower", vmin=0)
    ax.set_xticks(range(len(MODS)), MOD_NAMES, fontsize=8.5)
    ax.set_yticks(range(0, 48, 4))
    ax.set_ylabel("Layer")
    ax.grid(False)
    ax.set_title("Relative weight change after SFT (step 976)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("‖ΔW‖ / ‖W_base‖ (%)", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(out / "fig4_param_change_heatmap.png")
    plt.close(fig)


def fig_param_steps(res, out):
    pdm = pd.read_csv(res / "param_diff/sft/param_diff_modules.csv")
    pdm = pdm[pdm["layer"] >= 0].copy()
    pdm["step"] = pdm["ckpt"].map(step_of)
    pdm["d2"] = pdm["rms"] ** 2 * pdm["numel"]
    groups = [("Attention (q, k, v, o)", ["q_proj", "k_proj", "v_proj", "o_proj"], BLUE),
              ("Experts (gate, up, down)", ["experts.gate_proj", "experts.up_proj", "experts.down_proj"], ORANGE),
              ("Router", ["router"], AQUA)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, mods, c in groups:
        s = pdm[pdm["module"].isin(mods)].groupby("step")[["d2", "numel"]].sum()
        s = s[s.index > 1]  # step 1 is bit-identical to the base (no change to plot on a log axis)
        y = np.sqrt(s["d2"] / s["numel"])
        line(ax, s.index, y, c)
        ax.plot([], [], "-o", lw=2, ms=6, color=c, mec="white", label=name)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 21, 46, 98, 211, 454, 976], ["2", "5", "10", "21", "46", "98", "211", "454", "976"])
    ax.set_xlim(1.6, 1300)
    ax.set_xlabel("SFT optimizer step (log scale)")
    ax.set_ylabel("RMS weight change per entry (log scale)")
    ax.set_title("How fast the weights move during SFT")
    fig.savefig(out / "fig5_param_change_over_steps.png")
    plt.close(fig)


def fig_specialization(experts_gz, res, out):
    ex = pd.read_csv(experts_gz)
    d = ex[ex["ckpt"] == "sft_step_00976"].copy()
    d["rel_norm"] = d["rel"] / d.groupby("layer")["rel"].transform("median")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(d["affinity"], d["rel_norm"], s=6, color=BLUE, alpha=0.25, linewidths=0)
    # binned mean, to show the trend
    bins = np.quantile(d["affinity"], np.linspace(0, 1, 21))
    d["bin"] = pd.cut(d["affinity"], bins, include_lowest=True)
    b = d.groupby("bin", observed=True).agg(x=("affinity", "mean"), y=("rel_norm", "mean"))
    print("specialization bins (mean affinity, mean update/layer median):")
    print(b.round(3).to_string(index=False))
    for lo, hi in ((-99, -2), (-2, 0), (0, 2), (2, 99)):
        m = d[(d["affinity"] >= lo) & (d["affinity"] < hi)]
        print(f"affinity [{lo},{hi}): {len(m)} experts, mean update/median {m['rel_norm'].mean():.3f}")
    ax.plot(b["x"], b["y"], "-", lw=2, color=ORANGE)
    ax.plot(b["x"], b["y"], "o", ms=6, color=ORANGE, mec="white", mew=1.5)
    ax.annotate("mean of each 5% affinity bin", (b["x"].iloc[1], b["y"].iloc[1]), xytext=(12, -4),
                textcoords="offset points", ha="left", va="top", fontsize=9, color=INK2)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Math affinity of the expert in the base model\n(log share on math − log share on general text)")
    ax.set_ylabel("Expert update size / layer median")
    ax.set_title("Do math-specialized experts get larger SFT updates?")
    ax.set_ylim(min(d["rel_norm"].quantile(0.002), b["y"].min()) * 0.95, d["rel_norm"].quantile(0.998) * 1.12)
    ax.set_xlim(d["affinity"].quantile(0.002) - 0.5, d["affinity"].quantile(0.998) + 0.5)
    sp = pd.read_csv(res / "specialization/specialization_sft.csv")
    sp = sp[sp["ckpt"] == "sft_step_00976"]
    ax.text(0.02, 0.97, f"6,144 experts (48 layers × 128)\nrank correlation, mean over layers: {sp['spearman'].mean():+.2f}\n"
            f"top-10% math experts' share of update energy: {100 * sp['top10_share'].mean():.1f}% (10% = no preference)",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    fig.savefig(out / "fig6_expert_specialization.png")
    plt.close(fig)


def fig_specialization_steps(res, out):
    sp = pd.read_csv(res / "specialization/specialization_sft.csv")
    sp["step"] = sp["ckpt"].map(step_of)
    g = sp[sp["step"] > 1].groupby("step")[["spearman", "top10_share", "gini"]].mean()
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    line(axs[0], g.index, g["spearman"], BLUE)
    axs[0].axhline(0, color=MUTED, lw=1, ls="--")
    axs[0].set_title("Affinity vs update size\n(rank correlation, mean over layers)", fontsize=10)
    line(axs[1], g.index, 100 * g["top10_share"], BLUE)
    axs[1].axhline(10, color=MUTED, lw=1, ls="--")
    axs[1].text(2.1, 10.15, "10% = no preference", fontsize=8, color=MUTED)
    axs[1].set_title("Share of update energy in the\n10% most math-affine experts (%)", fontsize=10)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xticks([2, 10, 46, 211, 976], ["2", "10", "46", "211", "976"])
        ax.set_xlabel("SFT step (log scale)")
    fig.tight_layout()
    fig.savefig(out / "fig7_specialization_over_steps.png")
    plt.close(fig)


def fig_bench(res, out):
    """Base vs SFT accuracy (avg@n, 95% CI over questions) on the benchmarks both models have finished."""
    import json
    names = [("math500", "MATH-500"), ("gsm8k", "GSM8K"), ("aime24", "AIME 2024"), ("aime26", "AIME 2026"),
             ("hmmt_feb26", "HMMT Feb 2026"), ("mmlu", "MMLU"), ("mmlu_pro", "MMLU-Pro")]
    load = lambda m, b: json.load(open(res / m / f"{b}.summary.json"))  # noqa: E731
    names = [(b, n) for b, n in names if (res / "base" / f"{b}.summary.json").exists()
             and (res / "sft_final" / f"{b}.summary.json").exists()]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (m, lbl, c) in enumerate((("base", "Base", BLUE), ("sft_final", "SFT (step 976)", ORANGE))):
        acc = [100 * load(m, b)["accuracy"] for b, _ in names]
        ci = [100 * load(m, b)["ci95"] for b, _ in names]
        xs = x + (i - 0.5) * (w + 0.02)
        ax.bar(xs, acc, w, color=c, label=lbl, zorder=3)
        ax.errorbar(xs, acc, yerr=ci, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
        for xi, a, e in zip(xs, acc, ci):
            ax.text(xi, a + e + 1.5, f"{a:.0f}", ha="center", va="bottom", fontsize=8, color=INK2, zorder=5)
    ax.set_xticks(x, [n for _, n in names])
    ax.set_ylabel("Accuracy (%), mean over samples\nerror bars: 95% CI over questions")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Benchmark accuracy before and after SFT")
    fig.savefig(out / "fig8_benchmarks_base_vs_sft.png")
    plt.close(fig)


def main():
    res, experts_gz, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 4 and sys.argv[4] == "bench":  # only the benchmark figure
        fig_bench(res, out)
        print("wrote fig8_benchmarks_base_vs_sft.png")
        return
    fig_kl(res, out)
    fig_routing_steps(res, out)
    fig_routing_layers(res, out)
    fig_param_heatmap(res, out)
    fig_param_steps(res, out)
    fig_specialization(experts_gz, res, out)
    fig_specialization_steps(res, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
