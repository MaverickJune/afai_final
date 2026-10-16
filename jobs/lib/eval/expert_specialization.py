"""Are parameter updates localized to math-specialized experts? Combines the outputs of probe_analysis.py and
param_diff.py for one run (CPU, afai_env).

Math affinity of expert e in layer l, from the REFERENCE model's routing (ref_tokens in expert_load.csv.gz):
    affinity = log( share of math-probe tokens routed to e ) - log( share of general-probe tokens routed to e )
    (add-one smoothed; > 0: used more on math than on general text)
Update size of expert e at checkpoint c, from param_diff_tensors.csv.gz: relative change of its gate/up/down
projections pooled, rel = sqrt(sum ||dW||^2) / sqrt(sum ||W||^2).

Per (checkpoint, layer):
    spearman        rank correlation of affinity and update size over the layer's experts
    top10_share     share of the layer's total expert update energy (sum ||dW||^2) in the 10% most math-affine
                    experts (0.10 = no preference)
    gini            Gini coefficient of update energy over experts (0 = spread evenly, 1 = all in one expert)
    max_over_median largest / median expert update size

usage: python expert_specialization.py --probe-out DIR --param-diff DIR --out FILE.csv
                                       [--math-probe openr1_heldout] [--general-probe general]
"""

import argparse

import numpy as np
import pandas as pd


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-out", required=True)
    p.add_argument("--param-diff", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--math-probe", default="openr1_heldout")
    p.add_argument("--general-probe", default="general")
    a = p.parse_args()

    ld = pd.read_csv(f"{a.probe_out}/expert_load.csv.gz")
    ref = ld[ld["model"] == ld["model"].iloc[0]]  # ref_tokens are the same in every checkpoint's rows

    def share(probe):
        t = ref[ref["probe"] == probe].set_index(["layer", "expert"])["ref_tokens"].astype(float)
        return (t + 1) / (t.groupby(level="layer").transform("sum") + t.groupby(level="layer").transform("count"))
    aff = (np.log(share(a.math_probe)) - np.log(share(a.general_probe))).rename("affinity")

    td = pd.read_csv(f"{a.param_diff}/param_diff_tensors.csv.gz")
    ex = td[td["expert"] >= 0].copy()
    ex["d2"], ex["b2"] = ex["diff_fro"] ** 2, ex["base_fro"] ** 2
    upd = ex.groupby(["ckpt", "layer", "expert"])[["d2", "b2"]].sum().reset_index()
    upd["rel"] = np.sqrt(upd["d2"] / upd["b2"])
    upd = upd.join(aff, on=["layer", "expert"])
    assert upd["affinity"].notna().all(), "experts without affinity (probe/param_diff layer mismatch?)"

    rows = []
    for (ck, layer), g in upd.groupby(["ckpt", "layer"]):
        n_top = max(1, int(round(0.1 * len(g))))
        top = g.nlargest(n_top, "affinity")
        rows.append({"ckpt": ck, "layer": layer, "n_experts": len(g),
                     "spearman": g["affinity"].rank().corr(g["rel"].rank()),
                     "top10_share": top["d2"].sum() / g["d2"].sum() if g["d2"].sum() > 0 else float("nan"),
                     "gini": gini(g["d2"]),
                     "max_over_median": g["rel"].max() / g["rel"].median() if g["rel"].median() > 0 else float("nan")})
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    upd.to_csv(a.out.replace(".csv", "_experts.csv.gz"), index=False)
    print(res.groupby("ckpt")[["spearman", "top10_share", "gini", "max_over_median"]].mean().to_string())
    print(f"wrote {a.out} and per-expert table {a.out.replace('.csv', '_experts.csv.gz')}")


if __name__ == "__main__":
    main()
