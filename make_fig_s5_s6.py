# -*- coding: utf-8 -*-
"""
make_fig_s5_s6.py — V13 reviewer Additional-C figures: paired per-dataset
log-loss difference distributions for the ablation, with the null line,
the mean, and the bootstrap 95% CI. Reads review3_diagnostics5.json.

Fig S5: M1, three generating settings x three variants (T=80, R=30).
Fig S6: M2, three trial lengths x three variants (moderate setting, R=15).
Short suptitle (full sentence is in the caption) to avoid canvas overflow.
"""
import json
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
CODE_DIR = paths.CODE_DIR
J5 = paths.data_path("review3_diagnostics5.json")

with io.open(J5, encoding="utf-8") as f:
    d = json.load(f)
diffs = d["paired_diffs_for_figures"]
boot = d["ablation_bootstrap"]

COMP = [
    ("full_vs_mean_wo_sprime", "full vs no-s'"),
    ("full_vs_precision_simple", "full vs simpler-precision"),
    ("full_vs_rw", "full vs RW"),
]
SETTINGS = [
    ("setting_0", "moderate (0.5,2,0.05,3)"),
    ("setting_1", "high-gain (0,3,0.1,5)"),
    ("setting_2", "low-gain (-1,1.5,0.02,2)"),
]
TS = ["40", "80", "120"]


def draw_panel(ax, vals, mean, ci, title):
    vals = np.asarray(vals, dtype=float)
    lo, hi = ci
    bins = np.linspace(vals.min(), vals.max(), 12) if vals.max() > vals.min() else 10
    ax.hist(vals, bins=bins, color="#9ec5e8", edgecolor="white")
    ax.axvline(0.0, color="#c0392b", ls="--", lw=1.4)
    ax.axvline(mean, color="#e67e22", ls="-", lw=1.6)
    ax.axvspan(lo, hi, color="#e67e22", alpha=0.15)
    sig = (hi < 0.0)
    tag = "full better" if sig else ("variant better" if lo > 0.0 else "CI spans 0")
    ax.set_title(f"{title}\nmean {mean:+.3f} [{lo:+.3f},{hi:+.3f}]\n{tag}",
                 fontsize=8)
    ax.tick_params(labelsize=7)


def make_fig(kind):
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=False)
    if kind == "s5":
        rows = [s[0] for s in SETTINGS]
        row_labels = [s[1] for s in SETTINGS]
        fig.suptitle("Paired log-loss differences, M1: three generating settings (T=80)",
                     fontsize=12)
    else:
        rows = [f"T{t}" for t in TS]
        row_labels = [f"T={t}" for t in TS]
        fig.suptitle("Paired log-loss differences, M2: three trial lengths (moderate)",
                     fontsize=12)
    for i, rk in enumerate(rows):
        for j, (ck, clabel) in enumerate(COMP):
            ax = axes[i, j]
            vals = diffs[f"{rk}_{ck}"]
            if kind == "s5":
                stat = boot["m1_t80"][rk]["comparisons"][ck]
            else:
                stat = boot["m2_multi_T"][rk.lstrip("T")]["comparisons"][ck]
            draw_panel(ax, vals, stat["mean_diff"], stat["bootstrap_ci95"], clabel)
            if i == 2:
                ax.set_xlabel("paired $\\Delta$ log-loss (full $-$ variant)", fontsize=8)
            if j == 0:
                ax.set_ylabel(row_labels[i] + "\ncount", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = paths.figure_path(f"fig_{kind}_paired_diffs.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("SAVED", out)


make_fig("s5")
make_fig("s6")
