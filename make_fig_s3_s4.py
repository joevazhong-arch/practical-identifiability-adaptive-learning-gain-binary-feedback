# -*- coding: utf-8 -*-
"""make_fig_s3_s4.py —— OCT17_11 审稿 MC2 要求的 true-versus-estimated 散点图。
Fig S3: 24 条件网格（R100, T=80）四参数 true vs estimated（log/log-odds 空间），身份线。
Fig S4: 代表设置 θ*=(0.5,2,0.05,3) 五个试次长度（R100）四参数散点，颜色区分 T，身份线。
数据来自 review3_diagnostics4.json。
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

d = json.load(io.open(paths.data_path("review3_diagnostics4.json"),
                      encoding="utf-8"))

labels = {"mu": r"$\mu$ (log-odds)", "sigma": r"$\sigma$ ($\log$ scale)",
          "omega": r"$\omega$ ($\log$ scale)", "lambda": r"$\lambda$ ($\log$ scale)"}
keys = ["mu", "sigma", "omega", "lambda"]

# ---------------- Fig S3: grid scatter ----------------
g = d["scatter_grid_r100"]
cols = g["columns"]
recs = np.asarray(g["records"], dtype=float)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
for ax, k in zip(axes.ravel(), keys):
    it = cols.index(k + "_true")
    ih = cols.index(k + "_hat")
    x = recs[:, it]
    y = recs[:, ih]
    lo = float(min(x.min(), y.min())) - 0.2
    hi = float(max(x.max(), y.max())) + 0.2
    ax.scatter(x, y, s=7, alpha=0.28, color="#1f6fb2", edgecolors="none",
               rasterized=True)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="#b23a48",
            label="identity line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true " + labels[k], fontsize=9)
    ax.set_ylabel("estimated " + labels[k], fontsize=9)
    ax.set_title(labels[k], fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("True-versus-estimated scatterplots: 24-condition grid", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(paths.figure_path("fig_s3_grid_scatter.png"), dpi=200)
plt.close(fig)

# ---------------- Fig S4: multi-T scatter ----------------
m = d["scatter_multi_T_r100"]
Ts = sorted(m["by_T"].keys(), key=lambda z: int(z))
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(Ts)))
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
for ax, k in zip(axes.ravel(), keys):
    j = keys.index(k)
    z_true = m["by_T"][Ts[0]]["z_true"][j]
    lo_all, hi_all = z_true, z_true
    for ci, Tk in enumerate(Ts):
        recs_t = np.asarray(m["by_T"][Tk]["records"], dtype=float)
        y = recs_t[:, j]
        x = np.full_like(y, z_true)
        ax.scatter(x + np.random.default_rng(int(Tk)).normal(0, 0.03, size=len(x)),
                   y, s=8, alpha=0.35, color=colors[ci], edgecolors="none",
                   label="T=" + Tk, rasterized=True)
        lo_all = min(lo_all, float(y.min()))
        hi_all = max(hi_all, float(y.max()))
    lo = float(min(z_true, lo_all)) - 0.3
    hi = float(max(z_true, hi_all)) + 0.3
    ax.axhline(z_true, color="#b23a48", ls="--", lw=1.2, label="true value")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true " + labels[k] + " (jittered for visibility)", fontsize=9)
    ax.set_ylabel("estimated " + labels[k], fontsize=9)
    ax.set_title(labels[k], fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)
fig.suptitle("True-versus-estimated scatterplots: five trial lengths", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(paths.figure_path("fig_s4_multit_scatter.png"), dpi=200)
plt.close(fig)

print("SAVED fig_s3_grid_scatter.png and fig_s4_multit_scatter.png")
