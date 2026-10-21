# -*- coding: utf-8 -*-
"""supplement_figs.py —— 生成第六轮审稿补充图：
Fig S1: 1D profile negative log-likelihood (mu, sigma, omega, lambda), MLE/边界/参考线标注
Fig S2: 2D trade-off surfaces (sigma x omega, sigma x lambda, omega x lambda), delta-NLL 等值
输出 figures/fig_s1_profiles.png, fig_s2_tradeoffs.png（复现模式下写入 regenerated/figures/）
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

d = json.load(io.open(paths.data_path("review3_diagnostics.json"), encoding="utf-8"))
profiles = d["profile_abs_change"]
trades = d["tradeoff_surfaces"]

# ---------- Fig S1: 1D profiles ----------
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
labels = {"mu": r"$\mu$ (log-odds)", "sigma": r"$\sigma$ ($\log$ scale)",
          "omega": r"$\omega$ ($\log$ scale)", "lambda": r"$\lambda$ ($\log$ scale)"}
for ax, key in zip(axes.ravel(), ["mu", "sigma", "omega", "lambda"]):
    p = profiles[key]
    x = np.asarray(p["x_grid"]); y = np.asarray(p["profile_nll"])
    ax.plot(x, y, "-o", ms=4, lw=1.4, color="#1f6fb2")
    nll_mle = p["nll_mle"]
    ax.axhline(nll_mle, color="#b23a48", ls="--", lw=1.0)
    ax.axvline(0.0, color="0.65", ls=":", lw=1.0)
    # annotate MLE + reference line
    imin = int(np.argmin(y))
    ax.annotate("MLE", (x[imin], y[imin]), textcoords="offset points", xytext=(4, -12),
                fontsize=8, color="#b23a48")
    ax.text(0.02, 0.96, r"$\Delta$NLL$_{\max}$ = %.2f" % p["delta_nll_abs_max_min"],
            transform=ax.transAxes, fontsize=9, va="top")
    ax.set_title(labels[key], fontsize=11)
    ax.set_xlabel("parameter value" + (" ($\\log$ scale)" if key != "mu" else " (log-odds)"), fontsize=9)
    ax.set_ylabel("profile negative log-likelihood", fontsize=9)
    ax.grid(alpha=0.25)
    if key == "lambda":
        ax.set_ylim(78, None)
fig.suptitle("One-dimensional profile negative log-likelihoods", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(paths.figure_path("fig_s1_profiles.png"), dpi=200)
plt.close(fig)

# ---------- Fig S2: 2D trade-off surfaces ----------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
titles = {("sigma_omega"): r"$\sigma \times \omega$", ("sigma_lambda"): r"$\sigma \times \lambda$",
          ("omega_lambda"): r"$\omega \times \lambda$"}
for ax, (tk, t) in zip(axes, trades.items()):
    xi = t["axis_i"]; xj = t["axis_j"]
    X, Y = np.meshgrid(xi["grid"], xj["grid"])
    Z = np.asarray(t["delta_nll_matrix"]).T
    c = ax.contourf(X, Y, Z, levels=16, cmap="viridis")
    ax.contour(X, Y, Z, levels=[1.0, 2.0, 5.0], colors="white", linewidths=0.6, alpha=0.8)
    fig.colorbar(c, ax=ax, shrink=0.9, label=r"$\Delta$NLL (re-optimising others)")
    ax.set_xlabel("$\\log$ " + xi["name"], fontsize=9)
    ax.set_ylabel("$\\log$ " + xj["name"], fontsize=9)
    ax.set_title(titles[tk], fontsize=11)
fig.suptitle("Two-dimensional parameter trade-off surfaces", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(paths.figure_path("fig_s2_tradeoffs.png"), dpi=200)
plt.close(fig)

print("SAVED fig_s1_profiles.png and fig_s2_tradeoffs.png")
