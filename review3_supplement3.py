# -*- coding: utf-8 -*-
"""review3_supplement3.py —— 第六轮审稿补充计算：
A. 正确设定恢复实验的 true-vs-fitted NLL 审计（MC1）
B. 配对 OOS log-loss 差分布的 CI 与胜出计数（MC5，从 diagnostics2 计算）
注：图 S1（1D profiles）与 S2（2D trade-off surfaces）由 supplement_figs.py 生成。
输出 review3_diagnostics3.json。
"""
import json
import io
import os
import numpy as np

import paths
CODE_DIR = paths.CODE_DIR
OUT_JSON = paths.output_path("review3_diagnostics3.json")

import sys
sys.path.insert(0, CODE_DIR)
from adaptive_learning_gain import simulate, fit, nll

BOUNDS = [(-6.0, 6.0), (-3.0, 2.5), (-4.0, 1.0), (-4.0, 4.0)]  # log-space

# ---------- A. true-vs-fitted NLL audit, correctly specified recovery ----------
THETA_STAR = [0.5, 2.0, 0.05, 3.0]
T = 80
R = 50

nll_true_list = []
nll_fit_list = []
delta_list = []
boundary_counts = np.zeros(4)
for r in range(R):
    y = simulate(THETA_STAR, T, seed=r)
    nll_true = nll(y, THETA_STAR)
    theta_hat, nll_fit_v = fit(y, n_starts=4, seed=r, bounds=BOUNDS)
    nll_true_list.append(nll_true)
    nll_fit_list.append(nll_fit_v)
    delta_list.append(nll_fit_v - nll_true)
    z = np.array([theta_hat[0], np.log(theta_hat[1]), np.log(theta_hat[2]), np.log(theta_hat[3])])
    boundary_counts += (np.abs(z - np.array([b[0] for b in BOUNDS])) < 1e-6) | \
                       (np.abs(z - np.array([b[1] for b in BOUNDS])) < 1e-6)

def stat(x):
    a = np.asarray(x, dtype=float)
    se = a.std(ddof=1) / np.sqrt(a.size)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()), "se": float(se),
            "ci95": [float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se)]}

audit = {
    "analysis": "correctly-specified recovery likelihood audit",
    "generating_theta": THETA_STAR, "T": T, "R": R, "n_starts": 4, "bounds_log": BOUNDS,
    "nll_true": stat(nll_true_list),
    "nll_fitted": stat(nll_fit_list),
    "delta_fitted_minus_true": stat(delta_list),
    "fraction_fitted_le_true": float(np.mean(np.array(delta_list) <= 0.0)),
    "boundary_hit_fraction": [float(boundary_counts[k] / R) for k in range(4)],
    "note": "nll evaluated with the identical forward recursion and indexing as the fitted objective; "
            "delta = nll(theta_hat) - nll(theta_star); negative delta means the fitted optimum improves on the true vector.",
}

# ---------- B. paired OOS difference CI (from diagnostics2) ----------
d2 = json.load(io.open(paths.data_path("review3_diagnostics2.json"), encoding="utf-8"))
ood = d2["oos_diff_distribution"]
def paired_ci(mean, sd, n):
    se = sd / np.sqrt(n)
    return [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)]
oos_ci = {
    "n_datasets": ood["R"],
    "diff_model_vs_empirical_freq": {
        "mean": ood["diff_model_vs_empirical_freq"]["mean"],
        "sd": ood["diff_model_vs_empirical_freq"]["sd"],
        "ci95_mean": paired_ci(ood["diff_model_vs_empirical_freq"]["mean"],
                               ood["diff_model_vs_empirical_freq"]["sd"], ood["R"]),
        "fraction_model_worse": ood["diff_model_vs_empirical_freq"]["fraction_model_worse"],
        "fraction_model_better": 1.0 - ood["diff_model_vs_empirical_freq"]["fraction_model_worse"],
    },
    "diff_model_vs_s_mu_hat": {
        "mean": ood["diff_model_vs_s_mu_hat"]["mean"],
        "sd": ood["diff_model_vs_s_mu_hat"]["sd"],
        "ci95_mean": paired_ci(ood["diff_model_vs_s_mu_hat"]["mean"],
                               ood["diff_model_vs_s_mu_hat"]["sd"], ood["R"]),
        "fraction_model_worse": ood["diff_model_vs_s_mu_hat"]["fraction_model_worse"],
        "fraction_model_better": 1.0 - ood["diff_model_vs_s_mu_hat"]["fraction_model_worse"],
    },
    "note": "paired differences computed per dataset (same split rule, same generating parameters); "
            "CI is a normal-approximation 95% CI for the mean paired difference.",
}

result = {"seed": 0, "likelihood_audit": audit, "oos_paired_ci": oos_ci}

with io.open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
print("SAVED", OUT_JSON)
print("nll_true mean", audit["nll_true"]["mean"], "nll_fit mean", audit["nll_fitted"]["mean"],
      "delta mean", audit["delta_fitted_minus_true"]["mean"], "sd", audit["delta_fitted_minus_true"]["sd"],
      "frac fit<=true", audit["fraction_fitted_le_true"], "boundary frac", audit["boundary_hit_fraction"])
print("OOS CI emp-freq", oos_ci["diff_model_vs_empirical_freq"]["ci95_mean"],
      "wins", oos_ci["diff_model_vs_empirical_freq"]["fraction_model_better"],
      "s-mu CI", oos_ci["diff_model_vs_s_mu_hat"]["ci95_mean"],
      "wins", oos_ci["diff_model_vs_s_mu_hat"]["fraction_model_better"])
