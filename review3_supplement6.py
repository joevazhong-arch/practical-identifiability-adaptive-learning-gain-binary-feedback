# -*- coding: utf-8 -*-
"""
review3_supplement6.py — V13 reviewer Major-5: re-run the 50-dataset fixed-split
out-of-sample comparison, keep the per-dataset log-losses, and add bootstrap
paired 95% CIs alongside the normal-approximation CI. Also records the range of
the training-set empirical frequency q to document how extreme (all-0/all-1)
training frequencies were handled.

Mirrors review3_supplement2.run_oos_diff (same seed, theta, T=120, train=72,
n_starts=8) so the normal-approximation numbers match; adds per-dataset rows,
paired bootstrap percentile CIs, and the empirical-frequency q range.
Output: review3_diagnostics6.json
"""
import json
import io
import os

import numpy as np

from adaptive_learning_gain import simulate, fit, forward, sigmoid
import paths

CODE_DIR = paths.CODE_DIR
OUT = paths.output_path("review3_diagnostics6.json")

SEED = 0
R = 50
T = 120
TRAIN = 72
BOOT_B = 10000
theta = np.array([0.5, 2.0, 0.05, 3.0])
rng = np.random.default_rng(12345)


def boot_ci(d, B=BOOT_B):
    d = np.asarray(d, dtype=float)
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


rows = []
qs = []
for r in range(R):
    y = simulate(theta, T, seed=SEED + r)
    ytr, yte = y[:TRAIN], y[TRAIN:]
    zhat, _ = fit(ytr, n_starts=8, seed=SEED + r)
    p = np.clip(forward(yte, zhat), 1e-12, 1 - 1e-12)
    ll_model = -np.mean(yte * np.log(p) + (1 - yte) * np.log(1 - p))
    p_mu = np.clip(sigmoid(np.array([zhat[0]] * len(yte))), 1e-12, 1 - 1e-12)
    ll_mu = -np.mean(yte * np.log(p_mu) + (1 - yte) * np.log(1 - p_mu))
    q_raw = float(ytr.mean())
    q = min(max(q_raw, 1e-6), 1 - 1e-6)   # numerical clamp, not Laplace smoothing
    ll_freq = -np.mean(yte * np.log(q) + (1 - yte) * np.log(1 - q))
    rows.append({"ll_model": float(ll_model), "ll_s_mu_hat": float(ll_mu),
                 "ll_empirical_freq": float(ll_freq)})
    qs.append(q_raw)

llm = np.array([x["ll_model"] for x in rows])
llmu = np.array([x["ll_s_mu_hat"] for x in rows])
llf = np.array([x["ll_empirical_freq"] for x in rows])


def block(diff):
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    return {
        "mean": round(mean, 4), "sd": round(sd, 4),
        "p25": round(float(np.percentile(diff, 25)), 4),
        "p75": round(float(np.percentile(diff, 75)), 4),
        "normal_ci95": [round(mean - 1.96 * sd / np.sqrt(n), 4),
                        round(mean + 1.96 * sd / np.sqrt(n), 4)],
        "bootstrap_ci95": boot_ci(diff),
        "fraction_model_better": round(float(np.mean(diff < 0)), 3),
        "fraction_model_worse": round(float(np.mean(diff > 0)), 3),
        "n": int(n),
    }


result = {
    "seed": SEED, "R": R, "T": T, "train": TRAIN, "generating_theta": list(theta),
    "model_logloss": {"mean": round(float(llm.mean()), 4), "sd": round(float(llm.std(ddof=1)), 4)},
    "empirical_freq_logloss": {"mean": round(float(llf.mean()), 4), "sd": round(float(llf.std(ddof=1)), 4)},
    "s_mu_hat_logloss": {"mean": round(float(llmu.mean()), 4), "sd": round(float(llmu.std(ddof=1)), 4)},
    "diff_model_vs_empirical_freq": block(llm - llf),
    "diff_model_vs_s_mu_hat": block(llm - llmu),
    "empirical_freq_construction": {
        "q": "training-set mean of binary outcomes, used as a CONSTANT predictor on held-out trials",
        "extreme_frequency_handling": "raw training MLE clamped to [1e-6, 1-1e-6] for numerical safety (not Laplace / Bayesian shrinkage)",
        "q_min_raw": round(min(qs), 4), "q_max_raw": round(max(qs), 4),
        "n_extreme_clamped": int(sum(1 for q in qs if q <= 0.0 or q >= 1.0)),
    },
    "per_dataset_rows": rows,
}
with io.open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=1)

print("model   ", result["model_logloss"])
print("empfreq ", result["empirical_freq_logloss"])
print("s_mu    ", result["s_mu_hat_logloss"])
print("diff vs emp-freq:", result["diff_model_vs_empirical_freq"])
print("diff vs s(mu):  ", result["diff_model_vs_s_mu_hat"])
print("emp-freq construction:", result["empirical_freq_construction"])
print("SAVED", OUT)
