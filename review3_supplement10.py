# -*- coding: utf-8 -*-
"""
review3_supplement10.py  (V16 response to PLOS Comp Biol reviewer)
=================================================================
Candidate-model FITTING-PARITY audit: a return-order bug and a
common-bounds sensitivity.

The V15 review (Major concern 2) asked whether the nested fixed-gain
restricted adaptive filter was fitted on the same footing as the proposed
four-parameter model. Re-examining competitors.py revealed TWO parity issues:

  (1) RETURN-ORDER BUG (the dominant issue).
      competitors.fit_kalman optimises with decision variables
      z = (mu, log_omega, log_sigma) and evaluates the objective in the
      CORRECT order (mu, exp(log_sigma), exp(log_omega)) = (mu, sigma, omega),
      but it RETURNS (best_z[0], exp(best_z[1]), exp(best_z[2])) =
      (mu, omega, sigma). forward_kalman unpacks (mu, sigma, omega), so the
      fitted prediction swaps sigma and omega. For a representative
      fixed-gain dataset this raises the fitted block NLL from ~50.5 to
      ~1493 -- the fixed-gain filter was therefore almost never selected,
      an artifact of the interface rather than genuine non-discrimination.

  (2) FITTING-BOUND MISMATCH (the V15 reviewer's stated concern).
      Even at the equation level the fixed-gain candidate was fitted with
      wider bounds than the proposed model:
        proposed : mu [-6,6], log sigma [-3,2.5], log omega [-4,1]
        fixed-gain: mu [-8,8], log sigma [-6,6],  log omega [-8,4].

This script reruns the SAME block model-recovery protocol (T=80; 50 datasets
per generator; origins 0.30/0.55/0.80; 56 scored outcomes; L-BFGS-B;
n_starts=2; identical starting draws and likelihood) under three specs:

    V15_PROTOCOL            : legacy fit_kalman (return-order bug, wide bounds)
    CORRECTED_ORIG_BOUNDS   : fixed return order, wide bounds
    CORRECTED_COMMON_BOUNDS : fixed return order, common (proposed-model) bounds

Each gives the confusion matrix, overall/row-wise accuracy with Wilson
intervals, marginal winner counts, and the calibrated 50,000-permutation null
(fixed scores/winners, only generator labels permuted; same construction as
review3_supplement9). V15_PROTOCOL must reproduce the archived 58/150.

Output: review3_diagnostics10.json (fixed seed; deterministic).
"""

import os
import json
import math
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, CODE_DIR)

from scipy.optimize import minimize  # noqa: E402

from adaptive_learning_gain import simulate                    # noqa: E402
from competitors import (simulate_rw, forward_rw,   # noqa: E402
                         simulate_kalman, forward_kalman,
                         fit_kalman_pre_correction)
import review3_supplement7 as r7                    # noqa: E402
import paths                                        # noqa: E402

# Single shared historical-control implementation (kept identical to the
# pre-V16 return order); aliased to preserve the V15-protocol naming below.
fit_kalman_legacy_local = fit_kalman_pre_correction

SEED = 0
T = r7.T
R = r7.R
MODELS = r7.MODELS
REPS = r7.REPS
ORIGINS = [int(T * 0.30), int(T * 0.55), int(T * 0.80)]

# fit_kalman decision-vector order: (mu, log_omega, log_sigma)
KALMAN_ORIGINAL_BOUNDS = [(-8, 8), (-8, 4), (-6, 6)]   # mu, log_omega, log_sigma
KALMAN_COMMON_BOUNDS = [(-6, 6), (-4, 1), (-3, 2.5)]   # mu, log_omega, log_sigma

TAGS = ["v15_protocol", "corrected_original_bounds", "corrected_common_bounds"]
ARCHIVED_K = 58


def fit_kalman_bounded(y, bounds, n_starts=2, seed=0):
    """Fixed-gain fit with explicit bounds and CORRECT return order.

    z = (mu, log_omega, log_sigma); objective identical to the legacy
    implementation but returns (mu, sigma, omega) as forward_kalman expects.
    """
    def _nll(z):
        mu, log_omega, log_sigma = z
        theta = (mu, float(np.exp(log_sigma)), float(np.exp(log_omega)))
        ps = np.clip(forward_kalman(y, theta), 1e-12, 1.0 - 1e-12)
        return -float(np.sum(y * np.log(ps) + (1 - y) * np.log(1.0 - ps)))

    rng = np.random.default_rng(seed)
    best_z, best_nll = None, np.inf
    for _i in range(n_starts):
        z0 = np.array([rng.uniform(-3, 3), rng.uniform(-2, 1), rng.uniform(-1, 2)])
        res = minimize(_nll, z0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_nll:
            best_nll, best_z = float(res.fun), res.x
    theta = (float(best_z[0]), float(np.exp(best_z[2])), float(np.exp(best_z[1])))
    return theta, best_nll


# The V15 historical-control fit is `fit_kalman_legacy_local`, aliased at the
# top of this file to the shared competitors.fit_kalman_pre_correction (which
# reproduces the pre-V16 return-order swap exactly).


def _ce(p, yy):
    p = r7._clip(p)
    return -(yy * math.log(p) + (1 - yy) * math.log(1.0 - p))


def score_dataset(y):
    """Full-precision block scores under all three specs; adaptive_filter/rw fit once."""
    block = {tag: {m: 0.0 for m in MODELS} for tag in TAGS}
    nb = 0
    # separate counts of corrected wide-bounds fits whose optimum lies outside
    # the common bounds (total and by parameter)
    out_bounds = {"mu": 0, "sigma": 0, "omega": 0, "any": 0}
    n_kal = 0
    for bi, cut in enumerate(ORIGINS):
        end = ORIGINS[bi + 1] if bi + 1 < len(ORIGINS) else len(y)

        th_af, _ = r7.FIT["adaptive_filter"](y[:cut], 2)
        ps_af = r7.PRED["adaptive_filter"](y[:end], th_af)
        th_r, _ = r7.FIT["rw"](y[:cut], 2)
        ps_r = r7.PRED["rw"](y[:end], th_r)

        # legacy (return-order bug, wide bounds)
        th_leg, _ = fit_kalman_legacy_local(y[:cut], n_starts=2, seed=0)
        ps_leg = forward_kalman(y[:end], th_leg)
        # corrected, wide bounds
        th_cw, _ = fit_kalman_bounded(y[:cut], KALMAN_ORIGINAL_BOUNDS)
        ps_cw = forward_kalman(y[:end], th_cw)
        # corrected, common bounds
        th_cc, _ = fit_kalman_bounded(y[:cut], KALMAN_COMMON_BOUNDS)
        ps_cc = forward_kalman(y[:end], th_cc)

        mu_w, sigma_w, omega_w = th_cw
        log_s, log_w = math.log(sigma_w), math.log(omega_w)
        hit_mu = abs(mu_w) > 6 + 1e-9
        hit_s = log_s > 2.5 + 1e-9 or log_s < -3 - 1e-9
        hit_w = log_w > 1 + 1e-9 or log_w < -4 - 1e-9
        if hit_mu:
            out_bounds["mu"] += 1
        if hit_s:
            out_bounds["sigma"] += 1
        if hit_w:
            out_bounds["omega"] += 1
        if hit_mu or hit_s or hit_w:
            out_bounds["any"] += 1
        n_kal += 1

        for t in range(cut, end):
            block["v15_protocol"]["adaptive_filter"] += _ce(ps_af[t], y[t])
            block["v15_protocol"]["rw"] += _ce(ps_r[t], y[t])
            block["v15_protocol"]["kalman_fixed"] += _ce(ps_leg[t], y[t])
            block["corrected_original_bounds"]["adaptive_filter"] += _ce(ps_af[t], y[t])
            block["corrected_original_bounds"]["rw"] += _ce(ps_r[t], y[t])
            block["corrected_original_bounds"]["kalman_fixed"] += _ce(ps_cw[t], y[t])
            block["corrected_common_bounds"]["adaptive_filter"] += _ce(ps_af[t], y[t])
            block["corrected_common_bounds"]["rw"] += _ce(ps_r[t], y[t])
            block["corrected_common_bounds"]["kalman_fixed"] += _ce(ps_cc[t], y[t])
        nb += end - cut

    return ({tag: {m: block[tag][m] / nb for m in MODELS} for tag in TAGS},
            out_bounds, n_kal)


def package(S, g, n_perm=50000, perm_seed=20241018):
    winners = np.argmin(S, axis=1)
    n = len(g)
    conf = {m: {m2: 0 for m2 in MODELS} for m in MODELS}
    for gi, wi in zip(g, winners):
        conf[MODELS[int(gi)]][MODELS[int(wi)]] += 1

    overall_k = int(sum(conf[m][m] for m in MODELS))
    overall_acc = overall_k / n
    marg = {MODELS[j]: int(np.sum(winners == j)) for j in range(len(MODELS))}
    obs_row = {m: conf[m][m] for m in MODELS}
    null_row_rate = {m: marg[m] / n for m in MODELS}
    null_row_expected = {m: round(null_row_rate[m] * R, 2) for m in MODELS}

    rng = np.random.default_rng(perm_seed)
    perm_acc = np.empty(n_perm)
    perm_row = {m: np.empty(n_perm) for m in MODELS}
    for b in range(n_perm):
        gp = rng.permutation(g)
        perm_acc[b] = float((winners == gp).mean())
        for j, m in enumerate(MODELS):
            perm_row[m][b] = float(np.sum((gp == j) & (winners == j))) / R

    def pct(a, qs):
        return {f"p{int(q * 100)}": round(float(np.quantile(a, q)), 4)
                for q in qs}

    p_overall = (1 + np.sum(perm_acc >= overall_acc - 1e-12)) / (n_perm + 1)
    p_row = {m: round((1 + np.sum(perm_row[m] >=
                                  (obs_row[m] / R) - 1e-12)) / (n_perm + 1), 4)
             for m in MODELS}

    return {
        "confusion_counts": {m: conf[m] for m in MODELS},
        "confusion_row_proportion": {m: {m2: round(conf[m][m2] / R, 3)
                                         for m2 in MODELS} for m in MODELS},
        "accuracy_by_true_model": {m: round(conf[m][m] / R, 3) for m in MODELS},
        "overall_k": overall_k,
        "overall_n": n,
        "overall_accuracy": round(overall_acc, 4),
        "overall_wilson95": r7._wilson(overall_k, n),
        "row_wilson95": {m: r7._wilson(conf[m][m], R) for m in MODELS},
        "chance": round(1.0 / len(MODELS), 3),
        "marginal_predicted_winner_counts": marg,
        "marginal_predicted_winner_fraction":
            {m: round(marg[m] / n, 3) for m in MODELS},
        "observed_row_correct": obs_row,
        "null_per_row_expected_rate":
            {m: round(null_row_rate[m], 3) for m in MODELS},
        "null_per_row_expected_count": null_row_expected,
        "n_permutations": n_perm,
        "permutation_seed": perm_seed,
        "null_overall_mean": round(float(perm_acc.mean()), 4),
        "null_overall_sd": round(float(perm_acc.std()), 4),
        "null_overall_quantiles":
            pct(perm_acc, [0.025, 0.05, 0.5, 0.95, 0.975]),
        "permutation_p_overall": round(float(p_overall), 4),
        "permutation_p_per_row": p_row,
        "null_per_row_quantiles":
            {m: pct(perm_row[m], [0.025, 0.5, 0.975]) for m in MODELS},
    }


def main():
    out = {"seed": SEED, "T": T, "R": R, "origins": ORIGINS,
           "scored_future_outcomes_per_dataset": 56,
           "bounds_comparison": {
               "proposed_model_bounds": {
                   "mu": [-6, 6], "log_sigma": [-3, 2.5],
                   "log_omega": [-4, 1], "log_lambda": [-4, 4]},
               "fixed_gain_original_bounds": {
                   "mu": [-8, 8], "log_sigma": [-6, 6], "log_omega": [-8, 4]},
               "fixed_gain_common_bounds": {
                   "mu": [-6, 6], "log_sigma": [-3, 2.5], "log_omega": [-4, 1]},
               "shared_parameters": ["mu", "sigma", "omega"],
               "fit_parity": ("corrected fixed-gain fits use the same L-BFGS-B, "
                              "n_starts=2 per prefix, starting draws (seed=0), "
                              "likelihood (forward_kalman = proposed forward with "
                              "lambda=1) and convergence tolerances; the three "
                              "specs differ only in (a) the return order and "
                              "(b) the bounds.")},
           "row_wise_note": ("The overall permutation test is the primary "
                            "inferential result; row-wise p-values are "
                            "descriptive, uncorrected, and not used to claim "
                            "discrimination.")}

    S = {tag: [] for tag in TAGS}
    g = []
    out_bounds_total = {"mu": 0, "sigma": 0, "omega": 0, "any": 0}
    n_kal_total = 0
    idx = 0
    for mt in MODELS:
        for r in range(R):
            if mt == "adaptive_filter":
                y = simulate(REPS[mt], T, SEED + r)
            elif mt == "rw":
                y = simulate_rw(REPS[mt][0], T, SEED + r)
            else:
                y = simulate_kalman(REPS[mt], T, SEED + r)
            scores, ob, n_kal = score_dataset(y)
            for tag in TAGS:
                S[tag].append([scores[tag][m] for m in MODELS])
            g.append(MODELS.index(mt))
            for k in out_bounds_total:
                out_bounds_total[k] += ob[k]
            n_kal_total += n_kal
            idx += 1
            if idx % 15 == 0:
                print("  scored", idx, "/", R * len(MODELS), flush=True)

    g = np.asarray(g, dtype=int)
    pkgs = {}
    for tag in TAGS:
        print("packaging", tag, "...", flush=True)
        pkgs[tag] = package(np.asarray(S[tag], dtype=float), g)
        out[tag] = pkgs[tag]
        print("  k=%d acc=%.4f p=%.4f fixed_gain_row=%d"
              % (pkgs[tag]["overall_k"], pkgs[tag]["overall_accuracy"],
                 pkgs[tag]["permutation_p_overall"],
                 pkgs[tag]["observed_row_correct"]["kalman_fixed"]), flush=True)

    # winner differences between specs
    W = {tag: np.argmin(np.asarray(S[tag]), axis=1) for tag in TAGS}
    out["comparison"] = {
        "matches_archived_v15": bool(pkgs["v15_protocol"]["overall_k"] == ARCHIVED_K),
        "overall_k_chain": {tag: pkgs[tag]["overall_k"] for tag in TAGS},
        "overall_accuracy_chain": {tag: pkgs[tag]["overall_accuracy"] for tag in TAGS},
        "permutation_p_overall_chain":
            {tag: pkgs[tag]["permutation_p_overall"] for tag in TAGS},
        "fixed_gain_row_correct_chain":
            {tag: pkgs[tag]["observed_row_correct"]["kalman_fixed"] for tag in TAGS},
        "fixed_gain_winner_count_chain":
            {tag: pkgs[tag]["marginal_predicted_winner_counts"]["kalman_fixed"]
             for tag in TAGS},
        "datasets_winner_changed_v15_to_corrected":
            int(np.sum(W["v15_protocol"] != W["corrected_original_bounds"])),
        "datasets_winner_changed_wide_to_common":
            int(np.sum(W["corrected_original_bounds"] != W["corrected_common_bounds"])),
        "corrected_wide_fixed_gain_fits_outside_common_bounds": out_bounds_total,
        "fixed_gain_prefix_fits_total": n_kal_total,
        "conclusion": ("Separates the return-order correction (58->corrected) "
                       "from the bound effect (corrected wide->common).")}

    path = paths.output_path("review3_diagnostics10.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("SAVED", path, flush=True)


if __name__ == "__main__":
    main()
