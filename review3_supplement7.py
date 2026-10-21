# -*- coding: utf-8 -*-
"""
review3_supplement7.py  (V14 response to PLOS Comp Biol reviewer)
=================================================================
Three additions requested in the V13 review (computational validation revision):

  (MC3) MODEL RECOVERY WITH A HIGH-INFORMATION FORECAST SCORE
       The rolling-origin model-recovery score previously used ONE future
       observation per cut (3 scored outcomes per dataset). We now score a
       meaningful subsequent BLOCK at each origin (fit on the prefix, then
       sequential one-step-ahead predictions over the block to the next
       origin), so every candidate scores the SAME 56 held-out outcomes per
       dataset (T=80; origins 0.30/0.55/0.80; blocks 24-43, 44-63, 64-79).
       Parameters are refit on each prefix (no look-ahead); latent states are
       propagated online inside the block with the prefix-fitted parameters.
       The old 3-point score is recomputed on the identical fits and retained
       only as an explicit low-information pilot.
       Competitor labels: "kalman_fixed" is reported as a FIXED-GAIN
       RESTRICTED ADAPTIVE FILTER (the proposed filter with lambda fixed at 1),
       NOT a standard Kalman filter.

  (MC2) INDEPENDENT LIKELIHOOD IMPLEMENTATION CHECK
       A second, fully independent recursion written in plain Python (no
       numpy, no shared helpers, scalar math.exp, different variable names)
       is compared element-by-element against adaptive_learning_gain.forward/nll on
       many random sequences, including the worked-example and extreme
       sequences. This is a genuinely independent implementation check,
       distinct from the same-codebase unit tests.

  (MC1) PROFILE LIKELIHOOD IN A SECOND, QUALITATIVELY DIFFERENT REGIME
       The published profiles used the representative high-gain vector
       (0,3,0.02,5). We recompute the one-dimensional profiles for the
       low-gain vector (-1,1.5,0.02,2) (lambda=2 sits at the edge of the flat
       low-lambda region; smaller sigma) to test whether the flat geometry
       recurs in a regime with a different boundary pattern.

Output: review3_diagnostics7.json (fixed seed; deterministic).
"""

import os
import json
import math
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, CODE_DIR)

from scipy.optimize import minimize  # noqa: E402

import review3_supplement as rs                      # noqa: E402
from adaptive_learning_gain import simulate, nll, forward      # noqa: E402
from competitors import (simulate_rw, forward_rw, fit_rw,        # noqa: E402
                          simulate_kalman, forward_kalman,
                          fit_kalman_pre_correction)  # noqa: E402
import paths                                          # noqa: E402

SEED = 0
T = 80
R = 50
MODELS = ["adaptive_filter", "rw", "kalman_fixed"]
REPS = {"adaptive_filter": (0.5, 2.0, 0.05, 3.0),
        "rw": (0.30,),
        "kalman_fixed": (0.5, 2.0, 0.05)}


# ============================================================================
# MC3 : block-forecast rolling-origin scoring
# ============================================================================

def _fit_adaptive(y, ns):
    th, nv = rs.fit(y, n_starts=2, seed=0)
    return th, nv


def _fit_rw(y, ns):
    a, nv = fit_rw(y, seed=0)
    return a, nv


def _fit_kalman(y, ns):
    # Pre-correction (historical-control) routine. The block result here
    # reproduces the archived 58/150 (Table S8 control row, Text S6) and
    # the single-point pilot reproduces 57/150 (Table S5). The corrected
    # block protocol (67/150 wide, 72/150 common) is computed separately in
    # review3_supplement10.py and is the primary result.
    th, nv = fit_kalman_pre_correction(y, n_starts=2, seed=0)
    return th, nv


FIT = {"adaptive_filter": _fit_adaptive, "rw": _fit_rw, "kalman_fixed": _fit_kalman}
PRED = {"adaptive_filter": forward, "rw": forward_rw, "kalman_fixed": forward_kalman}


def _clip(p):
    return min(max(float(p), 1e-12), 1.0 - 1e-12)


def _scores_one_dataset(y, origins):
    """Return (block mean log-loss, single-point mean log-loss) per candidate.

    Both scores are computed from the SAME prefix fits, so the only difference
    is how many future outcomes are scored (block: to the next origin;
    single-point: just the first future outcome at each origin).
    """
    Tn = len(y)
    block = {m: 0.0 for m in MODELS}
    point = {m: 0.0 for m in MODELS}
    nb = 0
    for bi, cut in enumerate(origins):
        end = origins[bi + 1] if bi + 1 < len(origins) else Tn
        for m in MODELS:
            th, _ = FIT[m](y[:cut], 2)
            ps = PRED[m](y[:end], th)
            ll_block = 0.0
            ll_point = 0.0
            for t in range(cut, end):
                p = _clip(ps[t])
                ce = -(y[t] * math.log(p) + (1 - y[t]) * math.log(1.0 - p))
                ll_block += ce
                if t == cut:
                    ll_point += ce
            block[m] += ll_block
            point[m] += ll_point
        nb += (end - cut)
    return ({m: block[m] / nb for m in MODELS},
            {m: point[m] / len(origins) for m in MODELS},
            nb)


def _wilson(k, n, z=1.96):
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / den
    return [round(centre - half, 4), round(centre + half, 4)]


def run_model_recovery_block():
    origins = [int(T * 0.30), int(T * 0.55), int(T * 0.80)]
    n_scored = sum((origins[i + 1] if i + 1 < len(origins) else T) - origins[i]
                   for i in range(len(origins)))

    def _empty_conf():
        return {m: {m2: 0 for m2 in MODELS} for m in MODELS}
    conf_block, conf_point = _empty_conf(), _empty_conf()
    margin_block, margin_point = {m: [] for m in MODELS}, {m: [] for m in MODELS}
    per_ds_block = {m: [] for m in MODELS}

    for mt in MODELS:
        for r in range(R):
            if mt == "adaptive_filter":
                y = simulate(REPS[mt], T, SEED + r)
            elif mt == "rw":
                y = simulate_rw(REPS[mt][0], T, SEED + r)
            else:
                y = simulate_kalman(REPS[mt], T, SEED + r)
            sb, sp, _ = _scores_one_dataset(y, origins)
            per_ds_block[mt].append({m: round(float(sb[m]), 5) for m in MODELS})
            for scores, conf, margin in (
                    (sb, conf_block, margin_block),
                    (sp, conf_point, margin_point)):
                order = sorted(MODELS, key=lambda m: scores[m])
                conf[mt][order[0]] += 1
                margin[mt].append(scores[order[0]] - scores[order[1]])

    def _package(conf, margin, label, n_per_ds):
        acc = {m: conf[m][m] / R for m in MODELS}
        return {
            "label": label,
            "R": R, "T": T, "origins": origins,
            "scored_future_outcomes_per_dataset": n_per_ds,
            "confusion_counts": {m: conf[m] for m in MODELS},
            "confusion_row_proportion": {m: {m2: round(conf[m][m2] / R, 3)
                                             for m2 in MODELS} for m in MODELS},
            "accuracy_by_true_model": {m: round(conf[m][m] / R, 3) for m in MODELS},
            "overall_accuracy": round(sum(acc.values()) / len(MODELS), 4),
            "overall_k": int(sum(conf[m][m] for m in MODELS)),
            "overall_n": R * len(MODELS),
            "overall_wilson95": _wilson(sum(conf[m][m] for m in MODELS),
                                        R * len(MODELS)),
            "row_wilson95": {m: _wilson(conf[m][m], R) for m in MODELS},
            "chance": round(1.0 / len(MODELS), 3),
            "winner_minus_runnerup": {m: {
                "mean": round(float(np.mean(margin[m])), 4),
                "sd": round(float(np.std(margin[m])), 4),
                "p25": round(float(np.percentile(margin[m], 25)), 4),
                "p75": round(float(np.percentile(margin[m], 75)), 4)}
                for m in MODELS},
        }

    block_res = _package(conf_block, margin_block,
                         "block-forecast rolling-origin (primary; "
                         "sequential one-step-ahead over the held-out block)",
                         n_scored)
    point_res = _package(conf_point, margin_point,
                         "single-point rolling-origin (low-information pilot, "
                         "one future outcome per origin)", len(origins))
    block_res["per_dataset_block_logloss"] = per_ds_block
    return block_res, point_res


# ============================================================================
# MC2 : genuinely independent likelihood implementation
# ============================================================================

def _logistic_scalar(x):
    """Standalone numerically stable logistic using only math (no numpy)."""
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def independent_forward(y, mu, sigma, omega, lam):
    """Independent plain-Python transcription of the filter recursion.

    Deliberately written independently (scalar math, separate names, explicit
    loop structure) so it does not share code with adaptive_learning_gain.forward; the
    only shared object is the input sequence.
    """
    belief = float(mu)
    prec = float(sigma)
    probs = []
    for obs in y:
        pred_prec = 1.0 / (1.0 / prec + omega)
        prob = _logistic_scalar(belief)
        slope = prob * (1.0 - prob)
        probs.append(prob)
        surprisal = float(obs) - prob
        belief = belief + (lam / pred_prec) * slope * surprisal
        prec = pred_prec + lam * slope * slope
    return probs


def independent_nll(y, theta, eps=1e-12):
    ps = independent_forward(y, *[float(a) for a in theta])
    tot = 0.0
    for obs, p in zip(y, ps):
        p = min(max(p, eps), 1.0 - eps)
        tot -= (obs * math.log(p) + (1.0 - obs) * math.log(1.0 - p))
    return tot


def run_independent_likelihood_check():
    rng = np.random.default_rng(20240517)
    max_pdiff = 0.0
    max_nlldiff = 0.0
    n_seq = 0
    thetas = [(0.5, 2.0, 0.05, 3.0),
              (0.0, 3.0, 0.02, 5.0),
              (-1.0, 1.5, 0.02, 2.0),
              (4.0, 5.0, 0.001, 0.001),
              (-2.0, 0.1, 1.0, 0.05),
              (1.5, 8.0, 0.5, 2.0)]
    for th in thetas:
        for Tn in (3, 5, 40, 120, 240):
            for s in range(5):
                y = simulate(th, Tn, seed=int(rng.integers(0, 10_000_000)))
                p_ref = forward(y, np.asarray(th, dtype=float))
                p_ind = independent_forward(y, *th)
                d = max(abs(float(a) - b) for a, b in zip(p_ref, p_ind))
                nref = float(nll(y, np.asarray(th, dtype=float)))
                nind = independent_nll(y, th)
                max_pdiff = max(max_pdiff, d)
                max_nlldiff = max(max_nlldiff, abs(nref - nind))
                n_seq += 1
    # explicit hand-calculated short sequence (first three trials)
    th = (0.5, 2.0, 0.05, 3.0)
    y5 = simulate(th, 5, seed=SEED + 12345)
    p5 = independent_forward(y5, *th)
    ext = [1] * 50 + [0] * 50
    nll_ext_ind = independent_nll(ext, (4.0, 5.0, 0.001, 0.001))
    nll_ext_ref = float(nll(np.asarray(ext, dtype=int),
                            np.asarray((4.0, 5.0, 0.001, 0.001))))
    return {
        "n_sequences_compared": n_seq,
        "max_abs_probability_diff": f"{max_pdiff:.3e}",
        "max_abs_nll_diff": f"{max_nlldiff:.3e}",
        "probability_tol": 1e-9,
        "nll_tol": 1e-6,
        "pass": bool(max_pdiff < 1e-9 and max_nlldiff < 1e-6),
        "worked_example_y": [int(a) for a in y5],
        "worked_example_independent_p_first3": [round(p5[i], 6) for i in range(3)],
        "worked_example_expectation_from_paper": [0.62246, 0.6562, 0.6847],
        "extreme_sequence_independent_nll": round(nll_ext_ind, 4),
        "extreme_sequence_reference_nll": round(nll_ext_ref, 4),
        "extreme_finite": bool(math.isfinite(nll_ext_ind)),
        "note": ("Independent plain-Python recursion (math.exp, no numpy in the "
                 "recursion, separate variable names); shares no likelihood "
                 "code with adaptive_learning_gain.py. Agreement with the main "
                 "implementation across many random sequences and parameter "
                 "regimes supports recursion/indexing/initialisation correctness.")}


# ============================================================================
# MC1 : profiles in a second, qualitatively different regime (low-gain)
# ============================================================================

def run_profile_regime(theta, Tn=120, seed=SEED):
    y = simulate(np.asarray(theta, dtype=float), Tn, seed=seed)
    th_mle, nll_mle, _, _ = rs.fit_diag(y, n_starts=15, seed=seed)
    res = {"generating_theta": list(theta), "T": Tn, "seed": seed,
           "mle_theta": [round(float(a), 4) for a in th_mle],
           "nll_mle": round(float(nll_mle), 3)}
    for j in range(4):
        lo, hi = (-4.0, 4.0) if j == 0 else rs._BOUNDS[j]
        pts = np.linspace(lo, hi, 13)
        vals = []
        for val in pts:
            best = np.inf
            rng = np.random.default_rng(seed + j * 100 + int((val - lo) / (hi - lo) * 100))
            for i in range(4):
                z0 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 2),
                               rng.uniform(-2, 0), rng.uniform(-1, 2)])
                z0 = np.clip(z0, [b[0] for b in rs._BOUNDS], [b[1] for b in rs._BOUNDS])
                z0[j] = val
                bb = [list(b) for b in rs._BOUNDS]
                bb[j] = [val, val]
                r = minimize(lambda z: nll(y, rs._untransform(z)), z0,
                             method="L-BFGS-B", bounds=bb)
                if r.fun < best:
                    best = float(r.fun)
            vals.append(best)
        vals = np.array(vals)
        res[rs._NAMES[j]] = {
            "x_grid": list(np.round(pts, 3)),
            "profile_nll": list(np.round(vals, 3)),
            "delta_nll_abs_max_min": round(float(vals.max() - vals.min()), 3),
            "nll_at_bounds": {"lo": round(float(vals[0]), 3),
                              "hi": round(float(vals[-1]), 3)},
            "max_dev_from_mle": round(float((vals - nll_mle).max()), 3)}
    return res


def main():
    out = {"seed": SEED, "T_model_recovery": T, "R": R,
           "competitor_label_note": ("kalman_fixed is a fixed-gain restricted "
                                     "adaptive filter (the proposed filter with "
                                     "lambda=1); it is NOT a standard Kalman "
                                     "filter. RW is generated at alpha=0.30 and "
                                     "fitted with a free learning rate.")}
    print("[1/3] block model recovery ...", flush=True)
    block_res, point_res = run_model_recovery_block()
    out["model_recovery_block"] = block_res
    out["model_recovery_single_point_pilot"] = point_res
    print("  block overall =", block_res["overall_accuracy"],
          block_res["overall_wilson95"], "k=", block_res["overall_k"], flush=True)
    print("  point overall =", point_res["overall_accuracy"],
          point_res["overall_wilson95"], flush=True)
    print("[2/3] independent likelihood check ...", flush=True)
    out["independent_likelihood_check"] = run_independent_likelihood_check()
    print("  pass =", out["independent_likelihood_check"]["pass"],
          "max_pdiff =", out["independent_likelihood_check"]["max_abs_probability_diff"],
          flush=True)
    print("[3/3] low-gain regime profiles ...", flush=True)
    out["profile_low_gain"] = run_profile_regime((-1.0, 1.5, 0.02, 2.0))
    print("  low-gain delta-NLL:",
          {k: out["profile_low_gain"][k]["delta_nll_abs_max_min"]
           for k in ["mu", "sigma", "omega", "lambda"]}, flush=True)
    path = paths.output_path("review3_diagnostics7.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("SAVED", path, flush=True)


if __name__ == "__main__":
    main()
