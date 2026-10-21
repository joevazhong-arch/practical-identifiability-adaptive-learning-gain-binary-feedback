# -*- coding: utf-8 -*-
"""
review3_supplement.py
=====================
PLOS Comp Biol review3（Major revision）要求的计算证据升级。
只读 adaptive_learning_gain 库，固定 seed 复现。产出 review3_diagnostics.json：

  A) grid_recovery_r100   —— 24 条件 × R100（T=80, 8 starts）：
                            signed bias / MAE / RMSE / coverage / boundary，
                            且按生成条件逐条报告（审稿 MC1：按条件分开、高重复）。
  B) multi_T_recovery     —— 单设置 θ*=(0.5,2,0.05,3) × T∈{40,80,120,240,480} × R100。
  C) profile_abs_change   —— 四参数 profile 的 ΔNLL（绝对变化 max-min、相对 MLE 峰值）
                            （审稿 MC5：报告 profile 上 NLL 的绝对变化）。
  D) tradeoff_surfaces    —— (σ,ω),(σ,λ),(ω,λ) 三个二维面上固定该对、重优化其余两参
                            的最小 NLL 矩阵（审稿 MC5：参数权衡图数据）。
  E) recovery_r50         —— 3 生成模型 × R50（T=80）rolling-origin 模型恢复：
                            完整混淆矩阵（counts + 行归一）+ 每数据集得分差分布
                            （审稿 MC6：高重复 + 得分差 + 不确定性）。
  F) ablation_multi       —— 3 个生成设置 × R30：rolling log-loss + 拟合 NLL + 参数 SD
                            （审稿 MC8：多设置重复 ablation）。
  G) bound_sensitivity    —— high-gain 场景在 3 组替代边界下的 MLE 与边界命中
                            （审稿 3.3/MC5：边界敏感性）。
  H) unit_tests           —— simulator/forward 一致性；nll 有限差分；
                            概率边界安全性；首 3 个试次的完整数值示例（审稿 MC4）。
"""
import json
import os
import numpy as np
from scipy.optimize import minimize

import adaptive_learning_gain
from adaptive_learning_gain import simulate, nll, forward, sigmoid, default_grid, fit, _transform, _untransform
from competitors import (simulate_rw, forward_rw, nll_rw, fit_rw,
                         simulate_kalman, fit_kalman,
                         fit_kalman_pre_correction, forward_kalman)
from review2_diagnostics import (_transform as _tr, _untransform as _utr, _BOUNDS, _NAMES,
                                 fit_diag, _sim_variant, _forward_variant, _nll_variant,
                                 _fit_variant, _rolling_logloss)
import paths

OUT = paths.output_path("review3_diagnostics.json")

SEED = 0


def _rolling_logloss_clean(y, fit_fn, predict_fn, cuts=None, fit_starts=2):
    """rolling-origin 平均 log-loss（无前视泄漏）：

    在 cut 个试次上拟合，只用 y[:cut] 递推并取第 cut 步预测概率（不消费 y[cut]），
    用 y[cut] 记分。cuts 为绝对索引。
    """
    T = len(y)
    if cuts is None:
        cuts = [int(T * 0.25), int(T * 0.40), int(T * 0.60), int(T * 0.80)]
    ll = 0.0
    n = 0
    for cut in cuts:
        if cut < 20:
            continue
        th, _ = fit_fn(y[:cut], fit_starts)
        p = predict_fn(y[:cut], th)[-1]
        p = min(max(float(p), 1e-12), 1 - 1e-12)
        ll -= (y[cut] * np.log(p) + (1 - y[cut]) * np.log(1 - p))
        n += 1
    return ll / max(n, 1)


# ----------------------------------------------------------------------------
# A) 网格恢复 R100 + 按条件分组
# ----------------------------------------------------------------------------
def run_grid_recovery_r100(T=80, n_starts=4, R=100, tol=0.6, seed=SEED):
    grid = default_grid(n_mu=2, n_sigma=2, n_omega=2, n_lam=3, seed=seed)
    G = len(grid)
    per_cond = []
    agg = {k: np.zeros(4) for k in ("bias_log", "mae", "mse", "within", "bhit")}
    cnt = 0
    for g in range(G):
        th = grid[g]
        b = {k: np.zeros(4) for k in ("bias_log", "mae", "mse", "within", "bhit")}
        for r in range(R):
            y = simulate(th, T, seed=seed + g * R + r)
            hat, _, z, diag = fit_diag(y, n_starts=n_starts, seed=seed + g * R + r)
            b["bias_log"] += (z - _transform(th))
            b["mae"] += np.abs(_transform(th) - z)
            b["mse"] += (_transform(th) - z) ** 2
            b["within"] += (np.abs(_transform(th) - z) < tol).astype(float)
            b["bhit"] += np.array(diag["boundary_hit_rate"]) * n_starts
        for k in b:
            agg[k] += b[k]
        cnt += R
        per_cond.append({
            "condition": g, "theta": list(np.round(th, 4)),
            "signed_bias_log": np.round(b["bias_log"] / R, 3).tolist(),
            "mae_log": np.round(b["mae"] / R, 3).tolist(),
            "rmse_log": np.round(np.sqrt(b["mse"] / R), 3).tolist(),
            "coverage_within_tol": np.round(b["within"] / R, 3).tolist(),
            "boundary_hit_per_fit": np.round(b["bhit"] / R, 3).tolist(),
        })
    return {
        "n_conditions": G, "R": R, "T": T, "n_starts": n_starts, "tol_log": tol,
        "param_names": _NAMES,
        "signed_bias_log": np.round(agg["bias_log"] / cnt, 3).tolist(),
        "mae_log": np.round(agg["mae"] / cnt, 3).tolist(),
        "rmse_log": np.round(np.sqrt(agg["mse"] / cnt), 3).tolist(),
        "coverage_within_tol": np.round(agg["within"] / cnt, 3).tolist(),
        "boundary_hit_per_fit": np.round(agg["bhit"] / cnt, 3).tolist(),
        "per_condition": per_cond,
    }


# ----------------------------------------------------------------------------
# B) 多 T 恢复（单设置）
# ----------------------------------------------------------------------------
def run_multi_T(theta=None, Ts=(40, 80, 120, 240, 480), R=100, n_starts=4, seed=SEED):
    if theta is None:
        theta = np.array([0.5, 2.0, 0.05, 3.0])
    out = {}
    for T in Ts:
        b = {k: np.zeros(4) for k in ("bias_log", "mae", "mse", "within", "bhit")}
        for r in range(R):
            y = simulate(theta, T, seed=seed + T + r)
            hat, _, z, diag = fit_diag(y, n_starts=n_starts, seed=seed + T + r)
            b["bias_log"] += (z - _transform(theta))
            b["mae"] += np.abs(_transform(theta) - z)
            b["mse"] += (_transform(theta) - z) ** 2
            b["within"] += (np.abs(_transform(theta) - z) < 0.6).astype(float)
            b["bhit"] += np.array(diag["boundary_hit_rate"]) * n_starts
        out[str(T)] = {
            "signed_bias_log": np.round(b["bias_log"] / R, 3).tolist(),
            "mae_log": np.round(b["mae"] / R, 3).tolist(),
            "rmse_log": np.round(np.sqrt(b["mse"] / R), 3).tolist(),
            "coverage_within_tol": np.round(b["within"] / R, 3).tolist(),
            "boundary_hit_per_fit": np.round(b["bhit"] / R, 3).tolist(),
        }
    return {"generating_theta": list(theta), "R": R, "n_starts": n_starts, "by_T": out}


# ----------------------------------------------------------------------------
# C) Profile 绝对变化
# ----------------------------------------------------------------------------
def run_profile_abs(T=120, seed=SEED):
    y = simulate((0.0, 3.0, 0.02, 5.0), T, seed=seed)
    th_mle, nll_mle, z_mle, _ = fit_diag(y, n_starts=15, seed=seed)
    res = {}
    for j in range(4):
        if j == 0:
            lo, hi = -4.0, 4.0
        else:
            lo, hi = _BOUNDS[j]
        pts = np.linspace(lo, hi, 13)
        vals = []
        for val in pts:
            best = np.inf
            rng = np.random.default_rng(seed + j * 100 + int((val - lo) / (hi - lo) * 100))
            for i in range(4):
                z0 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 2),
                               rng.uniform(-2, 0), rng.uniform(-1, 2)])
                z0 = np.clip(z0, [b[0] for b in _BOUNDS], [b[1] for b in _BOUNDS])
                z0[j] = val
                bb = [list(b) for b in _BOUNDS]
                bb[j] = [val, val]
                r = minimize(lambda z: nll(y, _untransform(z)), z0, method="L-BFGS-B", bounds=bb)
                if r.fun < best:
                    best = float(r.fun)
            vals.append(best)
        vals = np.array(vals)
        res[_NAMES[j]] = {
            "x_grid": list(np.round(pts, 3)),
            "profile_nll": list(np.round(vals, 3)),
            "delta_nll_abs_max_min": round(float(vals.max() - vals.min()), 3),
            "nll_at_bounds": {"lo": round(float(vals[0]), 3), "hi": round(float(vals[-1]), 3)},
            "max_dev_from_mle": round(float((vals - nll_mle).max()), 3),
            "nll_mle": round(nll_mle, 3),
        }
    return res


# ----------------------------------------------------------------------------
# D) 参数权衡面数据（固定对，重优化其余两参）
# ----------------------------------------------------------------------------
def run_tradeoff(T=120, seed=SEED, grid_n=5):
    y = simulate((0.0, 3.0, 0.02, 5.0), T, seed=seed)
    pairs = [(1, 2), (1, 3), (2, 3)]   # (sigma,omega),(sigma,lambda),(omega,lambda)
    out = {}
    for (i, j) in pairs:
        lo_i, hi_i = _BOUNDS[i]
        lo_j, hi_j = _BOUNDS[j]
        xi = np.linspace(lo_i, hi_i, grid_n)
        xj = np.linspace(lo_j, hi_j, grid_n)
        M = np.full((grid_n, grid_n), np.nan)
        for a in range(grid_n):
            for b in range(grid_n):
                best = np.inf
                rng = np.random.default_rng(seed + i * 1000 + j * 100 + a * grid_n + b)
                for s in range(4):
                    z0 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 2),
                                   rng.uniform(-2, 0), rng.uniform(-1, 2)])
                    z0 = np.clip(z0, [b0[0] for b0 in _BOUNDS], [b0[1] for b0 in _BOUNDS])
                    z0[i] = xi[a]; z0[j] = xj[b]
                    bb = [list(b0) for b0 in _BOUNDS]
                    bb[i] = [xi[a], xi[a]]; bb[j] = [xj[b], xj[b]]
                    r = minimize(lambda z: nll(y, _untransform(z)), z0, method="L-BFGS-B", bounds=bb)
                    if r.fun < best:
                        best = float(r.fun)
                M[a, b] = best
        out[f"{_NAMES[i]}_{_NAMES[j]}"] = {
            "axis_i": {"name": _NAMES[i], "grid": list(np.round(xi, 3))},
            "axis_j": {"name": _NAMES[j], "grid": list(np.round(xj, 3))},
            "min_nll_matrix": [list(np.round(row, 3)) for row in M],
            "delta_nll_matrix": [list(np.round(row - M.min(), 3)) for row in M],
        }
    return out


# ----------------------------------------------------------------------------
# E) 模型恢复 R50 + 得分差分布
# ----------------------------------------------------------------------------
def run_recovery_r50(T=80, R=50, seed=SEED):
    models = ["adaptive_filter", "rw", "kalman_fixed"]
    reps = {"adaptive_filter": (0.5, 2.0, 0.05, 3.0), "rw": (0.30,),
            "kalman_fixed": (0.5, 2.0, 0.05)}
    conf = {m: {m2: 0 for m2 in models} for m in models}
    diffs = {m: [] for m in models}   # 每个数据集：最佳候选的 rolling-ll 与次佳之差（<0 表示识别成功）
    score_records = {m: [] for m in models}
    _CUTS = [int(T * 0.30), int(T * 0.55), int(T * 0.80)]
    def fit_adaptive(y, ns): return fit(y, n_starts=ns, seed=0)
    for mt in models:
        for r in range(R):
            th = reps[mt]
            if mt == "adaptive_filter":
                y = simulate(th, T, seed + r)
            elif mt == "rw":
                y = simulate_rw(th[0], T, seed + r)
            else:
                y = simulate_kalman(th, T, seed + r)
            scores = {}
            for m2 in models:
                if m2 == "adaptive_filter":
                    scores[m2] = _rolling_logloss_clean(y, fit_adaptive,
                                                        lambda yy, th2: forward(yy, th2),
                                                        cuts=_CUTS, fit_starts=2)
                elif m2 == "rw":
                    scores[m2] = _rolling_logloss_clean(y, lambda yy, ns: fit_rw(yy, seed=0),
                                                        forward_rw, cuts=_CUTS, fit_starts=2)
                else:
                    # Three-observation rolling-origin PILOT (Table S5): the
                    # fixed-gain candidate is fitted with the pre-correction
                    # (historical-control) routine so that this archived
                    # low-information pilot reproduces bit-for-bit (57/150).
                    # The corrected block protocol is in review3_supplement10.
                    scores[m2] = _rolling_logloss_clean(
                        y, fit_kalman_pre_correction, forward_kalman,
                        cuts=_CUTS, fit_starts=2)
            order = sorted(models, key=lambda m2: scores[m2])
            conf[mt][order[0]] += 1
            diffs[mt].append(scores[order[0]] - scores[order[1]])  # <0 且远离 0 => 可辨
            score_records[mt].append(scores)
    rows = {}
    for mt in models:
        rows[mt] = {"counts": conf[mt],
                    "row_proportion": {m2: round(conf[mt][m2] / R, 3) for m2 in models},
                    "winner_minus_runnerup_mean": round(float(np.mean(diffs[mt])), 4),
                    "winner_minus_runnerup_sd": round(float(np.std(diffs[mt])), 4),
                    "winner_minus_runnerup_p25": round(float(np.percentile(diffs[mt], 25)), 4),
                    "winner_minus_runnerup_p75": round(float(np.percentile(diffs[mt], 75)), 4),
                    "correct_wins_mean_margin": round(float(np.mean(
                        [s[mt] - min(v for k, v in s.items() if k != mt)
                         for s in score_records[mt]])), 4)}
    acc = {m: conf[m][m] / R for m in models}
    return {"R": R, "T": T, "confusion_counts": {m: rows[m]["counts"] for m in models},
            "confusion_row_proportion": {m: rows[m]["row_proportion"] for m in models},
            "score_diff_distribution": {m: {k: v for k, v in rows[m].items()
                                            if k not in ("counts", "row_proportion")} for m in models},
            "accuracy_by_true_model": {k: round(v, 3) for k, v in acc.items()},
            "overall_accuracy": round(sum(acc.values()) / len(models), 3),
            "chance": round(1 / len(models), 3)}


# ----------------------------------------------------------------------------
# F) Ablation 多设置
# ----------------------------------------------------------------------------
def run_ablation_multi(settings=None, R=30, seed=SEED):
    if settings is None:
        settings = [np.array([0.5, 2.0, 0.05, 3.0]),
                    np.array([0.0, 3.0, 0.10, 5.0]),
                    np.array([-1.0, 1.5, 0.02, 2.0])]
    kinds = ["full", "mean_wo_sprime", "precision_simple"]
    out = {}
    for si, theta in enumerate(settings):
        entry = {"generating_theta": list(theta), "R": R}
        pred = {k: [] for k in kinds}
        nllf = {k: [] for k in kinds}
        for k in kinds:
            for r in range(R):
                y = _sim_variant(theta, 80, seed + si * 1000 + r, "full")
                th, nllv = _fit_variant(y, k, n_starts=8, seed=seed + si * 1000 + 100 + r)
                nllf[k].append(nllv)
                pred[k].append(_rolling_logloss_clean(y,
                    lambda yy, ns, kk=k: _fit_variant(yy, kk, n_starts=ns, seed=0),
                    lambda yy, th2, kk=k: _forward_variant(yy, th2, kk)))
        rw_pred, rw_nll = [], []
        for r in range(R):
            y = _sim_variant(theta, 80, seed + si * 1000 + 500 + r, "full")
            a, n = fit_rw(y)
            rw_nll.append(n)
            rw_pred.append(_rolling_logloss_clean(y, lambda yy, ns: fit_rw(yy, seed=0),
                                                  lambda yy, a2: forward_rw(yy, a2)))
        entry["rolling_logloss"] = {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
                                    for k, v in pred.items()}
        entry["rolling_logloss"]["rw"] = (round(float(np.mean(rw_pred)), 4),
                                          round(float(np.std(rw_pred)), 4))
        entry["fitted_nll"] = {k: (round(float(np.mean(v)), 2), round(float(np.std(v)), 2))
                               for k, v in nllf.items()}
        entry["fitted_nll"]["rw"] = (round(float(np.mean(rw_nll)), 2),
                                     round(float(np.std(rw_nll)), 2))
        stab = {k: [] for k in kinds}
        for k in kinds:
            for r in range(R):
                y = _sim_variant(theta, 80, seed + si * 1000 + 900 + r, "full")
                th2, _ = _fit_variant(y, k, n_starts=10, seed=seed + si * 1000 + 1000 + r)
                stab[k].append(_transform(th2))
            entry[f"param_est_sd_log_{k}"] = np.round(np.array(stab[k]).std(axis=0), 3).tolist()
        out[f"setting_{si}"] = entry
    return out


# ----------------------------------------------------------------------------
# G) 边界敏感性
# ----------------------------------------------------------------------------
def run_bound_sensitivity(T=120, R=10, seed=SEED):
    theta = np.array([0.0, 3.0, 0.02, 5.0])
    variants = {
        "baseline": _BOUNDS,
        "sigma_upper_20": [(-6, 6), (-3, 3.0), (-4, 1.0), (-4, 4.0)],
        "omega_lower_0.002": [(-6, 6), (-3, 2.5), (-6, 1.0), (-4, 4.0)],
        "lambda_lower_0.002": [(-6, 6), (-3, 2.5), (-4, 1.0), (-6, 4.0)],
    }
    out = {}
    for name, bounds in variants.items():
        hits = np.zeros(4)
        hats = []
        nlls = []
        for r in range(R):
            y = simulate(theta, T, seed=seed + r)
            best_z, best_nll = None, np.inf
            rng = np.random.default_rng(seed + 100 + r)
            for i in range(8):
                z0 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 2),
                               rng.uniform(-2, 0), rng.uniform(-1, 2)])
                z0 = np.clip(z0, [b[0] for b in bounds], [b[1] for b in bounds])
                res = minimize(lambda z: nll(y, _untransform(z)), z0,
                               method="L-BFGS-B", bounds=bounds)
                if res.fun < best_nll:
                    best_nll, best_z = float(res.fun), res.x.copy()
            for j in range(4):
                if abs(best_z[j] - bounds[j][0]) < 1e-3 or abs(best_z[j] - bounds[j][1]) < 1e-3:
                    hits[j] += 1
            hats.append(_untransform(best_z))
            nlls.append(best_nll)
        out[name] = {"theta_hat_mean": list(np.round(np.mean(hats, axis=0), 3)),
                     "theta_hat_sd": list(np.round(np.std(hats, axis=0), 3)),
                     "nll_mean": round(float(np.mean(nlls)), 3),
                     "boundary_hit_fraction": (hits / R).tolist(),
                     "bounds_log": bounds}
    return {"generating_theta": list(theta), "R": R, "T": T, "variants": out}


# ----------------------------------------------------------------------------
# H) Unit tests + 前几试次数值示例
# ----------------------------------------------------------------------------
def _manual_replay(y, theta):
    """与 simulate/forward 相同递推的独立重放，返回每步 (v_hat, pi_hat, p, delta, v, pi)。"""
    mu, sigma, omega, lam = theta
    v, pi = float(mu), float(sigma)
    rows = []
    for t in range(len(y)):
        pi_hat = 1.0 / (1.0 / pi + omega)
        v_hat = v
        p = sigmoid(v_hat)
        sp = p * (1.0 - p)
        delta = y[t] - p
        rows.append({"t": t + 1, "v_hat": v_hat, "pi_hat": pi_hat, "p": p,
                     "delta": delta, "v_after": v_hat + (lam / pi_hat) * sp * delta,
                     "pi_after": pi_hat + lam * sp * sp})
        v = rows[-1]["v_after"]
        pi = rows[-1]["pi_after"]
    return rows


def run_unit_tests(seed=SEED):
    out = {}
    # H1: simulate 与 forward 一致（固定 y 的确定性重放）
    theta = np.array([0.5, 2.0, 0.05, 3.0])
    y = simulate(theta, 120, seed=seed)
    ps_forward = forward(y, theta)
    replay = _manual_replay(y, theta)
    max_diff = max(abs(ps_forward[t] - replay[t]["p"]) for t in range(len(y)))
    out["test_simulator_forward_consistency"] = {"max_abs_p_diff": round(float(max_diff), 12),
                                                 "pass": bool(max_diff < 1e-9)}
    # H2: nll 与 manual replay 一致
    nll_manual = -np.sum(y * np.log(np.clip([r["p"] for r in replay], 1e-12, 1 - 1e-12)) +
                         (1 - y) * np.log(np.clip(1 - np.array([r["p"] for r in replay]),
                                                  1e-12, 1 - 1e-12)))
    out["test_nll_manual"] = {"nll_func": round(float(nll(y, theta)), 9),
                              "nll_manual": round(float(nll_manual), 9),
                              "pass": bool(abs(nll(y, theta) - nll_manual) < 1e-6)}
    # H3: nll 有限差分梯度检查（z 空间）
    z0 = np.array([0.2, 0.0, -1.5, 0.8])
    eps = 1e-6
    grad_fd = np.zeros(4)
    for j in range(4):
        zp = z0.copy(); zp[j] += eps
        zm = z0.copy(); zm[j] -= eps
        grad_fd[j] = (nll(y, _untransform(zp)) - nll(y, _untransform(zm))) / (2 * eps)
    # 用 scipy 的数值梯度（approx_derivative 或手动多点验证）
    from scipy.optimize._numdiff import approx_derivative
    g = approx_derivative(lambda z: nll(y, _untransform(z)), z0, method="3-point")
    rel = np.abs(g - grad_fd) / (np.abs(g) + 1e-12)
    out["test_finite_difference_gradient"] = {"grad_fd": list(np.round(grad_fd, 6)),
                                              "grad_scipy": list(np.round(g, 6)),
                                              "max_rel_diff": round(float(rel.max()), 6),
                                              "pass": bool(rel.max() < 1e-3)}
    # H4: 概率边界安全性（极端序列不产生 inf/nan）
    ext = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
    nll_ext = nll(ext, np.array([4.0, 5.0, 0.001, 0.001]))
    out["test_extreme_probability_safety"] = {"nll_extreme_finite": bool(np.isfinite(nll_ext)),
                                              "nll_extreme_value": round(float(nll_ext), 3)}
    # H5: 首 3 个试次的完整数值示例
    y5 = simulate(theta, 5, seed=seed + 12345)
    replay5 = _manual_replay(y5, theta)
    out["worked_example_first_trials"] = {
        "theta": list(theta), "y": list(map(int, y5)),
        "note": ("v_1^- = mu (no decay applied to the mean); pi_1^- = 1/(1/sigma + omega) "
                 "i.e. the precision is decayed BEFORE the first prediction; "
                 "p_t = sigmoid(v_t^-); v_t = v_t^- + (lambda/pi_t^-)*s'(v_t^-)*delta_t; "
                 "pi_t = pi_t^- + lambda*s'(v_t^-)^2"),
        "trials": replay5}
    return out


def main():
    print("== A) Grid recovery R100 ==", flush=True)
    A = run_grid_recovery_r100()
    print("agg bias", A["signed_bias_log"], flush=True)
    print("agg mae", A["mae_log"], flush=True)
    print("agg coverage", A["coverage_within_tol"], flush=True)
    print("agg boundary/fit", A["boundary_hit_per_fit"], flush=True)
    print("== B) Multi-T recovery ==", flush=True)
    B = run_multi_T()
    for Tk, v in B["by_T"].items():
        print(f"  T={Tk} mae={v['mae_log']} coverage={v['coverage_within_tol']}", flush=True)
    print("== C) Profile abs change ==", flush=True)
    C = run_profile_abs()
    for nm in _NAMES:
        d = C[nm]
        print(f"  {nm}: deltaNLL(max-min)={d['delta_nll_abs_max_min']} "
              f"maxdev={d['max_dev_from_mle']}", flush=True)
    print("== D) Trade-off surfaces ==", flush=True)
    D = run_tradeoff()
    for k, v in D.items():
        print(f"  {k}: min-nll shape {len(v['min_nll_matrix'])}x{len(v['min_nll_matrix'][0])} "
              f"delta-range={np.round(np.array(v['delta_nll_matrix']).max(),3)}", flush=True)
    print("== E) Recovery R50 ==", flush=True)
    E = run_recovery_r50()
    print("confusion", json.dumps(E["confusion_counts"]), flush=True)
    print("row-prop", json.dumps(E["confusion_row_proportion"]), flush=True)
    print("overall", E["overall_accuracy"], "chance", E["chance"], flush=True)
    print("diff-dist", json.dumps(E["score_diff_distribution"]), flush=True)
    print("== F) Ablation multi-setting ==", flush=True)
    F = run_ablation_multi()
    for k, v in F.items():
        print(f"  {k}: ll={v['rolling_logloss']}", flush=True)
    print("== G) Bound sensitivity ==", flush=True)
    G = run_bound_sensitivity()
    for k, v in G["variants"].items():
        print(f"  {k}: hat={v['theta_hat_mean']} hits={v['boundary_hit_fraction']}", flush=True)
    print("== H) Unit tests ==", flush=True)
    H = run_unit_tests()
    for k, v in H.items():
        if isinstance(v, dict) and "pass" in v:
            print(f"  {k}: pass={v['pass']}", flush=True)
        elif k == "worked_example_first_trials":
            print(f"  worked example: y={v['y']} trials={len(v['trials'])}", flush=True)
    print("  first-trial details:", flush=True)
    for tr in H["worked_example_first_trials"]["trials"]:
        print(f"    t={tr['t']} v_hat={tr['v_hat']:.4f} pi_hat={tr['pi_hat']:.4f} "
              f"p={tr['p']:.4f} delta={tr['delta']:.4f} "
              f"v_after={tr['v_after']:.4f} pi_after={tr['pi_after']:.4f}", flush=True)

    results = {"seed": SEED, "grid_recovery_r100": A, "multi_T_recovery": B,
               "profile_abs_change": C, "tradeoff_surfaces": D,
               "recovery_r50": E, "ablation_multi": F,
               "bound_sensitivity": G, "unit_tests": H}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
