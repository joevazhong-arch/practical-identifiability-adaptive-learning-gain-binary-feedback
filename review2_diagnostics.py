# -*- coding: utf-8 -*-
"""
review2_diagnostics.py
=======================
PLOS Comp Biol review2 要求的、此前标为"planned"的分析的可执行版本。
只读 adaptive_learning_gain 库，不改动原管线；全部随机性由固定 seed 控制。

产出（写入 review2_diagnostics.json）：
  A) ablation          —— 提出模型 vs 标准 RW vs 无 s'(v) 均值因子版 vs
                          更简单精度更新版：序列级预测 log-loss + 拟合 nll
                          比较，并报告各变体的参数估计稳定性。
  B) grid_recovery     —— 参数网格（低/中/高）多条件联合恢复：signed bias、
                          MAE、RMSE、容差覆盖率、边界命中率；正参 raw 与 log 尺度。
  C) profile_likelihood—— 四参数 profile likelihood（每点重优化其余参数），
                          判断 95% profile CI 宽 / ridge / plateau / 到边界。
  D) optimization_diag —— 多起点 + 替代优化器(SLSQP/Powell)：边界命中率、
                          目标值分布、是否收敛到一致 MLE。
  E) rolling_recovery  —— rolling-origin（仅用此前观测预测未来）模型恢复混淆矩阵。
  F) predictive_dist   —— 完整预测分布：run 长度、序列依赖、时变校准、单步
                          log-loss；模型 vs s(mu_hat) 基线 vs 经验频率基线。
"""
import json
import os
import numpy as np
from scipy.optimize import minimize

from adaptive_learning_gain import simulate, nll, forward, sigmoid, default_grid, fit
from competitors import simulate_rw, forward_rw, nll_rw, fit_rw, simulate_kalman, fit_kalman, forward_kalman

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "review2_diagnostics.json")

# ----------------------------------------------------------------------------
# 拟合包装：返回 untransformed theta、log 空间 z、nll、边界命中与优化诊断
# ----------------------------------------------------------------------------

def _transform(theta):
    mu, sigma, omega, lam = theta
    return np.array([mu, np.log(sigma), np.log(omega), np.log(lam)])

def _untransform(z):
    return np.array([z[0], np.exp(z[1]), np.exp(z[2]), np.exp(z[3])])

_BOUNDS = [(-6, 6), (-3, 2.5), (-4, 1.0), (-4, 4.0)]
_NAMES = ["mu", "sigma", "omega", "lambda"]

def fit_diag(y, n_starts=10, seed=0, method="L-BFGS-B"):
    """多起点拟合，返回 (theta_hat, nll, logz, {diag})。"""
    rng = np.random.default_rng(seed)
    best_z, best_nll, best_res = None, np.inf, None
    boundary_hits = np.zeros(4, dtype=int)
    all_nll = []
    for i in range(n_starts):
        z0 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 2),
                       rng.uniform(-2, 0), rng.uniform(-1, 2)])
        z0 = np.clip(z0, [b[0] for b in _BOUNDS], [b[1] for b in _BOUNDS])
        res = minimize(lambda z: nll(y, _untransform(z)), z0,
                       method=method, bounds=_BOUNDS)
        all_nll.append(float(res.fun))
        z = res.x
        for j in range(4):
            if abs(z[j] - _BOUNDS[j][0]) < 1e-3 or abs(z[j] - _BOUNDS[j][1]) < 1e-3:
                boundary_hits[j] += 1
        if res.fun < best_nll:
            best_nll, best_z, best_res = float(res.fun), z.copy(), res
    return (_untransform(best_z), best_nll, best_z,
            {"boundary_hit_rate": (boundary_hits / n_starts).tolist(),
             "nll_min": min(all_nll), "nll_max": max(all_nll),
             "nll_std": float(np.std(all_nll)),
             "converged": int(best_res.success), "iterations": int(best_res.nit)})


# ----------------------------------------------------------------------------
# A) Ablation：提出模型与三个变体 + 标准 RW 的序列级预测与参数稳定性
# ----------------------------------------------------------------------------

# 三个"ablation 变体"都保留 (mu,sigma,omega,lambda) 四参，仅改更新式：
#  - mean_wo_sprime : 均值更新去掉 s'(v) 因子  v += (lam/pi)*delta；精度更新保留 s'^2
#  - precision_simple: 精度更新改为常数增量  pi += lam；均值更新保留 s'(v)/pi
#  - full            : 提出模型（现有）
def _sim_variant(theta, T, seed, kind):
    mu, sigma, omega, lam = theta
    rng = np.random.default_rng(seed)
    y = np.zeros(T, dtype=int)
    v, pi = float(mu), float(sigma)
    for t in range(T):
        pi_hat = 1.0 / (1.0 / pi + omega)
        p = sigmoid(v)
        y[t] = rng.binomial(1, p)
        sp = p * (1.0 - p)
        delta = y[t] - p
        if kind == "mean_wo_sprime":
            v = v + (lam / pi_hat) * delta
            pi = pi_hat + lam * sp * sp
        elif kind == "precision_simple":
            v = v + (lam / pi_hat) * sp * delta
            pi = pi_hat + lam
        else:  # full
            v = v + (lam / pi_hat) * sp * delta
            pi = pi_hat + lam * sp * sp
    return y

def _forward_variant(y, theta, kind):
    mu, sigma, omega, lam = theta
    T = len(y)
    ps = np.empty(T)
    v, pi = float(mu), float(sigma)
    for t in range(T):
        pi_hat = 1.0 / (1.0 / pi + omega)
        p = sigmoid(v)
        ps[t] = p
        sp = p * (1.0 - p)
        delta = y[t] - p
        if kind == "mean_wo_sprime":
            v = v + (lam / pi_hat) * delta
            pi = pi_hat + lam * sp * sp
        elif kind == "precision_simple":
            v = v + (lam / pi_hat) * sp * delta
            pi = pi_hat + lam
        else:
            v = v + (lam / pi_hat) * sp * delta
            pi = pi_hat + lam * sp * sp
    return ps

def _nll_variant(y, theta, kind):
    ps = np.clip(_forward_variant(y, theta, kind), 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(ps) + (1 - y) * np.log(1 - ps))

def _fit_variant(y, kind, n_starts=8, seed=0):
    rng = np.random.default_rng(seed)
    best_z, best_nll = None, np.inf
    for i in range(n_starts):
        z0 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 2),
                       rng.uniform(-2, 0), rng.uniform(-1, 2)])
        z0 = np.clip(z0, [b[0] for b in _BOUNDS], [b[1] for b in _BOUNDS])
        res = minimize(lambda z: _nll_variant(y, _untransform(z), kind),
                       z0, method="L-BFGS-B", bounds=_BOUNDS)
        if res.fun < best_nll:
            best_nll, best_z = float(res.fun), res.x.copy()
    return _untransform(best_z), best_nll

def _rolling_logloss(y, fit_fn, predict_fn, cuts=None, fit_starts=4):
    """rolling-origin：训练前缀 cut，用此前观测拟合，预测下一条；累计负对数似然/试次。
    默认用 4 个代表性切点以控制计算成本（仍是"仅用此前观测预测未来"的时序评估）。"""
    T = len(y)
    if cuts is None:
        cuts = [int(T * 0.25), int(T * 0.40), int(T * 0.60), int(T * 0.80)]
    ll = 0.0
    n = 0
    for cut in cuts:
        if cut < 20:
            continue
        th, _ = fit_fn(y[:cut], fit_starts)
        p = predict_fn(y[:cut + 1], th)[-1]
        p = min(max(p, 1e-12), 1 - 1e-12)
        ll -= (y[cut] * np.log(p) + (1 - y[cut]) * np.log(1 - p))
        n += 1
    return ll / max(n, 1)

def run_ablation(T=80, R=12, seed=0):
    theta = np.array([0.5, 2.0, 0.05, 3.0])
    kinds = ["full", "mean_wo_sprime", "precision_simple"]
    out = {"generating_theta": list(theta), "T": T, "R": R,
           "note": "fixed-theta comparison: all models fitted to datasets generated from the full model"}
    # 序列级预测：rolling-origin log-loss 与整段拟合 nll
    pred = {k: [] for k in kinds}
    nllf = {k: [] for k in kinds}
    for k in kinds:
        for r in range(R):
            y = _sim_variant(theta, T, seed + r, "full")   # 生成自 full 模型
            th, nllv = _fit_variant(y, k, n_starts=8, seed=seed + 100 + r)
            nllf[k].append(nllv)
            pred[k].append(_rolling_logloss(y,
                lambda yy, ns, kk=k: _fit_variant(yy, kk, n_starts=ns, seed=0),
                lambda yy, th, kk=k: _forward_variant(yy, th, kk)))
    # 标准 RW：同一批 full 生成数据上拟合（公平对比；RW 无 s' 结构，作基线）
    rw_pred, rw_nll = [], []
    for r in range(R):
        y = _sim_variant(theta, T, seed + 10 * r, "full")
        a, n = fit_rw(y)
        rw_nll.append(n)
        rw_pred.append(_rolling_logloss(y,
            lambda yy, ns: fit_rw(yy, seed=0),
            lambda yy, a: forward_rw(yy, a)))
    out["rolling_logloss"] = {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
                              for k, v in pred.items()}
    out["rolling_logloss"]["rw"] = (round(float(np.mean(rw_pred)), 4),
                                    round(float(np.std(rw_pred)), 4))
    out["fitted_nll"] = {k: (round(float(np.mean(v)), 2), round(float(np.std(v)), 2))
                         for k, v in nllf.items()}
    out["fitted_nll"]["rw"] = (round(float(np.mean(rw_nll)), 2),
                               round(float(np.std(rw_nll)), 2))
    # 参数估计稳定性（多起点方差）：full 与两变体各拟合 N 个数据集
    stab = {k: [] for k in kinds}
    for k in kinds:
        for r in range(R):
            y = _sim_variant(theta, T, seed + 500 + r, "full")
            th, _ = _fit_variant(y, k, n_starts=10, seed=seed + 600 + r)
            stab[k].append(_transform(th))   # log 空间
        stab[k] = np.array(stab[k])
        out[f"param_est_sd_log_{k}"] = np.round(stab[k].std(axis=0), 3).tolist()
    return out


# ----------------------------------------------------------------------------
# B) 参数网格联合恢复：多条件 signed bias / MAE / RMSE / 覆盖率 / 边界命中率
# ----------------------------------------------------------------------------

def run_grid_recovery(T=80, n_starts=8, R=4, tol=0.6, seed=0):
    grid = default_grid(n_mu=2, n_sigma=2, n_omega=2, n_lam=3, seed=seed)  # 24 条件
    G = len(grid)
    bias_log = np.zeros(4)      # mu raw 差；正参 log 差
    bias_raw = np.zeros(4)
    mae = np.zeros(4)
    mse = np.zeros(4)
    within = np.zeros(4)
    bhit = np.zeros(4)
    cnt = 0
    for g in range(G):
        th = grid[g]
        for r in range(R):
            y = simulate(th, T, seed=seed + g * R + r)
            hat, _, z, diag = fit_diag(y, n_starts=n_starts, seed=seed + g * R + r)
            err_log = np.abs(_transform(th) - z)     # 全部在 log/raw 混合空间
            err_raw = np.abs(th - hat)
            bias_log += (z - _transform(th))
            bias_raw += (hat - th)
            mae += err_log
            mse += err_log ** 2
            within += (err_log < tol).astype(float)
            bhit += np.array(diag["boundary_hit_rate"]) * n_starts
            cnt += 1
    out = {"n_conditions": G, "R": R, "T": T, "n_starts": n_starts,
           "tol_log": tol, "param_names": _NAMES}
    out["signed_bias_log"] = np.round(bias_log / cnt, 3).tolist()
    out["signed_bias_raw"] = np.round(bias_raw / cnt, 3).tolist()
    out["mae_log"] = np.round(mae / cnt, 3).tolist()
    out["rmse_log"] = np.round(np.sqrt(mse / cnt), 3).tolist()
    out["coverage_within_tol"] = np.round(within / cnt, 3).tolist()
    out["boundary_hit_count_per_fit"] = np.round(bhit / cnt, 3).tolist()
    # mu 的 log-odds 误差 → 接受概率平移示例
    out["mu_prob_translation"] = {"mu_mae_log_odds": float(mae[0] / cnt),
                                  "s_0.62_reference": "s(0.5)~0.62"}
    return out


# ----------------------------------------------------------------------------
# C) Profile likelihood（四参数，每点重优化其余）
# ----------------------------------------------------------------------------

def _profile(y, n_starts=4, seed=0):
    """对每参数在其变换空间取网格，固定它、重优化其余，返回 profile 表。"""
    # 先求 MLE
    th_mle, nll_mle, z_mle, _ = fit_diag(y, n_starts=15, seed=seed)
    profiles = {}
    for j in range(4):
        if j == 0:   # mu raw
            lo, hi = -4.0, 4.0
            pts = np.linspace(lo, hi, 13)
        else:        # log(positive)
            lo, hi = _BOUNDS[j]
            pts = np.linspace(lo, hi, 13)
        rows = []
        for val in pts:
            best_nll = np.inf
            best_z = None
            rng = np.random.default_rng(seed + j * 100 + int((val - lo) / (hi - lo) * 100))
            for i in range(n_starts):
                z0 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 2),
                               rng.uniform(-2, 0), rng.uniform(-1, 2)])
                z0 = np.clip(z0, [b[0] for b in _BOUNDS], [b[1] for b in _BOUNDS])
                z0[j] = val
                # 将第 j 维固定（bounds 设为 [val,val]），重优化其余参数
                bb = [list(b) for b in _BOUNDS]
                bb[j] = [val, val]
                res = minimize(lambda z: nll(y, _untransform(z)), z0, method="L-BFGS-B",
                               bounds=bb)
                if res.fun < best_nll:
                    best_nll, best_z = float(res.fun), res.x.copy()
            x_native = float(np.exp(val)) if j > 0 else float(val)
            rows.append({"x": float(val), "x_native": x_native,
                         "profile_nll": round(best_nll, 3),
                         "profile_nll_minus_mle": round(best_nll - nll_mle, 3)})
        profiles[_NAMES[j]] = rows
        # 计算 95% profile CI 宽（nll < nll_mle + 3.84 的范围），并判定形态
        good = [r for r in rows if r["profile_nll_minus_mle"] < 3.84]
        span = (max(r["x"] for r in good) - min(r["x"] for r in good)) if good else float("nan")
        full_span = rows[-1]["x"] - rows[0]["x"]
        span_ratio = span / full_span if full_span > 0 else float("nan")
        min_dev = min(r["profile_nll_minus_mle"] for r in rows)
        profiles[f"{_NAMES[j]}_summary"] = {
            "profile_95_width_transformed": round(float(span), 3),
            "span_ratio_of_full_range": round(float(span_ratio), 3),
            "min_relative_nll": round(float(min_dev), 3),
            "note": ("wide profile CI (ratio>0.9) => weak practical identifiability; "
                     "flat profile => broad plateau; min_relative_nll large => misspecified/far from MLE"),
        }
    return {"mle_theta": list(np.round(th_mle, 3)), "mle_nll": round(nll_mle, 3),
            "profiles": profiles}


# ----------------------------------------------------------------------------
# D) 优化诊断：多起点 + 替代优化器
# ----------------------------------------------------------------------------

def run_opt_diag(T=120, seed=0):
    y = simulate((0.0, 3.0, 0.02, 5.0), T, seed=seed)   # high_learning_gain 场景
    res_lbfgs = fit_diag(y, n_starts=30, seed=seed, method="L-BFGS-B")
    # 替代优化器（log 空间，同边界）
    def run_method(method, n_starts=12):
        rng = np.random.default_rng(seed + 900)
        best, bnll, hits = None, np.inf, np.zeros(4)
        for i in range(n_starts):
            z0 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 2),
                           rng.uniform(-2, 0), rng.uniform(-1, 2)])
            z0 = np.clip(z0, [b[0] for b in _BOUNDS], [b[1] for b in _BOUNDS])
            r = minimize(lambda z: nll(y, _untransform(z)), z0, method=method,
                         bounds=_BOUNDS)
            for j in range(4):
                if abs(r.x[j] - _BOUNDS[j][0]) < 1e-3 or abs(r.x[j] - _BOUNDS[j][1]) < 1e-3:
                    hits[j] += 1
            if r.fun < bnll:
                bnll, best = float(r.fun), r.x.copy()
        return _untransform(best), bnll, (hits / n_starts).tolist()
    slsqp = run_method("SLSQP")
    powell = run_method("Powell")
    return {
        "L-BFGS-B": {"theta_hat": list(np.round(res_lbfgs[0], 3)),
                     "nll": round(res_lbfgs[1], 3),
                     "boundary_hit_rate": res_lbfgs[3]["boundary_hit_rate"],
                     "nll_min": res_lbfgs[3]["nll_min"],
                     "nll_max": res_lbfgs[3]["nll_max"],
                     "nll_std": res_lbfgs[3]["nll_std"],
                     "converged": res_lbfgs[3]["converged"]},
        "SLSQP": {"theta_hat": list(np.round(slsqp[0], 3)), "nll": round(slsqp[1], 3),
                  "boundary_hit_rate": slsqp[2]},
        "Powell": {"theta_hat": list(np.round(powell[0], 3)), "nll": round(powell[1], 3),
                   "boundary_hit_rate": powell[2]},
    }


# ----------------------------------------------------------------------------
# E) rolling-origin 模型恢复（仅用此前观测预测未来）
# ----------------------------------------------------------------------------

def run_rolling_recovery(T=80, R=12, seed=0):
    models = ["adaptive_filter", "rw", "kalman_fixed"]
    reps = {"adaptive_filter": (0.5, 2.0, 0.05, 3.0), "rw": (0.30,),
            "kalman_fixed": (0.5, 2.0, 0.05)}
    conf = {m: {m2: 0 for m2 in models} for m in models}
    def fit_adaptive(y, ns): return fit(y, n_starts=ns, seed=0)
    def predict_adaptive(yy, th): return forward(yy, th)
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
                    scores[m2] = _rolling_logloss(y, fit_adaptive, predict_adaptive)
                elif m2 == "rw":
                    scores[m2] = _rolling_logloss(y, lambda yy, ns: fit_rw(yy, seed=0),
                                                  forward_rw)
                else:
                    scores[m2] = _rolling_logloss(y, fit_kalman, forward_kalman)
            conf[mt][min(scores, key=scores.get)] += 1
    acc = {m: conf[m][m] / R for m in models}
    return {"confusion_rolling_origin": conf, "R": R, "T": T,
            "accuracy_by_true_model": {k: round(v, 3) for k, v in acc.items()},
            "overall_accuracy": round(sum(acc.values()) / len(models), 3)}


# ----------------------------------------------------------------------------
# F) 预测分布：run 长度 / 序列依赖 / 时变校准 / 单步 log-loss + 双基线
# ----------------------------------------------------------------------------

def _run_stats(y):
    d = np.diff(y)
    runs = np.split(np.cumsum(np.abs(d)), np.where(d != 0)[0] + 1)
    lens = [len(r) for r in runs]
    # 正/负 run 的最大长度
    pos = y == 1
    return {"mean_run_len": float(np.mean(lens)) if lens else float("nan"),
            "max_run_len": float(max(lens)) if lens else 1.0,
            "lag1_autocorr": float(np.corrcoef(y[:-1], y[1:])[0, 1]) if len(y) > 2 else 0.0,
            "switches": float(np.sum(np.diff(y) != 0))}

def run_predictive_dist(theta=(0.5, 2.0, 0.05, 3.0), T=120, n_sim=120, seed=0):
    y = simulate(theta, T, seed=seed)
    th_hat, nll_mle, _, _ = fit_diag(y, n_starts=12, seed=seed)
    obs = _run_stats(y)
    sims = {k: [] for k in obs}
    rng = np.random.default_rng(seed + 5000)
    for i in range(n_sim):
        ys = simulate(th_hat, T, seed=seed + 6000 + i)
        st = _run_stats(ys)
        for k in obs:
            sims[k].append(st[k])
    ppp = {k: float((np.array(sims[k]) >= obs[k]).mean()) for k in obs}
    # 时变校准：前后半段平均预测概率 vs 观测接受率
    ps = forward(y, th_hat)
    h = T // 2
    cal = {
        "first_half_pred_mean": round(float(ps[:h].mean()), 3),
        "first_half_obs_rate": round(float(y[:h].mean()), 3),
        "second_half_pred_mean": round(float(ps[h:].mean()), 3),
        "second_half_obs_rate": round(float(y[h:].mean()), 3),
    }
    # 单步 log-loss：模型 vs s(mu_hat) 基线 vs 经验频率基线（用训练前缀）
    def stepwise_ll(predictor):
        tot = 0.0
        for cut in range(30, T):
            p = predictor(y[:cut], cut)
            p = min(max(p, 1e-12), 1 - 1e-12)
            tot -= (y[cut] * np.log(p) + (1 - y[cut]) * np.log(1 - p))
        return tot / (T - 30)
    # 模型：用 y[:cut] 拟合，预测下一条
    model_ll = stepwise_ll(lambda yy, cut: forward(yy, fit(yy, n_starts=6, seed=0)[0])[-1])
    base_mu = stepwise_ll(lambda yy, cut: sigmoid(np.array([fit(yy, n_starts=6, seed=0)[0][0]]))[0])
    base_emp = stepwise_ll(lambda yy, cut: float(np.mean(yy)))
    return {"observed": obs, "ppp": ppp, "n_sim": n_sim,
            "time_varying_calibration": cal,
            "one_step_logloss": {
                "model": round(model_ll, 4),
                "baseline_s_mu_hat": round(base_mu, 4),
                "baseline_empirical_freq": round(base_emp, 4)},
            "mle_theta": list(np.round(th_hat, 3))}


def main():
    print("== A) Ablation ==", flush=True)
    ablation = run_ablation()
    print(json.dumps({k: v for k, v in ablation.items()
                      if k not in ("generating_theta",)},
                     ensure_ascii=False, indent=1), flush=True)
    print("== B) Grid recovery ==", flush=True)
    grid = run_grid_recovery()
    print("signed_bias_log", grid["signed_bias_log"], flush=True)
    print("signed_bias_raw", grid["signed_bias_raw"], flush=True)
    print("mae_log", grid["mae_log"], flush=True)
    print("rmse_log", grid["rmse_log"], flush=True)
    print("coverage", grid["coverage_within_tol"], flush=True)
    print("boundary_hit/fit", grid["boundary_hit_count_per_fit"], flush=True)
    print("== C) Profile likelihood ==", flush=True)
    yp = simulate((0.0, 3.0, 0.02, 5.0), 120, seed=0)
    prof = _profile(yp, n_starts=4, seed=0)
    for nm in _NAMES:
        s = prof["profiles"].get(f"{nm}_summary")
        if s:
            print(f"  {nm}: 95%width={s['profile_95_width_transformed']} "
                  f"ratio={s['span_ratio_of_full_range']} minrel={s['min_relative_nll']}",
                  flush=True)
    print("== D) Optimization diagnostics ==", flush=True)
    opt = run_opt_diag()
    for m, d in opt.items():
        print(f"  {m}: nll={d['nll']} boundary={d['boundary_hit_rate']} "
              f"hat={d['theta_hat']}", flush=True)
    print("== E) Rolling-origin model recovery ==", flush=True)
    roll = run_rolling_recovery()
    print("confusion", json.dumps(roll["confusion_rolling_origin"]), flush=True)
    print("accuracy", roll["accuracy_by_true_model"], "overall", roll["overall_accuracy"], flush=True)
    print("== F) Predictive distribution ==", flush=True)
    pred = run_predictive_dist()
    print("ppp", json.dumps(pred["ppp"]), flush=True)
    print("calibration", json.dumps(pred["time_varying_calibration"]), flush=True)
    print("one_step_logloss", json.dumps(pred["one_step_logloss"]), flush=True)

    results = {"ablation": ablation, "grid_recovery": grid, "profile_likelihood": prof,
               "optimization_diag": opt, "rolling_recovery": roll,
               "predictive_dist": pred, "seed": 0}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
