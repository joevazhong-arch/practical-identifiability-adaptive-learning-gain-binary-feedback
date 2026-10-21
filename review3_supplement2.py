# -*- coding: utf-8 -*-
"""
review3_supplement2.py
=====================
补跑第五轮审稿要求的两个数据段：
  J) oos_diff_distribution —— MC7：多独立数据集上 out-of-sample log-loss 差分布
      模型 vs 经验频率基线（训练集常数频率）vs s(mu_hat) 基线；口径与 3.6 一致
      （T=120, train 72 / test 48, R=50, theta=(0.5,2,0.05,3)）。
  K) ablation_multi_T —— MC8：ablation 在多个 T 下重复（setting0, T in {40,80,120},
      R=15，四变体 rolling log-loss，无前视泄漏）。
产出 review3_diagnostics2.json
"""
import json
import os
import numpy as np
import adaptive_learning_gain
from adaptive_learning_gain import simulate, nll, forward, sigmoid, fit, _transform
from competitors import fit_rw, forward_rw, simulate_rw
from review2_diagnostics import (_transform as _tr, _untransform as _utr, _BOUNDS, _NAMES,
                                 _sim_variant, _forward_variant, _fit_variant)
from review3_supplement import _rolling_logloss_clean
import paths

OUT = paths.output_path("review3_diagnostics2.json")
SEED = 0


# ----------------------------------------------------------------------------
# J) OOS 多数据集 log-loss 差分布（MC7）
# ----------------------------------------------------------------------------
def run_oos_diff(R=50, T=120, train=72, seed=SEED):
    theta = np.array([0.5, 2.0, 0.05, 3.0])
    rows = []
    for r in range(R):
        y = simulate(theta, T, seed=seed + r)
        ytr, yte = y[:train], y[train:]
        zhat, nll_fit = fit(ytr, n_starts=8, seed=seed + r)
        p = forward(yte, zhat)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll_model = -np.mean(yte * np.log(p) + (1 - yte) * np.log(1 - p))
        # 基线1：s(mu_hat)（用训练集拟合的 mu，常数预测）
        p_mu = sigmoid(np.array([zhat[0]] * len(yte)))
        p_mu = np.clip(p_mu, 1e-12, 1 - 1e-12)
        ll_mu = -np.mean(yte * np.log(p_mu) + (1 - yte) * np.log(1 - p_mu))
        # 基线2：训练集经验频率（常数预测）
        q = ytr.mean()
        q = min(max(q, 1e-6), 1 - 1e-6)
        ll_freq = -np.mean(yte * np.log(q) + (1 - yte) * np.log(1 - q))
        rows.append({"ll_model": float(ll_model), "ll_s_mu_hat": float(ll_mu),
                     "ll_empirical_freq": float(ll_freq)})
    llm = np.array([x["ll_model"] for x in rows])
    llmu = np.array([x["ll_s_mu_hat"] for x in rows])
    llf = np.array([x["ll_empirical_freq"] for x in rows])
    out = {
        "generating_theta": list(theta), "T": T, "train": train, "R": R,
        "model_logloss": {"mean": round(float(llm.mean()), 4), "sd": round(float(llm.std()), 4)},
        "s_mu_hat_logloss": {"mean": round(float(llmu.mean()), 4), "sd": round(float(llmu.std()), 4)},
        "empirical_freq_logloss": {"mean": round(float(llf.mean()), 4), "sd": round(float(llf.std()), 4)},
        "diff_model_vs_empirical_freq": {
            "mean": round(float((llm - llf).mean()), 4), "sd": round(float((llm - llf).std()), 4),
            "p25": round(float(np.percentile(llm - llf, 25)), 4),
            "p75": round(float(np.percentile(llm - llf, 75)), 4),
            "fraction_model_worse": round(float(np.mean(llm > llf)), 4)},
        "diff_model_vs_s_mu_hat": {
            "mean": round(float((llm - llmu).mean()), 4), "sd": round(float((llm - llmu).std()), 4),
            "p25": round(float(np.percentile(llm - llmu, 25)), 4),
            "p75": round(float(np.percentile(llm - llmu, 75)), 4),
            "fraction_model_worse": round(float(np.mean(llm > llmu)), 4)},
        "note": ("empirical-frequency baseline uses the training-set frequency as a CONSTANT "
                 "predictor on held-out trials (no updating); s(mu_hat) uses the fitted prior-mean "
                 "probability as a constant predictor; both fitted without held-out outcomes."),
    }
    return out


# ----------------------------------------------------------------------------
# K) Ablation 多 T（MC8）
# ----------------------------------------------------------------------------
def run_ablation_multi_T(theta=None, Ts=(40, 80, 120), R=15, seed=SEED):
    if theta is None:
        theta = np.array([0.5, 2.0, 0.05, 3.0])
    kinds = ["full", "mean_wo_sprime", "precision_simple"]
    out = {}
    for T in Ts:
        entry = {}
        for k in kinds:
            vals = []
            for r in range(R):
                y = _sim_variant(theta, T, seed + T * 100 + r, "full")
                vals.append(_rolling_logloss_clean(
                    y,
                    lambda yy, ns, kk=k: _fit_variant(yy, kk, n_starts=ns, seed=0),
                    lambda yy, th2, kk=k: _forward_variant(yy, th2, kk)))
            entry[k] = (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))
        rw = []
        for r in range(R):
            y = _sim_variant(theta, T, seed + T * 100 + 500 + r, "full")
            rw.append(_rolling_logloss_clean(y, lambda yy, ns: fit_rw(yy, seed=0),
                                             lambda yy, a2: forward_rw(yy, a2)))
        entry["rw"] = (round(float(np.mean(rw)), 4), round(float(np.std(rw)), 4))
        out[str(T)] = entry
    return {"generating_theta": list(theta), "R": R, "by_T": out,
            "note": ("ablation variants fitted on datasets generated by the FULL model "
                     "(self-generated data); rolling-origin without look-ahead; "
                     "metric = mean per-trial log-loss (mean, SD across datasets).")}


def main():
    print("== J) OOS diff distribution ==", flush=True)
    J = run_oos_diff()
    print("model", J["model_logloss"], "s_mu", J["s_mu_hat_logloss"], "freq", J["empirical_freq_logloss"],
          "diff_vs_freq", J["diff_model_vs_empirical_freq"], flush=True)
    print("== K) Ablation multi-T ==", flush=True)
    K = run_ablation_multi_T()
    for Tk, v in K["by_T"].items():
        print(f"  T={Tk}: {v}", flush=True)
    results = {"oos_diff_distribution": J, "ablation_multi_T": K}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
