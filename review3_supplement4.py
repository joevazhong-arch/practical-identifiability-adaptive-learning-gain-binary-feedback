# -*- coding: utf-8 -*-
"""
review3_supplement4.py
=====================
OCT17_11（第七轮，与第六轮同审稿意见）补齐两个计算缺口：

  L) scatter_pairs —— MC2 要求的 true-versus-estimated 散点图原始数据：
       L1) 24 条件网格 × R100（T=80, 4 starts）：保存每条 replicate 的
           (transformed true, transformed estimate)，用于四参数散点+身份线；
       L2) 代表设置 θ*=(0.5,2,0.05,3) × T∈{40,80,120,240,480} × R100 同样保存。
  M) ablation_paired —— MC6 要求的同数据集配对差与不确定性：
       M1) 3 生成设置 × R30（T=80）：每个数据集只生成一次（full 模型），
           full / no-s' / precision-simple / RW 在【同一 y】上计算 rolling-origin
           log-loss，配对方差（full-minus-variant：mean/SD/95% CI/ full 胜出数）。
           （修正 review3_supplement.py 中 RW 使用 +500 偏移种子、在不同数据集上
             评估、无法严格配对的问题。）
       M2) 代表设置 × T∈{40,80,120} × R15，同口径配对。

输出 review3_diagnostics4.json。固定 seed=0，与既有诊断完全一致。
"""
import json
import os
import numpy as np

import adaptive_learning_gain
from adaptive_learning_gain import simulate, default_grid
from competitors import fit_rw, forward_rw
from review2_diagnostics import (_transform, _untransform, _BOUNDS, _NAMES,
                                 fit_diag, _sim_variant, _forward_variant,
                                 _fit_variant)
from review3_supplement import _rolling_logloss_clean
import paths

OUT = paths.output_path("review3_diagnostics4.json")
SEED = 0


# ----------------------------------------------------------------------------
# L) 逐 replicate 估计值（散点图）
# ----------------------------------------------------------------------------
def run_grid_scatter(T=80, n_starts=4, R=100, seed=SEED):
    grid = default_grid(n_mu=2, n_sigma=2, n_omega=2, n_lam=3, seed=seed)
    recs = []
    for g, th in enumerate(grid):
        z_true = _transform(th)
        for r in range(R):
            y = simulate(th, T, seed=seed + g * R + r)
            _, _, z_hat, _ = fit_diag(y, n_starts=n_starts, seed=seed + g * R + r)
            recs.append([g] + list(np.round(z_true, 4)) + list(np.round(z_hat, 4)))
    grid_raw = [list(np.round(th, 4)) for th in grid]
    return {"n_conditions": len(grid), "R": R, "T": T, "n_starts": n_starts,
            "param_names": _NAMES,
            "grid_theta_raw": grid_raw,
            "columns": ["condition", "mu_true", "sigma_true", "omega_true",
                        "lambda_true", "mu_hat", "sigma_hat", "omega_hat",
                        "lambda_hat"],
            "records": recs}


def run_multi_T_scatter(theta=None, Ts=(40, 80, 120, 240, 480), R=100,
                        n_starts=4, seed=SEED):
    if theta is None:
        theta = np.array([0.5, 2.0, 0.05, 3.0])
    z_true = _transform(theta)
    by_T = {}
    for T in Ts:
        recs = []
        for r in range(R):
            y = simulate(theta, T, seed=seed + T + r)
            _, _, z_hat, _ = fit_diag(y, n_starts=n_starts, seed=seed + T + r)
            recs.append(list(np.round(z_hat, 4)))
        by_T[str(T)] = {"z_true": list(np.round(z_true, 4)), "records": recs}
    return {"generating_theta": list(theta), "R": R, "n_starts": n_starts,
            "param_names": _NAMES, "by_T": by_T}


# ----------------------------------------------------------------------------
# M) Ablation 同数据集配对差
# ----------------------------------------------------------------------------
VARIANTS = ["full", "mean_wo_sprime", "precision_simple", "rw"]


def _scores_one_dataset(y, fit_starts=2):
    """同一 y 上四模型的 rolling-origin 平均 log-loss（无前视泄漏）。"""
    scores = {}
    for k in ["full", "mean_wo_sprime", "precision_simple"]:
        scores[k] = _rolling_logloss_clean(
            y,
            lambda yy, ns, kk=k: _fit_variant(yy, kk, n_starts=ns, seed=0),
            lambda yy, th2, kk=k: _forward_variant(yy, th2, kk),
            fit_starts=fit_starts)
    scores["rw"] = _rolling_logloss_clean(
        y, lambda yy, ns: fit_rw(yy, seed=0),
        lambda yy, a2: forward_rw(yy, a2), fit_starts=fit_starts)
    return scores


def _paired_summary(diff, n, label):
    m = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n)
    return {
        "comparison": label,
        "n_datasets": n,
        "mean_diff_full_minus_variant": round(m, 4),
        "sd_diff": round(sd, 4),
        "ci95_mean": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)],
        "datasets_full_better": int(np.sum(np.array(diff) < 0.0)),
        "datasets_variant_better": int(np.sum(np.array(diff) > 0.0)),
        "datasets_tied": int(np.sum(np.array(diff) == 0.0)),
        "note": ("negative mean diff / full_better means the full model has lower "
                 "rolling log-loss; CI = normal-approximation 95% CI for the mean "
                 "paired difference."),
    }


def run_ablation_paired(settings=None, R=30, T=80, seed=SEED):
    if settings is None:
        settings = [np.array([0.5, 2.0, 0.05, 3.0]),
                    np.array([0.0, 3.0, 0.10, 5.0]),
                    np.array([-1.0, 1.5, 0.02, 2.0])]
    out = {}
    for si, theta in enumerate(settings):
        per_ds = {k: [] for k in VARIANTS}
        for r in range(R):
            # 同一 y 只生成一次，四模型在同一数据集上评估（严格配对）
            y = _sim_variant(theta, T, seed + si * 1000 + r, "full")
            sc = _scores_one_dataset(y)
            for k in VARIANTS:
                per_ds[k].append(sc[k])
        comparisons = {}
        full = np.array(per_ds["full"])
        for k in ["mean_wo_sprime", "precision_simple", "rw"]:
            diff = full - np.array(per_ds[k])
            comparisons[f"full_vs_{k}"] = _paired_summary(
                diff, R, f"full minus {k} (setting {si}, T={T})")
        means = {k: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)]
                 for k, v in per_ds.items()}
        out[f"setting_{si}"] = {
            "generating_theta": list(theta), "T": T, "R": R,
            "rolling_logloss_mean_sd": means,
            "paired_differences": comparisons,
            "per_dataset_rolling_logloss": {k: [round(float(x), 5) for x in v]
                                            for k, v in per_ds.items()}}
    return out


def run_ablation_paired_multi_T(theta=None, Ts=(40, 80, 120), R=15, seed=SEED):
    if theta is None:
        theta = np.array([0.5, 2.0, 0.05, 3.0])
    by_T = {}
    for T in Ts:
        per_ds = {k: [] for k in VARIANTS}
        for r in range(R):
            y = _sim_variant(theta, T, seed + T * 100 + r, "full")
            sc = _scores_one_dataset(y)
            for k in VARIANTS:
                per_ds[k].append(sc[k])
        comparisons = {}
        full = np.array(per_ds["full"])
        for k in ["mean_wo_sprime", "precision_simple", "rw"]:
            diff = full - np.array(per_ds[k])
            comparisons[f"full_vs_{k}"] = _paired_summary(
                diff, R, f"full minus {k} (T={T})")
        means = {k: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)]
                 for k, v in per_ds.items()}
        by_T[str(T)] = {"rolling_logloss_mean_sd": means,
                        "paired_differences": comparisons,
                        "per_dataset_rolling_logloss": {
                            k: [round(float(x), 5) for x in v]
                            for k, v in per_ds.items()}}
    return {"generating_theta": list(theta), "R": R, "by_T": by_T}


def main():
    print("== L1) Grid scatter pairs R100 ==", flush=True)
    L1 = run_grid_scatter()
    print("  records:", len(L1["records"]), flush=True)
    print("== L2) Multi-T scatter pairs R100 ==", flush=True)
    L2 = run_multi_T_scatter()
    print("  Ts:", list(L2["by_T"].keys()), flush=True)
    print("== M1) Ablation paired (3 settings x R30, T=80) ==", flush=True)
    M1 = run_ablation_paired()
    for sk, v in M1.items():
        print(f"  {sk}:", flush=True)
        for ck, c in v["paired_differences"].items():
            print(f"    {ck}: mean {c['mean_diff_full_minus_variant']} "
                  f"CI {c['ci95_mean']} full_better {c['datasets_full_better']}/{c['n_datasets']}",
                  flush=True)
    print("== M2) Ablation paired multi-T (R15) ==", flush=True)
    M2 = run_ablation_paired_multi_T()
    for tk, v in M2["by_T"].items():
        print(f"  T={tk}:", flush=True)
        for ck, c in v["paired_differences"].items():
            print(f"    {ck}: mean {c['mean_diff_full_minus_variant']} "
                  f"CI {c['ci95_mean']} full_better {c['datasets_full_better']}/{c['n_datasets']}",
                  flush=True)
    results = {"seed": SEED,
               "scatter_grid_r100": L1,
               "scatter_multi_T_r100": L2,
               "ablation_paired_t80": M1,
               "ablation_paired_multi_T": M2}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
