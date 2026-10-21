# -*- coding: utf-8 -*-
"""
competitors.py
==============
Step 2（模型恢复）用到的竞争模型：
  1) Rescorla–Wagner（RW）学习器      —— 单一学习率 alpha，非层级、非贝叶斯
  2) 固定精度 Kalman 滤波器          —— 随机游走，但学习增益 lambda 固定为 1
最小自适应学习增益模型见 adaptive_learning_gain.py。

模型恢复即：用自适应滤波器与这两个竞争模型分别生成数据，再用三者各自拟合，
看交叉验证/信息准则能否识别出真实生成模型。
"""

import numpy as np
from scipy.optimize import minimize

from adaptive_learning_gain import sigmoid, nll


# ----------------------------------------------------------------------------
# 1) Rescorla–Wagner
# ----------------------------------------------------------------------------

def simulate_rw(alpha, T, seed=None):
    """RW：信念 q 直接更新 q <- q + alpha*(y - q)，预测概率 p = q。"""
    rng = np.random.default_rng(seed)
    q = 0.5
    y = np.zeros(T, dtype=int)
    for t in range(T):
        p = min(max(q, 1e-9), 1 - 1e-9)
        y[t] = rng.binomial(1, p)
        q = q + alpha * (y[t] - q)
    return y


def forward_rw(y, alpha):
    T = len(y)
    q = 0.5
    ps = np.empty(T)
    for t in range(T):
        ps[t] = min(max(q, 1e-9), 1 - 1e-9)
        q = q + alpha * (y[t] - q)
    return ps


def fit_rw(y, grid=None, seed=0):
    """RW 单参数：在 (0,1) 上网格 + 细优化。"""
    if grid is None:
        grid = np.linspace(0.02, 0.98, 49)
    best_a = grid[0]
    best_nll = np.inf
    for a in grid:
        n = -np.sum(y * np.log(np.clip(forward_rw(y, a), 1e-9, 1 - 1e-9)) +
                    (1 - y) * np.log(np.clip(1 - forward_rw(y, a), 1e-9, 1 - 1e-9)))
        if n < best_nll:
            best_nll = n
            best_a = a
    # 细优化
    res = minimize(lambda a: nll_rw(y, a), [best_a], method="L-BFGS-B",
                   bounds=[(0.001, 0.999)])
    return float(res.x[0]), float(res.fun)


def nll_rw(y, alpha):
    alpha = float(np.asarray(alpha).reshape(-1)[0])
    ps = np.clip(forward_rw(y, alpha), 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(ps) + (1 - y) * np.log(1 - ps))


# ----------------------------------------------------------------------------
# 2) 固定精度 Kalman 滤波器
# ----------------------------------------------------------------------------

def forward_kalman(y, theta):
    """与自适应滤波器相同递推，但 lambda 固定为 1；自由参数 (mu, omega)。"""
    mu, sigma, omega = theta
    T = len(y)
    ps = np.empty(T)
    v = float(mu)
    pi = float(sigma)
    for t in range(T):
        pi_hat = 1.0 / (1.0 / pi + omega)
        p = sigmoid(v)
        ps[t] = p
        sp = p * (1.0 - p)
        delta = y[t] - p
        v = v + (1.0 / pi_hat) * sp * delta
        pi = pi_hat + 1.0 * sp * sp
    return ps


def simulate_kalman(theta, T, seed=None):
    """固定精度 Kalman 生成模型；theta=(mu, sigma, omega)。"""
    rng = np.random.default_rng(seed)
    mu, sigma, omega = theta
    y = np.zeros(T, dtype=int)
    v = float(mu)
    pi = float(sigma)
    for t in range(T):
        pi_hat = 1.0 / (1.0 / pi + omega)
        p = sigmoid(v)
        y[t] = rng.binomial(1, p)
        sp = p * (1.0 - p)
        delta = y[t] - p
        v = v + (1.0 / pi_hat) * sp * delta
        pi = pi_hat + 1.0 * sp * sp
    return y


def _obj_kalman(z, y):
    mu, log_omega, log_sigma = z
    theta = (mu, np.exp(log_sigma), np.exp(log_omega))
    return _nll_kalman(y, theta)


def _nll_kalman(y, theta):
    ps = np.clip(forward_kalman(y, theta), 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(ps) + (1 - y) * np.log(1 - ps))


def fit_kalman(y, n_starts=20, seed=0):
    rng = np.random.default_rng(seed)
    best_z = None
    best_nll = np.inf
    bounds = [(-8, 8), (-8, 4), (-6, 6)]  # mu, log omega, log sigma
    for i in range(n_starts):
        z0 = np.array([rng.uniform(-3, 3), rng.uniform(-2, 1), rng.uniform(-1, 2)])
        res = minimize(_obj_kalman, z0, args=(y,), method="L-BFGS-B", bounds=bounds)
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_z = res.x
    # decision vector z = (mu, log_omega, log_sigma); return in the SAME
    # (mu, sigma, omega) order that forward_kalman/simulate_kalman expect.
    return (best_z[0], np.exp(best_z[2]), np.exp(best_z[1])), best_nll


def fit_kalman_pre_correction(y, n_starts=20, seed=0):
    """Pre-correction (historical-control) fixed-gain fit.

    Deliberately reproduces the PRE-V16 return-order defect in
    ``fit_kalman``: the optimisation/objective are identical and correct,
    but the returned tuple uses the swapped order
    ``(mu, exp(log_omega), exp(log_sigma)) = (mu, omega, sigma)``, which
    ``forward_kalman`` misreads as ``(mu, sigma, omega)``.

    It exists ONLY so that the archived pre-correction model-recovery
    results can be regenerated bit-for-bit as a historical control:
      * the three-observation rolling-origin pilot (Table S5;
        review3_supplement.py / review3_supplement7.py, 57/150 = 0.38),
      * the pre-correction block protocol (Table S8 control row and
        Text S6; review3_supplement7.py / review3_supplement9.py,
        58/150 = 0.387, permutation p = 0.079).
    The corrected primary results (67/150 and 72/150) are produced by
    ``fit_kalman`` together with review3_supplement10.py.
    Do NOT use this function for new analyses.
    """
    rng = np.random.default_rng(seed)
    best_z = None
    best_nll = np.inf
    bounds = [(-8, 8), (-8, 4), (-6, 6)]  # mu, log_omega, log_sigma
    for _i in range(n_starts):
        z0 = np.array([rng.uniform(-3, 3), rng.uniform(-2, 1),
                       rng.uniform(-1, 2)])
        res = minimize(_obj_kalman, z0, args=(y,), method="L-BFGS-B",
                       bounds=bounds)
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_z = res.x
    # intentionally swapped return order (pre-V16 historical control)
    return (best_z[0], np.exp(best_z[1]), np.exp(best_z[2])), best_nll


# ----------------------------------------------------------------------------
# 模型比较指标（论文 2.3 Step 2）
# ----------------------------------------------------------------------------

def bic(k, T, nll_min):
    """BIC = k*ln(T) + 2*nll。"""
    return k * np.log(T) + 2.0 * nll_min


def kfold_logloss(y, fit_fn, predict_fn, k=5, seed=0):
    """
    交叉验证负对数似然（k 折，按试次顺序切分）。
    每折：用其余试次拟合参数，评估本折试次的平均负对数似然。
    返回: 平均每折负对数似然（越小越好）。
    """
    rng = np.random.default_rng(seed)
    T = len(y)
    idx = np.arange(T)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    total = 0.0
    n = 0
    for fi in range(k):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fi])
        # 用训练子序列拟合（保持时序）
        y_train = y[train_idx]
        theta, _ = fit_fn(y_train)
        y_test = y[test_idx]
        # 对测试试次，用训练拟合的参数计算预测（单步滤波重放）
        ps = np.clip(predict_fn(y, theta), 1e-12, 1 - 1e-12)
        ps = ps[test_idx]
        total += -np.sum(y_test * np.log(ps) + (1 - y_test) * np.log(1 - ps))
        n += len(test_idx)
    return total / n


# 让本模块对自适应滤波器也暴露 fit / simulate，便于模型恢复统一调用
from adaptive_learning_gain import simulate as simulate_adaptive
from adaptive_learning_gain import fit as fit_adaptive
from adaptive_learning_gain import forward as forward_adaptive


MODELS = {
    "adaptive_filter": {
        "simulate": lambda theta, T, seed: simulate_adaptive(theta, T, seed),
        "fit": lambda y: fit_adaptive(y, n_starts=10, seed=0),
        "predict": forward_adaptive,
        "k": 4,
    },
    "rw": {
        "simulate": lambda theta, T, seed: simulate_rw(theta[0], T, seed),
        "fit": lambda y: fit_rw(y, seed=0),
        "predict": forward_rw,
        "k": 1,
    },
    "kalman_fixed": {
        "simulate": lambda theta, T, seed: simulate_kalman(theta, T, seed),
        "fit": lambda y: fit_kalman(y, n_starts=10, seed=0),
        "predict": forward_kalman,
        "k": 3,
    },
}
