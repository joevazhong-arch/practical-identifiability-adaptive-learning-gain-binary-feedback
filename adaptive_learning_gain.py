# -*- coding: utf-8 -*-
"""
adaptive_learning_gain.py
=========================
最小自适应学习增益模型（社会反馈） —— 与预印本论文对应。

模型（论文 2.1 节）：
  - 观测:  y_t in {0,1} (接受=1 / 拒绝=0)
  - 生成:  p(y_t=1 | v_t) = s(v_t),  s(x) = 1/(1+exp(-x))
  - 潜信念: v_t | v_{t-1} ~ N(v_{t-1}, omega)
            初始信念确定性设定 v_1 = mu，初始置信精度 pi_1 = sigma
            （Model B；不从高斯先验采样）
  - 自适应学习增益更新（精度形式）:
      预测   : pi_t^- = 1/(1/pi_{t-1} + omega),  p_t = s(v_t^-)
      后验   : delta_t = y_t - p_t,
               v_t = v_t^- + (lambda/pi_t^-)*s'(v_t^-)*delta_t
               pi_t = pi_t^- + lambda*s'(v_t^-)^2
  - 参数  theta = (mu, sigma, omega, lambda):
      mu    先验均值(基线信念)
      sigma 置信精度(初始内部置信度; 非高斯先验方差的倒数)
      omega 漂移方差(时间波动性/volatility)
      lambda学习增益(learning gain / feedback gain; 每条反馈更新信念的强度)

依赖: numpy, scipy。所有随机性由固定 seed 控制（可复现）。
"""

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------

def sigmoid(x):
    """逻辑斯蒂函数 s(x) = 1/(1+exp(-x))，数值稳定。"""
    out = np.empty_like(np.asarray(x, dtype=float))
    a = np.asarray(x, dtype=float)
    out[a >= 0] = 1.0 / (1.0 + np.exp(-a[a >= 0]))
    out[a < 0] = np.exp(a[a < 0]) / (1.0 + np.exp(a[a < 0]))
    return out


# ----------------------------------------------------------------------------
# 生成模型（论文伪代码 simulate）
# ----------------------------------------------------------------------------

def simulate(theta, T, seed=None):
    """
    按生成模型模拟 T 次伯努利观测序列。
    theta = (mu, sigma, omega, lambda)。
    返回: 长度为 T 的 0/1 数组。
    """
    mu, sigma, omega, lam = theta
    rng = np.random.default_rng(seed)
    y = np.zeros(T, dtype=int)
    v = float(mu)
    pi = float(sigma)          # 初始先验精度
    for t in range(T):
        pi_hat = 1.0 / (1.0 / pi + omega)   # 预测精度
        v_hat = v                            # 预测均值
        p = sigmoid(v_hat)
        y[t] = rng.binomial(1, p)
        sp = p * (1.0 - p)                   # s'(v)
        delta = y[t] - p
        v = v_hat + (lam / pi_hat) * sp * delta
        pi = pi_hat + lam * sp * sp
    return y


# ----------------------------------------------------------------------------
# 滤波 / 似然
# ----------------------------------------------------------------------------

def forward(y, theta):
    """
    正向滤波：给定观测序列 y 和参数 theta，返回每步预测概率 p_t。
    与 simulate 中的递推完全一致（确定性部分）。
    """
    mu, sigma, omega, lam = theta
    T = len(y)
    ps = np.empty(T, dtype=float)
    v = float(mu)
    pi = float(sigma)
    for t in range(T):
        pi_hat = 1.0 / (1.0 / pi + omega)
        v_hat = v
        p = sigmoid(v_hat)
        ps[t] = p
        sp = p * (1.0 - p)
        delta = y[t] - p
        v = v_hat + (lam / pi_hat) * sp * delta
        pi = pi_hat + lam * sp * sp
    return ps


def nll(y, theta, eps=1e-12):
    """负对数似然：-sum_t log p(y_t | 模型, 参数)。"""
    ps = np.clip(forward(y, theta), eps, 1.0 - eps)
    return -np.sum(y * np.log(ps) + (1.0 - y) * np.log(1.0 - ps))


# ----------------------------------------------------------------------------
# 参数估计（论文 2.2 节：最大似然 + 多起点数值优化）
# ----------------------------------------------------------------------------

def _transform(theta):
    """正参数 (sigma, omega, lambda) 取对数；mu 保持原值（无界优化）。"""
    mu, sigma, omega, lam = theta
    return np.array([mu, np.log(sigma), np.log(omega), np.log(lam)])


def _untransform(z):
    return np.array([z[0], np.exp(z[1]), np.exp(z[2]), np.exp(z[3])])


def _obj(z, y):
    return nll(y, _untransform(z))


# 代表性起始点（覆盖论文合理参数范围），保证优化不因起点远离而错失主峰
_REPRESENTATIVE_STARTS = [
    [0.0, 0.5, 0.05, 1.0],
    [0.0, 2.0, 0.05, 3.0],
    [0.0, 3.0, 0.10, 5.0],
    [-1.0, 1.0, 0.10, 2.0],
    [1.0, 1.0, 0.02, 4.0],
]


def fit(y, n_starts=20, seed=0, method="L-BFGS-B", bounds=None):
    """
    最大似然估计：对 log 变换后的参数做多起点数值优化，返回
    (theta_hat, nll_min)。

    - 正参数 (sigma, omega, lambda) 在 log 空间优化；
    - n_starts 个起点：前若干个为代表性起点（覆盖论文参数范围），
      其余为固定 seed 的随机起点；保证优化可复现（固定种子）。
    - 边界经适度收紧，避免 sigma/omega 退化解挤占全局主峰。
    """
    rng = np.random.default_rng(seed)
    if bounds is None:
        # 在 log 空间给出合理边界（收紧，防退化）
        bounds = [(-6, 6),          # mu
                  (-3, 2.5),        # log sigma   -> sigma in ~[0.05, 12]
                  (-4, 1.0),        # log omega   -> omega in ~[0.0002, 2.7]
                  (-4, 4.0)]        # log lambda  -> lambda in ~[0.018, 55]
    best_z = None
    best_nll = np.inf
    n_repr = min(len(_REPRESENTATIVE_STARTS), n_starts)
    for i in range(n_starts):
        if i < n_repr:
            z0 = np.log(np.maximum(np.array(_REPRESENTATIVE_STARTS[i]), 1e-12))
            z0[0] = _REPRESENTATIVE_STARTS[i][0]   # mu 不取对数
        else:
            z0 = np.array([
                rng.uniform(-2, 2),
                rng.uniform(-1, 2),
                rng.uniform(-2, 0),
                rng.uniform(-1, 2),
            ])
        # 把起点投影到边界内
        z0 = np.clip(z0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(_obj, z0, args=(y,), method=method, bounds=bounds)
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_z = res.x
    return _untransform(best_z), best_nll


# ----------------------------------------------------------------------------
# 参数网格（Step 1 参数恢复用）
# ----------------------------------------------------------------------------

def default_grid(n_mu=3, n_sigma=3, n_omega=3, n_lam=3, seed=0):
    """构造覆盖论文参数范围的 n 个真实参数向量（确定性网格）。"""
    rng = np.random.default_rng(seed)
    mus = rng.uniform(-2.0, 2.0, n_mu)
    sigmas = 10 ** rng.uniform(-0.5, 1.5, n_sigma)
    omegas = 10 ** rng.uniform(-1.5, 0.0, n_omega)
    lams = 10 ** rng.uniform(-0.5, 1.5, n_lam)
    grid = []
    for mu in mus:
        for sigma in sigmas:
            for omega in omegas:
                for lam in lams:
                    grid.append(np.array([mu, sigma, omega, lam]))
    return np.array(grid)


if __name__ == "__main__":
    # 快速自检：模拟 + 拟合一个数据集
    rng = np.random.default_rng(0)
    theta_true = np.array([0.5, 2.0, 0.05, 3.0])
    y = simulate(theta_true, T=120, seed=1)
    theta_hat, ll = fit(y, n_starts=5, seed=0)
    print("true :", np.round(theta_true, 3))
    print("hat  :", np.round(theta_hat, 3))
    print("nll  :", round(ll, 3))
