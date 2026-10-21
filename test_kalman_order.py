# -*- coding: utf-8 -*-
"""
test_kalman_order.py
====================
回归测试：固定增益受限自适应滤波器的参数返回顺序，必须与
``forward_kalman`` / ``simulate_kalman`` 期望的 ``(mu, sigma, omega)`` 一致。

历史上 ``fit_kalman`` 曾错误地返回 ``(mu, omega, sigma)``，使预测把
``sigma`` 与 ``omega`` 位置互换；对代表性数据集（seed=7）这把拟合块
NLL 从约 50.46 抬高到约 1493.20，曾导致固定增益模型几乎永不被选中
（那是接口缺陷，而非真正的不可区分）。本测试断言该缺陷已被修复：

  * ``fit_kalman`` 返回顺序与 ``forward_kalman`` 一致，重新计算的 NLL
    等于优化器报告的最优值；
  * 若手动把第 2/3 个返回值交换（重现旧缺陷），NLL 跳到约 1493。

可直接运行（``python test_kalman_order.py``），也可被 pytest 收集。
"""

import os
import sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from competitors import (simulate_kalman, fit_kalman,
                          fit_kalman_pre_correction, _nll_kalman)  # noqa: E402


def check_return_order(seed=7):
    y = simulate_kalman((0.5, 2.0, 0.05), 80, seed=seed)
    theta, n_opt = fit_kalman(y, n_starts=20, seed=0)
    mu, sigma, omega = theta

    # returned order is consistent with forward_kalman -> NLL equals optimum
    nll_as_returned = _nll_kalman(y, (mu, sigma, omega))
    # swapping the 2nd/3rd entries reproduces the old (defective) path
    nll_legacy_swapped = _nll_kalman(y, (mu, omega, sigma))

    print("seed=%d" % seed)
    print("  optimized best_nll       = %.4f" % n_opt)
    print("  nll as returned (fixed) = %.4f" % nll_as_returned)
    print("  nll legacy swapped      = %.4f" % nll_legacy_swapped)

    assert np.isclose(nll_as_returned, n_opt, atol=1e-6), (
        "returned parameter order inconsistent with forward_kalman")
    assert nll_legacy_swapped > n_opt + 100.0, (
        "legacy swapped path did not reproduce the defective NLL inflation")
    return n_opt, nll_as_returned, nll_legacy_swapped


def check_pre_correction(seed=7):
    """The historical-control routine must reproduce the swapped order.

    ``fit_kalman_pre_correction`` runs the identical optimisation, so the
    optimizer-reported optimum is unchanged (~50.46); only the returned
    parameter order is swapped. When that tuple is fed back into
    ``forward_kalman`` (as the original scoring code did), the NLL inflates
    to the documented defective value (~1493.20). This guards the archived
    pre-correction controls (57/150 pilot, 58/150 block) against accidental
    "fixing" that would make them impossible to reproduce.
    """
    y = simulate_kalman((0.5, 2.0, 0.05), 80, seed=seed)
    (mu_c, sig_c, om_c), n_c = fit_kalman(y, n_starts=20, seed=0)
    (mu_p, sig_p, om_p), n_p = fit_kalman_pre_correction(y, n_starts=20, seed=0)

    assert np.isclose(mu_c, mu_p, atol=1e-8), "mu must agree"
    assert np.isclose(sig_c, om_p, atol=1e-8), "swapped order (sigma/omega)"
    assert np.isclose(om_c, sig_p, atol=1e-8), "swapped order (sigma/omega)"
    # identical optimisation -> identical reported objective value
    assert np.isclose(n_c, n_p, atol=1e-8), "objective should be identical"
    # interpreted by forward_kalman the pre-correction tuple is defective
    nll_pre = _nll_kalman(y, (mu_p, sig_p, om_p))
    print("  pre-correction nll     = %.4f (optimum reported %.4f)"
          % (nll_pre, n_p))
    assert np.isclose(nll_pre, 1493.1953, atol=2.0), nll_pre
    assert nll_pre > n_c + 100.0, "pre-correction path not defective"
    return nll_pre


def test_return_order_seeds():
    for s in (0, 1, 2, 7):
        check_return_order(seed=s)


if __name__ == "__main__":
    n_opt, nll_fixed, nll_legacy = check_return_order(seed=7)
    # paper-reported reference values (Supplementary Text S7), seed=7
    assert np.isclose(n_opt, 50.4632, atol=0.1), n_opt
    assert np.isclose(nll_fixed, 50.4632, atol=0.1), nll_fixed
    assert np.isclose(nll_legacy, 1493.1953, atol=2.0), nll_legacy
    nll_pre = check_pre_correction(seed=7)
    print("ALL CHECKS PASSED")
