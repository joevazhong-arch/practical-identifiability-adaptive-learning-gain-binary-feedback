# -*- coding: utf-8 -*-
"""
paths.py
========
统一的路径解析，保证仓库在任意机器、任意目录、任意用户名/盘符下都能运行
（仓库中不再出现任何硬编码的绝对路径）。

两套位置：

- 归档（论文报告的权威值）
    JSON 位于仓库根目录（``review3_diagnostics*.json``），图位于 ``figures/``。

- 重新生成（复现）
    当设置环境变量 ``REPRO_DIR`` 后，所有脚本把新输出写到该目录
    （JSON 直接位于 ``REPRO_DIR``，图位于 ``REPRO_DIR/figures``）；并且
    “读取上一步结果”时也从 ``REPRO_DIR`` 读取，从而实现端到端的全新复现，
    全程不触碰归档值。

默认（不设置 ``REPRO_DIR``）时行为与历史一致：脚本在仓库根就地读写。
``reproduce.py`` 会自动把 ``REPRO_DIR`` 指向仓库内的 ``regenerated/``。
"""

import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def _repro_dir():
    """返回 REPRO_DIR 的绝对路径；未设置时为 None。"""
    d = os.environ.get("REPRO_DIR")
    return os.path.abspath(d) if d else None


def data_dir():
    """输入 JSON 所在目录：复现时为 REPRO_DIR，否则为仓库根。"""
    return _repro_dir() or CODE_DIR


def output_dir():
    """输出 JSON 所在目录：复现时为 REPRO_DIR（自动创建），否则为仓库根。"""
    d = _repro_dir()
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    return CODE_DIR


def data_path(name):
    """读取上一步 JSON 的完整路径。"""
    return os.path.join(data_dir(), name)


def output_path(name):
    """写出 JSON 的完整路径。"""
    return os.path.join(output_dir(), name)


def figure_path(name):
    """写出 PNG 的完整路径：复现时为 REPRO_DIR/figures，否则为仓库 figures。"""
    d = _repro_dir()
    fig = os.path.join(d, "figures") if d else os.path.join(CODE_DIR, "figures")
    os.makedirs(fig, exist_ok=True)
    return os.path.join(fig, name)
