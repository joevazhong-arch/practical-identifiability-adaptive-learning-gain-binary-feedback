# -*- coding: utf-8 -*-
"""
verify_reproduction.py
======================
核对归档结果（仓库根目录的 ``review3_diagnostics*.json``，即论文报告值）
与重新生成结果（``regenerated/`` 下同名文件）是否一致。

递归遍历每个 JSON，逐数值比较：
  * 绝对差 <= ABS_TOL (1e-6)，或
  * 相对差 <= REL_TOL (1e-6)，
视为通过。整数计数、索引、字符串与键结构必须完全一致。

用法：
  python verify_reproduction.py
  python verify_reproduction.py --regenerated-dir <path>
退出码：0 全部通过；1 存在差异。
"""

import argparse
import json
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGEN = os.path.join(CODE_DIR, "regenerated")

FILES = ["review3_diagnostics.json"] + [
    "review3_diagnostics%d.json" % n for n in range(2, 11)]

ABS_TOL = 1e-6
REL_TOL = 1e-6


def _new_stats():
    return {"n": 0, "max_abs": 0.0, "max_rel": 0.0,
            "mismatch": 0, "examples": [], "structural": []}


def _compare(a, b, path, st):
    # bool must be checked before int (bool is a subclass of int)
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            st["structural"].append((path, a, b))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        st["n"] += 1
        d = abs(float(a) - float(b))
        scale = max(abs(float(a)), abs(float(b)), 1e-12)
        rd = d / scale
        st["max_abs"] = max(st["max_abs"], d)
        st["max_rel"] = max(st["max_rel"], rd)
        if d > ABS_TOL and rd > REL_TOL:
            st["mismatch"] += 1
            if len(st["examples"]) < 10:
                st["examples"].append((path, a, b, "%.3e" % d, "%.3e" % rd))
        return
    if isinstance(a, str) or isinstance(b, str):
        if a != b:
            st["structural"].append((path, a, b))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a:
                st["structural"].append(
                    ("%s.%s" % (path, k), "archived-missing", "regenerated-only"))
            elif k not in b:
                st["structural"].append(
                    ("%s.%s" % (path, k), "archived-only", "regenerated-missing"))
            else:
                _compare(a[k], b[k], "%s.%s" % (path, k), st)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            st["structural"].append((path + ".length", len(a), len(b)))
        for i in range(min(len(a), len(b))):
            _compare(a[i], b[i], "%s[%d]" % (path, i), st)
        return
    if a != b:
        st["structural"].append((path, a, b))


def verify(archived_dir=CODE_DIR, regen_dir=DEFAULT_REGEN, verbose=True):
    overall = True
    for fn in FILES:
        ap = os.path.join(archived_dir, fn)
        rp = os.path.join(regen_dir, fn)
        if not os.path.exists(rp):
            if verbose:
                print("MISSING %s (regenerated file not found)" % fn)
            overall = False
            continue
        with open(ap, encoding="utf-8") as f:
            a = json.load(f)
        with open(rp, encoding="utf-8") as f:
            b = json.load(f)
        st = _new_stats()
        _compare(a, b, "$", st)
        ok = st["mismatch"] == 0 and not st["structural"]
        overall = overall and ok
        if verbose:
            print("%-28s values=%5d max_abs=%.2e max_rel=%.2e "
                  "mismatch=%d struct=%d  %s"
                  % (fn, st["n"], st["max_abs"], st["max_rel"],
                     st["mismatch"], len(st["structural"]),
                     "PASS" if ok else "FAIL"))
            for ex in st["examples"][:5]:
                print("    MISMATCH", ex)
            for sx in st["structural"][:5]:
                print("    STRUCT  ", sx)
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerated-dir", default=DEFAULT_REGEN)
    parser.add_argument("--archived-dir", default=CODE_DIR)
    args = parser.parse_args()
    ok = verify(args.archived_dir, args.regenerated_dir)
    print("VERIFICATION:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
