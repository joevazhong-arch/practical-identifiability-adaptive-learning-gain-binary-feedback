# -*- coding: utf-8 -*-
"""
review3_supplement8.py  (V14 response to MC6)
=============================================
Single authoritative recomputation of the paired ablation results.

The V13 package computed win counts twice from different precisions:
review3_supplement4.py counted wins from unrounded per-dataset log-loss,
while review3_supplement5.py re-counted from per-dataset values rounded to
5 decimals. A handful of near-zero differences changed sign under rounding,
so Tables S3 and S6 disagreed by one dataset in four full-vs-precision
comparisons.

This script regenerates the SAME paired datasets (identical seeds and
settings) at full precision and computes, from ONE per-dataset array per
comparison: mean, SD, normal-approximation 95% CI, paired-bootstrap 95% CI
(10,000 resamples), and win/tie counts. All values are reported at 5 decimals
so endpoints near zero are legible. Direction: full minus variant; NEGATIVE
means the full model has lower log-loss.

Output: review3_diagnostics8.json (the single source for Tables S3 and S6).
"""
import os
import json
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, CODE_DIR)

import review3_supplement4 as r4   # noqa: E402
import paths                       # noqa: E402

SEED = r4.SEED
BOOT_B = 10000
BOOT_SEED = 12345
SETTINGS = [np.array([0.5, 2.0, 0.05, 3.0]),
            np.array([0.0, 3.0, 0.10, 5.0]),
            np.array([-1.0, 1.5, 0.02, 2.0])]
MOD_T = (0.5, 2.0, 0.05, 3.0)
VARIANTS = ["mean_wo_sprime", "precision_simple", "rw"]
COMPS = ["full_vs_mean_wo_sprime", "full_vs_precision_simple", "full_vs_rw"]
KEYS = ["mean_wo_sprime", "precision_simple", "rw"]

rng = np.random.default_rng(BOOT_SEED)


def _collect(theta, Tn, R, gen_seed_offset):
    """Generate R paired datasets (identical y for all 4 models) at full precision."""
    per = {k: [] for k in (["full"] + VARIANTS)}
    for r in range(R):
        y = r4._sim_variant(np.asarray(theta, dtype=float), Tn,
                            gen_seed_offset + r, "full")
        sc = r4._scores_one_dataset(y)
        per["full"].append(sc["full"])
        for k in VARIANTS:
            per[k].append(sc[k])
    return per


def _stats(full, variant):
    d = np.asarray(full, dtype=float) - np.asarray(variant, dtype=float)
    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    norm_lo = mean - 1.96 * sd / np.sqrt(n)
    norm_hi = mean + 1.96 * sd / np.sqrt(n)
    idx = rng.integers(0, n, size=(BOOT_B, n))
    bm = d[idx].mean(axis=1)
    blo, bhi = np.percentile(bm, [2.5, 97.5])
    return {
        "n": int(n),
        "mean_diff_full_minus_variant": round(mean, 5),
        "sd_diff": round(sd, 5),
        "normal_ci95": [round(float(norm_lo), 5), round(float(norm_hi), 5)],
        "bootstrap_ci95": [round(float(blo), 5), round(float(bhi), 5)],
        "datasets_full_better": int(np.sum(d < 0.0)),
        "datasets_variant_better": int(np.sum(d > 0.0)),
        "datasets_tied": int(np.sum(d == 0.0)),
        "direction": "full minus variant; negative = full lower log-loss (better)",
    }


def _block(per):
    comps = {lab: _stats(per["full"], per[k]) for lab, k in zip(COMPS, KEYS)}
    return {
        "comparisons": comps,
        "per_dataset_rolling_logloss": {
            k: [float(x) for x in v] for k, v in per.items()},
    }


def main():
    out = {"seed": SEED, "bootstrap_seed": BOOT_SEED, "bootstrap_B": BOOT_B,
           "direction": "full minus variant; negative = full lower log-loss",
           "m1_t80": {}, "m2_multi_T": {}}
    print("== M1 paired ablation (3 settings x R30, T=80) ==", flush=True)
    for si, theta in enumerate(SETTINGS):
        per = _collect(theta, 80, 30, SEED + si * 1000)
        blk = _block(per)
        out["m1_t80"][f"setting_{si}"] = {
            "generating_theta": [float(a) for a in theta], "T": 80, "R": 30,
            **{k: v for k, v in blk.items() if k != "per_dataset_rolling_logloss"}}
        out["m1_t80"][f"setting_{si}"]["per_dataset_rolling_logloss"] = \
            blk["per_dataset_rolling_logloss"]
        for lab in COMPS:
            c = blk["comparisons"][lab]
            print(f"  setting_{si} {lab:26s} mean={c['mean_diff_full_minus_variant']:+.5f} "
                  f"normal={c['normal_ci95']} boot={c['bootstrap_ci95']} "
                  f"full_better={c['datasets_full_better']}/{c['n']}", flush=True)
    print("== M2 paired ablation (moderate, T=40/80/120, R15) ==", flush=True)
    for Tn in (40, 80, 120):
        per = _collect(MOD_T, Tn, 15, SEED + Tn * 100)
        blk = _block(per)
        out["m2_multi_T"][str(Tn)] = {
            "T": Tn, "R": 15,
            "comparisons": blk["comparisons"],
            "per_dataset_rolling_logloss": blk["per_dataset_rolling_logloss"]}
        for lab in COMPS:
            c = blk["comparisons"][lab]
            print(f"  T={Tn:>3d} {lab:26s} mean={c['mean_diff_full_minus_variant']:+.5f} "
                  f"normal={c['normal_ci95']} boot={c['bootstrap_ci95']} "
                  f"full_better={c['datasets_full_better']}/{c['n']}", flush=True)
    path = paths.output_path("review3_diagnostics8.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("SAVED", path, flush=True)


if __name__ == "__main__":
    main()
