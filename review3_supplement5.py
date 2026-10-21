# -*- coding: utf-8 -*-
"""
review3_supplement5.py — V13 reviewer (2026-10-18) post-processing.
No new simulation: recomputes sensitivity / uncertainty statistics from the
existing per-replicate and per-dataset records already saved in
review3_diagnostics4.json (scatter records, paired per-dataset log-loss) and
review3_diagnostics.json (model-recovery confusion counts).

Produces:
  A) tolerance sensitivity at tol = 0.3 / 0.6 / 1.0 transformed units
     for the 24-condition grid (T=80) and the five-trial-length experiment.
  B) rank (individual-ordering) diagnostics: pooled Spearman and
     condition-mean Spearman of true vs estimated transformed parameters.
  C) Wilson score 95% CIs for model-recovery overall accuracy and row-wise
     recovery rates (from the confusion counts, n=50 per generating model).
  D) bootstrap (paired-percentile) 95% CIs for every ablation paired
     comparison (M1 three settings, M2 three trial lengths), alongside the
     existing normal-approximation CIs.
  E) raw per-dataset paired differences are saved for Fig S5/S6.

Output: review3_diagnostics5.json
"""
import json
import io
import os

import numpy as np
from scipy import stats

import paths
CODE_DIR = paths.CODE_DIR
J4 = paths.data_path("review3_diagnostics4.json")
J1 = paths.data_path("review3_diagnostics.json")
OUT = paths.output_path("review3_diagnostics5.json")

TOLS = [0.3, 0.6, 1.0]
PNAMES = ["mu", "sigma", "omega", "lambda"]
BOOT_B = 10000
SEED = 12345

with io.open(J4, encoding="utf-8") as f:
    d4 = json.load(f)
with io.open(J1, encoding="utf-8") as f:
    d1 = json.load(f)

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# A) tolerance sensitivity
# ---------------------------------------------------------------------------
def tol_rates_from_true_hat(true_arr, hat_arr):
    """true_arr, hat_arr: (N,4) transformed arrays. Returns {param:{tol:rate}}."""
    n = true_arr.shape[0]
    err = np.abs(hat_arr - true_arr)
    out = {}
    for j, p in enumerate(PNAMES):
        out[p] = {str(tol): round(float(np.mean(err[:, j] < tol)), 4) for tol in TOLS}
    return out


# grid: columns [condition, true*4, hat*4]
g = d4["scatter_grid_r100"]
grec = np.array(g["records"], dtype=float)
g_true = grec[:, 1:5]
g_hat = grec[:, 5:9]
tol_grid = tol_rates_from_true_hat(g_true, g_hat)

# per-condition (24 rows) tolerance rates, for supplementary table
tol_grid_bycond = []
for c in range(g["n_conditions"]):
    sel = grec[:, 0] == c
    rates = tol_rates_from_true_hat(g_true[sel], g_hat[sel])
    tol_grid_bycond.append({"condition": c, "rates": rates})

# multi-T: by_T -> {z_true:[4], records:[[hat*4],...]}
mt = d4["scatter_multi_T_r100"]["by_T"]
tol_multiT = {}
for Tk, v in sorted(mt.items(), key=lambda kv: int(kv[0])):
    z_true = np.array(v["z_true"], dtype=float)
    hat = np.array(v["records"], dtype=float)
    true_tile = np.tile(z_true, (hat.shape[0], 1))
    tol_multiT[Tk] = tol_rates_from_true_hat(true_tile, hat)


# ---------------------------------------------------------------------------
# B) rank (individual ordering) diagnostics, grid only (distinct true values)
# ---------------------------------------------------------------------------
def spearman(x, y):
    rho, p = stats.spearmanr(x, y)
    return round(float(rho), 3), round(float(p), 4)


rank_pooled = {p: spearman(g_true[:, j], g_hat[:, j]) for j, p in enumerate(PNAMES)}
# one point per condition (the "individual" = generating condition), mean estimate
cond_mean_hat = []
cond_true = []
for c in range(g["n_conditions"]):
    sel = grec[:, 0] == c
    cond_mean_hat.append(g_hat[sel].mean(axis=0))
    cond_true.append(g_true[sel][0])
cond_mean_hat = np.array(cond_mean_hat)
cond_true = np.array(cond_true)
rank_cond_mean = {p: spearman(cond_true[:, j], cond_mean_hat[:, j])
                  for j, p in enumerate(PNAMES)}
# also rank when individual = single replicate, but condition-mean is the fair
# phenotyping comparison (n=24 distinct individuals)
ranking = {
    "pooled_spearman_rho_p": rank_pooled,
    "condition_mean_spearman_rho_p": rank_cond_mean,
    "n_conditions_individuals": g["n_conditions"],
    "n_replicates_per_condition": g["R"],
    "note": ("Spearman rank correlation between true and estimated transformed "
             "parameter across the 24 generating conditions (each condition is "
             "one distinct generating parameter vector; trial-length experiment "
             "uses a single true vector and therefore cannot assess ranking)."),
}


# ---------------------------------------------------------------------------
# C) Wilson score 95% CI for model recovery
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = (z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / den
    return round(float(center - half), 3), round(float(center + half), 3)


conf = d1["recovery_r50"]["confusion_counts"]
n_per = d1["recovery_r50"]["R"]
MODEL_ROWS = ["adaptive_filter", "rw", "kalman_fixed"]
diag_k = {r: int(conf[r][r]) for r in MODEL_ROWS}
total_k = sum(diag_k.values())
total_n = n_per * 3
recovery_ci = {
    "overall": {"k": total_k, "n": total_n, "estimate": round(total_k / total_n, 3),
                "wilson95": wilson(total_k, total_n), "nominal_chance": 0.333},
    "rows": {nm: {"k": diag_k[nm], "n": n_per, "estimate": round(diag_k[nm] / n_per, 3),
                  "wilson95": wilson(diag_k[nm], n_per)}
             for nm in MODEL_ROWS},
}


# ---------------------------------------------------------------------------
# D) bootstrap paired CIs for ablation
# ---------------------------------------------------------------------------
def bootstrap_paired(full, variant, B=BOOT_B):
    full = np.asarray(full, dtype=float)
    var = np.asarray(variant, dtype=float)
    d = full - var
    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    norm_lo = mean - 1.96 * sd / np.sqrt(n)
    norm_hi = mean + 1.96 * sd / np.sqrt(n)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = d[idx].mean(axis=1)
    blo, bhi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n": int(n),
        "mean_diff": round(mean, 4),
        "sd_diff": round(sd, 4),
        "normal_ci95": [round(float(norm_lo), 4), round(float(norm_hi), 4)],
        "bootstrap_ci95": [round(float(blo), 4), round(float(bhi), 4)],
        "datasets_full_better": int(np.sum(d < 0)),
        "datasets_variant_better": int(np.sum(d > 0)),
        "datasets_tied": int(np.sum(d == 0)),
    }


def abl_block(block, collect_diffs, key_prefix):
    """block: dict with per_dataset_rolling_logloss (one setting or one T)."""
    per = block["per_dataset_rolling_logloss"]
    full = per["full"]
    comps = {}
    for variant_key, label in [("mean_wo_sprime", "full_vs_mean_wo_sprime"),
                               ("precision_simple", "full_vs_precision_simple"),
                               ("rw", "full_vs_rw")]:
        stat = bootstrap_paired(full, per[variant_key])
        comps[label] = stat
        diffs = (np.asarray(full) - np.asarray(per[variant_key]))
        collect_diffs[f"{key_prefix}_{label}"] = np.round(diffs, 5).tolist()
    return comps


ablation_boot = {"m1_t80": {}, "m2_multi_T": {}}
all_diffs = {}
for skey, sblock in d4["ablation_paired_t80"].items():
    ablation_boot["m1_t80"][skey] = {
        "generating_theta": sblock["generating_theta"],
        "T": sblock["T"],
        "comparisons": abl_block(sblock, all_diffs, skey),
    }
for Tk, tblock in d4["ablation_paired_multi_T"]["by_T"].items():
    ablation_boot["m2_multi_T"][Tk] = {
        "comparisons": abl_block(tblock, all_diffs, f"T{Tk}"),
    }

# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
results = {
    "seed": SEED,
    "tolerances": TOLS,
    "tolerance_sensitivity_grid_t80": tol_grid,
    "tolerance_sensitivity_grid_by_condition": tol_grid_bycond,
    "tolerance_sensitivity_multi_T": tol_multiT,
    "ranking_grid": ranking,
    "model_recovery_wilson_ci": recovery_ci,
    "ablation_bootstrap": ablation_boot,
    "paired_diffs_for_figures": all_diffs,
}
with io.open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------------------
# console summary
# ---------------------------------------------------------------------------
print("=== A) tolerance sensitivity, grid T=80 (rate within tol) ===")
for p in PNAMES:
    print(f"  {p:7s} " + "  ".join(f"tol={t}: {tol_grid[p][str(t)]:.3f}" for t in TOLS))
print("=== A) tolerance sensitivity, multi-T representative ===")
for Tk in sorted(tol_multiT, key=int):
    row = tol_multiT[Tk]
    print(f"  T={Tk:>3s} " + " | ".join(
        f"{p}: " + "/".join(f"{row[p][str(t)]:.2f}" for t in TOLS) for p in PNAMES))
print("=== B) ranking (condition-mean Spearman rho, n=24) ===")
for p in PNAMES:
    print(f"  {p:7s} pooled={rank_pooled[p]}  cond-mean={rank_cond_mean[p]}")
print("=== C) model recovery Wilson 95% CI ===")
print("  overall:", recovery_ci["overall"])
for nm, v in recovery_ci["rows"].items():
    print(f"  {nm:16s} {v}")
print("=== D) ablation bootstrap vs normal CI (M1) ===")
for skey, sb in ablation_boot["m1_t80"].items():
    for ck, c in sb["comparisons"].items():
        print(f"  {skey} {ck:26s} mean={c['mean_diff']:+.4f} "
              f"normal={c['normal_ci95']} boot={c['bootstrap_ci95']} "
              f"full_better={c['datasets_full_better']}/{c['n']}")
print("=== D) ablation bootstrap (M2) ===")
for Tk in sorted(ablation_boot["m2_multi_T"], key=int):
    for ck, c in ablation_boot["m2_multi_T"][Tk]["comparisons"].items():
        print(f"  T={Tk:>3s} {ck:26s} mean={c['mean_diff']:+.4f} "
              f"normal={c['normal_ci95']} boot={c['bootstrap_ci95']} "
              f"full_better={c['datasets_full_better']}/{c['n']}")
print("SAVED", OUT)
