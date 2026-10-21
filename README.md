# Practical Identifiability of a Minimal Adaptive Learning-Gain Model for Binary Feedback: A Simulation Study

Reproducibility package for the manuscript **"Practical Identifiability of a Minimal Adaptive Learning-Gain Model for Binary Feedback: A Simulation Study"** and its Supplementary Information.

This repository contains the model implementation, the simulation and analysis scripts, the **archived outputs** reported in the paper, the supplement figures, and a single command that regenerates every result from scratch and checks it against the archived values. There is **no external data**: every result is produced by fixed-seed simulations.

---

## Status and honesty notice (read before citing)

- The code, fixed seeds, archived outputs and figures are provided in this repository.
- The authors are **Wei Li** and **Jie Zhong** (corresponding author; jzhong@pku.edu.cn), and the license/copyright metadata now name both authors. The **public, persistent repository URL** and the **Zenodo DOI** do not yet exist; they will be created when the repository is published as a versioned release. These fields are marked `TODO` and must be completed before publication; they are not invented here.
- **Independent clean-environment reproduction** by someone other than the script author: **not yet performed** at the time of package preparation.
- **Author clean-environment verification: performed.** The script author ran `python reproduce.py` end-to-end in a fresh virtual environment (Python 3.13; NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2). `verify_reproduction.py` reported all ten output families `PASS` (28,972 numerical values; maximum absolute difference 0 under tolerance 1e-6) and the six supplementary figures were reproduced byte-for-byte (identical SHA-256). This confirms the package regenerates the reported results; it is not the independent third-party reproduction noted above.

---

## What is reproduced

All supplementary tables and figures (**Tables S1–S10, Figures S1–S6**) and the numbers behind the main-text results are produced by the `review3_supplement*.py` scripts. The earlier pilot scripts (`run_all.py`, `validation.py`, etc.) are **not** part of the reported analyses and are intentionally excluded.

## Requirements

- **Python 3.10+** (the documented run below used Python 3.13).
- `numpy`, `scipy`, `matplotlib` — see `requirements.txt`.
- For an exact, byte-for-byte environment, use the fully pinned `requirements-lock.txt`.

## Quick start (clean environment)

```bash
# 1. create and activate a fresh virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2. install dependencies (use requirements-lock.txt for exact versions)
pip install -r requirements.txt

# 3. regenerate everything and verify against the archived values
python reproduce.py
```

`reproduce.py` writes **only** into `regenerated/` (JSON) and `regenerated/figures/` (PNG); it never overwrites the archived files in the repository root or `figures/`.

### Runtime

The pipeline is computationally heavy: it performs several thousand multi-start
fits (the grid and multi-T recovery alone cover roughly 2,900 datasets, plus
rolling-origin recovery and ablations). On a single CPU core the full
`reproduce.py` takes on the order of **1–2 hours** (the first step,
`review3_supplement.py`, is the longest). Use `--no-figures` to skip the
figures if you only need the JSON, or `--step N` to run a single family. The
archived outputs and figures are already included, so a full run is only
needed to verify reproducibility rather than to read the results.

### Optional flags

- `python reproduce.py --no-figures` — regenerate the JSON analyses only.
- `python reproduce.py --no-test` — skip the regression test.
- `python reproduce.py --no-verify` — regenerate without the final comparison.
- `python reproduce.py --step N` — run only analysis step `N` (1–10).

You can also run the comparison on its own:

```bash
python verify_reproduction.py
```

The verifier exits `0` when every regenerated value matches the archive within
absolute tolerance `1e-6` (or relative tolerance `1e-6`); integer counts,
indices, strings and key structure must match exactly.

---

## Analysis → output → manuscript map

Each analysis script writes one `review3_diagnostics*.json` file. Scripts are
run in the order below; later steps read earlier outputs from the same
(regenerated) directory.

| Step | Script | Output file | Contents | Tables / figures |
|---|---|---|---|---|
| 1 | `review3_supplement.py` | `review3_diagnostics.json` | 24-condition grid recovery (R=100, T=80); multi-T recovery (T=40–480); 1-D profile likelihoods; 2-D trade-off surfaces; model recovery (R=50); ablations; boundary sensitivity; unit checks | **Table S1**; data for **Fig S1–S2** |
| 2 | `review3_supplement2.py` | `review3_diagnostics2.json` | Out-of-sample log-loss difference distribution (R=50); multi-T ablation | input to step 3; main-text prediction results |
| 3 | `review3_supplement3.py` | `review3_diagnostics3.json` | Correct-specification true-vs-fitted NLL audit; paired OOS difference CIs | **Table S2** |
| 4 | `review3_supplement4.py` | `review3_diagnostics4.json` | Per-replicate true-vs-estimated scatter data; paired ablations | data for **Fig S3–S4** |
| 5 | `review3_supplement5.py` | `review3_diagnostics5.json` | Tolerance sensitivity (0.3/0.6/1.0); ranking; Wilson CIs; bootstrap CIs | **Tables S4–S5**; data for **Fig S5–S6** |
| 6 | `review3_supplement6.py` | `review3_diagnostics6.json` | 50-dataset fixed-split OOS paired comparison + bootstrap | **Table S7** |
| 7 | `review3_supplement7.py` | `review3_diagnostics7.json` | Block rolling-origin prediction; independent pure-Python likelihood cross-check; low-gain regime profiles | **Tables S8–S9** |
| 8 | `review3_supplement8.py` | `review3_diagnostics8.json` | Single authoritative paired ablation (10,000 bootstrap) | **Tables S3, S6** |
| 9 | `review3_supplement9.py` | `review3_diagnostics9.json` | Forecast-index / no-look-ahead causality audit; calibrated permutation null | supports **Tables S8–S9** |
| 10 | `review3_supplement10.py` | `review3_diagnostics10.json` | Three-protocol fitting parity (corrected common/wide vs pre-correction control); return-order vs bound separation | **Table S10**; primary **Table S8** rows |

### Figures

| Script | Figures produced | Data source |
|---|---|---|
| `supplement_figs.py` | **Fig S1** (1-D profiles), **Fig S2** (2-D trade-offs) | `review3_diagnostics.json` |
| `make_fig_s3_s4.py` | **Fig S3** (grid scatter), **Fig S4** (multi-T scatter) | `review3_diagnostics4.json` |
| `make_fig_s5_s6.py` | **Fig S5**, **Fig S6** (paired-difference plots) | `review3_diagnostics5.json` |

The archived copies of these figures are in `figures/`. Axis labels and titles
are embedded in the rendered PNGs and are not edited here.

---

## Model code

- **`adaptive_learning_gain.py`** — the proposed four-parameter adaptive
  learning-gain model (the binary-feedback adaptive filter): `simulate`,
  `forward`, `nll`, multi-start `fit`, and `default_grid`.
- **`competitors.py`** — the reference models (Rescorla–Wagner, fixed-gain
  Kalman), BIC and k-fold helpers, and the unified `MODELS` dictionary.
- **`review2_diagnostics.py`** — shared utilities for variant
  simulation/fitting, bounds and rolling log-loss.
- **`paths.py`** — resolves all input/output locations with no hard-coded
  absolute paths; the `REPRO_DIR` environment variable redirects outputs to
  `regenerated/`.

## Two fixed-gain fits: corrected vs pre-correction (historical control)

`competitors.py` deliberately contains **two** fixed-gain Kalman fitters. Do
not conflate them:

- **`fit_kalman`** — the **corrected** fitter. It returns parameters in the
  `(mu, sigma, omega)` order that `forward_kalman` expects. This is the
  correct routine and is used for the corrected primary model-recovery
  results in `review3_supplement10.py` (67/150 = 0.447 wide bounds;
  72/150 = 0.480 common bounds).
- **`fit_kalman_pre_correction`** — an explicit **historical-control**
  fitter. It runs the identical optimisation (the objective and reported
  optimum are correct) but returns the parameters in the **pre-V16 swapped
  order** `(mu, omega, sigma)`, which `forward_kalman` misreads. It exists
  only so that the archived pre-correction results can be regenerated
  bit-for-bit: the three-observation rolling-origin pilot (**Table S5**,
  57/150 = 0.38; `review3_supplement.py`, `review3_supplement7.py`) and the
  pre-correction block protocol (**Table S8** control row / **Text S6**,
  58/150 = 0.387, permutation p = 0.079; `review3_supplement7.py`,
  `review3_supplement9.py`, `review3_supplement10.py`).

The swapped-order path was an interface defect (a parameter return-order
bug), not evidence that the fixed-gain model is intrinsically
indistinguishable; the paper reports both the corrected primary results and
the pre-correction historical control, and labels them as such. **Do not use
`fit_kalman_pre_correction` for new analyses.** The regression test below
guards both routines so the historical controls cannot silently change.

## Tests

- **`test_kalman_order.py`** — regression test for the fixed-gain Kalman
  parameter return order `(mu, sigma, omega)`. It asserts that the corrected
  `fit_kalman` return order is consistent with `forward_kalman` (NLL equals
  the optimum ≈ 50.46 for seed=7); that manually swapping the 2nd/3rd
  entries reproduces the historical defective NLL (≈ 1493); and that
  `fit_kalman_pre_correction` reproduces the same defective value while the
  optimiser still reports the correct optimum. Run it directly or with
  `pytest`:

```bash
python test_kalman_order.py
# or
pytest -q
```

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── requirements-lock.txt
├── paths.py
├── reproduce.py
├── verify_reproduction.py
├── adaptive_learning_gain.py
├── competitors.py
├── review2_diagnostics.py
├── review3_supplement.py
├── review3_supplement2.py … review3_supplement10.py
├── supplement_figs.py
├── make_fig_s3_s4.py
├── make_fig_s5_s6.py
├── test_kalman_order.py
├── review3_diagnostics.json … review3_diagnostics10.json   # archived (paper) values
├── figures/                    # archived Figures S1–S6
└── regenerated/                # regenerated outputs (git-ignored except .gitkeep)
```

## Archived vs regenerated outputs

The JSON files in the repository root are the **archived, paper-reported**
values and are referenced by the same file names in Supplementary Text S4.
Running `reproduce.py` regenerates the equivalent files into `regenerated/`
and compares them, so the authoritative values are never overwritten. Because
all randomness is controlled by fixed seeds, the regenerated results match.

## Authors and corresponding author

- **Wei Li** — Department of Psychiatry, Shanxi Bethune Hospital, Shanxi Academy of Medical Sciences, Third Hospital of Shanxi Medical University, Tongji Shanxi Hospital, Taiyuan 030032, China.
- **Jie Zhong** — School of Psychological and Cognitive Sciences, Peking University, Beijing 100871, China. **Corresponding author**, e-mail: jzhong@pku.edu.cn.

This work was supported by the Peking University–Guixian Education Joint Laboratory for Family and Child Development. The authors declare no competing interests.

## License

Released under the MIT License, Copyright © 2026 Wei Li and Jie Zhong; see [`LICENSE`](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff). The repository URL and DOI fields are
marked `TODO` and will be completed once the versioned release is published
and archived.
