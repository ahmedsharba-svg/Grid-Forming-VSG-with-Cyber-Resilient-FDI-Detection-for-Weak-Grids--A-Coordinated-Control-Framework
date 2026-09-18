# GFM/VSG + Cyber-Resilient FDI Detection — Simulation Toolkit

A small simulation toolkit for a grid-forming inverter under Virtual
Synchronous Generator (VSG) control, coupled to an RBF-SVM detector for
False Data Injection (FDI) attacks, aimed at weak-grid interconnection
scenarios (low short-circuit ratio, reduced synchronous inertia).

It includes:

- A calibrated single-diode PV module model with a 3-section
  bypass-diode partial-shading extension.
- A first-order swing-equation solver comparing a VSG-controlled
  grid-forming inverter against a minimal-inertia grid-following
  baseline.
- An RBF-SVM classifier for detecting False Data Injection attacks from
  three power-quality features, trained and cross-validated on a
  synthetic dataset with realistic class overlap in both directions.
- An ANN-guided PSO maximum-power-point tracker, including an
  adaptive-window mechanism that falls back to a full-range search when
  the ANN's initial guess does not lead to convergence.

## 1. Requirements

- Python 3.9+
- A terminal (Linux, macOS, or Windows/WSL)

## 2. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run everything at once

```bash
python3 run_all_experiments.py
```

Prints, and saves to `results/results.xlsx` (one sheet per table):

1. GFM/VSG frequency response (nadir, RoCoF) vs. a GFL baseline, plus a
   downsampled time-series trace
2. GFM voltage step response (overshoot, settling time, steady-state error)
3. A consolidated GFM/VSG performance summary (both of the above,
   alongside the GFL baseline)
4. A parameter ablation isolating the contribution of inertia (H) and
   damping (Dp) -- uses the same 2.0 s settling window as (1), since the
   weak-damping rows settle slowly and a shorter window under-reports
   their nadir
5. ANN training via genuine Levenberg-Marquardt optimisation
6. FDI detection performance via 10-fold cross-validated SVM training,
   plus the raw confusion matrix
7. An FDI feature ablation (accuracy, precision, recall, F1, AUC-ROC)
   across every non-empty subset of the three input features

Expected runtime: **~1–2 minutes** (the ANN training step, using true
Levenberg-Marquardt via `scipy.optimize.least_squares`, is the slowest
part, typically 30–60 seconds).

## 4. Notes on run-to-run variation

Re-running this code may give slightly different numbers each time
(e.g. 94.5% vs. 94.6% accuracy, or an AUC-ROC of 0.964 vs. 0.976). This
is expected and comes from ordinary sources of variation:

- **Random seeds**: the FDI classifier and ANN training both involve
  randomised data splits; different NumPy/scikit-learn versions can
  produce slightly different pseudo-random sequences from the same
  seed.
- **Solver tolerances**: the swing-equation integration uses a
  fixed-step Euler method — a finer `dt` or longer `t_end` in
  `vsg_control.py`'s `frequency_response()` will converge closer to the
  exact closed-form values.
- **The "(B) Strong H, weak Dp" and "(A) Weak H, weak Dp" ablation
  rows** are sensitive to the chosen simulation window (`t_end`)
  because their damping is very weak, so they settle slowly (time
  constant τ = J/Dp ≈ 1 s). `run_all_experiments.py` uses `t_end = 2.0`
  s for the ablation, matching `frequency_response()`'s default, so
  that all rows reach the same settled reference; using a shorter or
  longer window will shift these two rows' reported nadir.

None of this affects the qualitative conclusions: which parameter fixes
which target (H → RoCoF, Dp → nadir), whether the FDI detector meets
its accuracy targets, or whether the three FDI features are
individually redundant on this synthetic dataset.

## 5. Generating plots

```bash
python3 make_plots.py
```

Writes three PNGs to `./results/figures/`:

| File | Content |
|---|---|
| `frequency_response.png` | GFM/VSG vs. GFL: delta-f(t) and RoCoF(t) |
| `voltage_step_response.png` | GFM inverter voltage step response |
| `confusion_matrix.png` | FDI detector confusion matrix |

## 5b. Output layout

After running both scripts, results are organised as:

```
results/
├── results.xlsx        # every table, one sheet each (see Section 3)
└── figures/
    ├── frequency_response.png
    ├── voltage_step_response.png
    └── confusion_matrix.png
```

## 6. Running with Google Colab

Open `run_all_experiments_colab.ipynb` in Google Colab (upload it via
File → Upload notebook), then run the cells in order — the first cell
lets you upload this repository as a zip, after which the rest installs
dependencies and runs everything.

## 7. Running pieces individually

```bash
python3 -m src.vsg_control      # prints GFM/GFL nadir and RoCoF
python3 -m src.neural_pso       # trains the ANN once, prints test RMSE
python3 -m src.fdi_detector     # trains + cross-validates the SVM
python3 -m src.pv_model         # prints the calibrated STC operating point
```

## 8. File overview

| File | What it does |
|---|---|
| `src/vsg_control.py` | Swing-equation integration for the VSG-controlled inverter and the minimal-inertia GFL baseline. |
| `src/fdi_detector.py` | Synthetic dataset generation and RBF-SVM training/cross-validation for FDI detection. |
| `src/neural_pso.py` | ANN (Levenberg-Marquardt training) and PSO refinement for MPPT, including the adaptive-window mechanism. |
| `src/pv_model.py` | Calibrated single-diode PV model and 3-section bypass-diode partial-shading model. |
| `src/mppt_algorithms.py` | Reference implementations of P&O, INC, GWO, and ABC. |
| `run_all_experiments.py` | Runs everything above and prints the results described in Section 3. |
| `make_plots.py` | Generates the plots described in Section 5. |

## 9. If something looks wrong

If a value differs by more than the variation described in Section 4,
please check:

1. You're using the `requirements.txt` versions (very old/new
   scikit-learn or scipy versions can shift SVM/optimizer behaviour
   more than usual).
2. You ran the scripts from *inside* this directory (they use relative
   imports, `from src...`).
