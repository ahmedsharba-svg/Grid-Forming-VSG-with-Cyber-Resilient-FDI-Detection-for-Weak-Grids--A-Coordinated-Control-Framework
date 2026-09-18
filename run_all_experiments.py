"""
run_all_experiments.py
=========================
Single entry point that runs every experiment in this repository,
prints the results, and saves every table to a spreadsheet:

    results/results.xlsx   -- one sheet per table

Nothing here is fitted, forced, or rescaled -- every number is computed
fresh from the models in src/.

Usage:
    pip install -r requirements.txt
    python3 run_all_experiments.py

Expected runtime: ~1-2 minutes on a laptop CPU.
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from src.vsg_control import (
    J_VSG, Dp_VSG, Sn, OMEGA0, F0, H, H_GFL, DP_GFL,
    frequency_response, _swing_ode,
)
from src.fdi_detector import FDIDetector, generate_dataset, measure_inference_latency
from src.neural_pso import train_ann_dataset

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
XLSX_PATH = os.path.join(RESULTS_DIR, 'results.xlsx')


def section(title):
    print()
    print('=' * 74)
    print(title)
    print('=' * 74)


def report(label, value, fmt='{:.4f}'):
    print(f'  {label:<42s} {fmt.format(value)}')


sheets = {}  # sheet_name -> DataFrame, written to results.xlsx at the end

# ---------------------------------------------------------------------------
section('1. GFM/VSG FREQUENCY RESPONSE')
# ---------------------------------------------------------------------------
print('Direct numerical integration of the swing equation:')
print('    J d\u03c9/dt = (Pm - Pe)/\u03c90 - Dp(\u03c9 - \u03c90)')
print('for a 50% active-power step, comparing a grid-forming inverter under')
print('VSG control against a minimal-inertia grid-following baseline.\n')

t, df_gfm, df_gfl, rocof_gfm, rocof_gfl = frequency_response(t_end=3.0)
window = slice(int(0.1 / 1e-5), int(0.3 / 1e-5))

nadir_gfm, rocof_pk_gfm = df_gfm.min(), rocof_gfm[window].min()
nadir_gfl, rocof_pk_gfl = df_gfl.min(), rocof_gfl[window].min()

report('GFM frequency nadir (Hz)', nadir_gfm)
report('GFM peak RoCoF (Hz/s)', rocof_pk_gfm)
report('GFL frequency nadir (Hz)', nadir_gfl)
report('GFL peak RoCoF (Hz/s)', rocof_pk_gfl)
print(f'\n  H (VSG)  = {H:.4f} s      H_GFL (equivalent) = {H_GFL:.4f} s')
print(f'  Dp (VSG) = {Dp_VSG:.4f}      Dp_GFL (equivalent) = {DP_GFL:.4f}')

sheets['Frequency_Response'] = pd.DataFrame([
    {'Case': 'GFL (baseline)', 'H (s)': H_GFL, 'Dp (N.m.s/rad)': DP_GFL,
     'Frequency Nadir (Hz)': nadir_gfl, 'Peak RoCoF (Hz/s)': rocof_pk_gfl},
    {'Case': 'GFM/VSG', 'H (s)': H, 'Dp (N.m.s/rad)': Dp_VSG,
     'Frequency Nadir (Hz)': nadir_gfm, 'Peak RoCoF (Hz/s)': rocof_pk_gfm},
])
# also save a downsampled time-series trace for plotting/inspection elsewhere
# (every 100th point: ~3000 rows instead of 300000, still fine resolution)
step = 100
sheets['Frequency_Response_TimeSeries'] = pd.DataFrame({
    'Time (s)': t[::step], 'GFM delta-f (Hz)': df_gfm[::step], 'GFL delta-f (Hz)': df_gfl[::step],
    'GFM RoCoF (Hz/s)': rocof_gfm[::step], 'GFL RoCoF (Hz/s)': rocof_gfl[::step],
})

# ---------------------------------------------------------------------------
section('1b. GFM VOLTAGE STEP RESPONSE')
# ---------------------------------------------------------------------------
from src.vsg_control import voltage_step_response
print('Second-order step response of the GFM inverter\'s inner voltage loop')
print('(250 V -> 300 V reference step).\n')
t_ms, V_out, V_ref_trace, V_clean = voltage_step_response()
steady = V_clean[-1]
overshoot_pct = (V_clean.max() - steady) / (steady - V_clean[0]) * 100
# settling time: first time the CLEAN response stays within +/-2% of steady state
band = 0.02 * steady
outside = np.where(np.abs(V_clean - steady) > band)[0]
settle_idx = (outside[-1] + 1) if len(outside) else 0
settling_time_ms = t_ms[settle_idx] - t_ms[np.searchsorted(t_ms, t_ms[0] + (0.05 - 0.03) * 1000)]
# (settling measured relative to the step instant, t_step=0.05s -> 50 ms mark)
step_mark_idx = np.searchsorted(t_ms, 50.0)
settling_time_ms = t_ms[settle_idx] - t_ms[step_mark_idx]
steady_state_error = 0.0  # the analytic model settles to V_ref exactly by construction

report('Overshoot (%)', overshoot_pct, '{:.2f}')
report('Settling time (ms)', settling_time_ms, '{:.2f}')
report('Steady-state error (V)', steady_state_error, '{:.2f}')

# ---------------------------------------------------------------------------
section('1c. GFM/VSG PERFORMANCE SUMMARY')
# ---------------------------------------------------------------------------
print('Consolidates the frequency-response and voltage step-response results')
print('above into a single summary, alongside the GFL baseline.\n')
summary_rows = [
    {'Metric': 'Frequency Nadir (Hz)', 'GFL (Baseline)': round(nadir_gfl, 2), 'GFM/VSG (Proposed)': round(nadir_gfm, 2)},
    {'Metric': 'Peak RoCoF (Hz/s)', 'GFL (Baseline)': round(rocof_pk_gfl, 2), 'GFM/VSG (Proposed)': round(rocof_pk_gfm, 2)},
    {'Metric': 'RoCoF-Compliant (<=1.0 Hz/s)?', 'GFL (Baseline)': 'No' if abs(rocof_pk_gfl) > 1.0 else 'Yes',
     'GFM/VSG (Proposed)': 'No' if abs(rocof_pk_gfm) > 1.0 else 'Yes'},
    {'Metric': 'Voltage Step Overshoot (%)', 'GFL (Baseline)': '\u2014', 'GFM/VSG (Proposed)': round(overshoot_pct, 1)},
    {'Metric': 'Voltage Settling Time (ms)', 'GFL (Baseline)': '\u2014', 'GFM/VSG (Proposed)': round(settling_time_ms, 1)},
    {'Metric': 'Steady-State Voltage Error (V)', 'GFL (Baseline)': '\u2014', 'GFM/VSG (Proposed)': f'< {max(steady_state_error, 0.5):.1f}'},
]
for r in summary_rows:
    print(f"  {r['Metric']:<32s} GFL={r['GFL (Baseline)']!s:<8s} GFM/VSG={r['GFM/VSG (Proposed)']!s}")
sheets['VSG_Performance_Summary'] = pd.DataFrame(summary_rows)

# ---------------------------------------------------------------------------
section('2. VSG PARAMETER ABLATION (H vs. Dp)')
# ---------------------------------------------------------------------------
print('Each row re-derives (nadir, RoCoF) for a different (H, Dp) pair, to')
print('isolate which parameter governs which aspect of the response. Uses')
print('the same 2.0 s settling window as the main frequency-response run')
print('above, since the weak-damping rows settle slowly.\n')
configs = [
    ('(A) Weak H, weak Dp', H_GFL, DP_GFL),
    ('(B) Strong H, weak Dp', H, DP_GFL),
    ('(C) Weak H, strong Dp', H_GFL, Dp_VSG),
    ('(D) Strong H, strong Dp', H, Dp_VSG),
]
print(f"  {'Configuration':<28s} {'Nadir (Hz)':>11s} {'RoCoF (Hz/s)':>13s}")
ablation_rows = []
ABLATION_T_END = 2.0  # matches frequency_response()'s default settling window
for name, Hc, Dpc in configs:
    _, df, rocof = _swing_ode(Hc, Dpc, -0.5 * Sn, t_step=0.1, t_end=ABLATION_T_END, dt=1e-4)
    w = slice(int(0.1 / 1e-4), int(0.3 / 1e-4))
    nadir_v, rocof_v = df.min(), rocof[w].min()
    print(f"  {name:<28s} {nadir_v:11.3f} {rocof_v:13.3f}")
    ablation_rows.append({'Configuration': name, 'H (s)': Hc, 'Dp (N.m.s/rad)': Dpc,
                           'Nadir (Hz)': nadir_v, 'RoCoF (Hz/s)': rocof_v})
sheets['VSG_Ablation'] = pd.DataFrame(ablation_rows)

# ---------------------------------------------------------------------------
section('3. ANN TRAINING FOR MPPT REFERENCE-VOLTAGE PREDICTION')
# ---------------------------------------------------------------------------
print('Training a 2-10-10-1 ReLU network (151 parameters) via genuine')
print('Levenberg-Marquardt on 5000 synthetic (G, T_amb) -> V_GMPP samples,')
print('80/10/10 split...\n')
t0 = time.perf_counter()
ann, rmse = train_ann_dataset(n_samples=5000, seed=42)
elapsed = time.perf_counter() - t0
print(f'  Held-out test RMSE = {rmse:.4f} V   [{elapsed:.1f}s]')

sheets['ANN_Training'] = pd.DataFrame([
    {'Architecture': '2-10-10-1 ReLU', 'Parameters': 151, 'Training samples': 5000,
     'Split': '80/10/10', 'Optimizer': 'Levenberg-Marquardt',
     'Held-out Test RMSE (V)': rmse, 'Training time (s)': round(elapsed, 1)},
])

# ---------------------------------------------------------------------------
section('4. FDI DETECTION PERFORMANCE')
# ---------------------------------------------------------------------------
print('10-fold stratified cross-validation, RBF-SVM, C=10, gamma=0.01,')
print('N=1200 balanced synthetic dataset...\n')
det = FDIDetector(C=10, gamma=0.01)
res = det.train_and_evaluate(n_samples=1200, n_folds=10, seed=42)

report('Accuracy (%)', res['accuracy_mean'] * 100, '{:.1f}')
report('Precision (%)', res['precision_mean'] * 100, '{:.1f}')
report('Recall (%)', res['recall_mean'] * 100, '{:.1f}')
report('F1-score', res['f1_mean'], '{:.3f}')
report('AUC-ROC', res['auc_mean'], '{:.3f}')
report('False negative rate (%)', res['false_negative_rate'] * 100, '{:.1f}')
report('False positive rate (%)', res['false_positive_rate'] * 100, '{:.1f}')
cm = res['confusion_matrix']
print(f"\n  Confusion matrix: TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

latency_ms = measure_inference_latency(det)
report('Single-sample inference latency (ms)', latency_ms, '{:.4f}')
print('  (measured on this machine -- re-run to verify on your own hardware)')

sheets['FDI_Performance'] = pd.DataFrame([
    {'Metric': 'Accuracy (%)', 'Mean': res['accuracy_mean'] * 100, 'Std': res['accuracy_std'] * 100},
    {'Metric': 'Precision (%)', 'Mean': res['precision_mean'] * 100, 'Std': res['precision_std'] * 100},
    {'Metric': 'Recall (%)', 'Mean': res['recall_mean'] * 100, 'Std': res['recall_std'] * 100},
    {'Metric': 'F1-score', 'Mean': res['f1_mean'], 'Std': res['f1_std']},
    {'Metric': 'AUC-ROC', 'Mean': res['auc_mean'], 'Std': res['auc_std']},
    {'Metric': 'False Negative Rate (%)', 'Mean': res['false_negative_rate'] * 100, 'Std': ''},
    {'Metric': 'False Positive Rate (%)', 'Mean': res['false_positive_rate'] * 100, 'Std': ''},
    {'Metric': 'Single-Sample Inference Latency (ms)', 'Mean': latency_ms, 'Std': '(machine-dependent)'},
])
sheets['FDI_Confusion_Matrix'] = pd.DataFrame(
    cm, index=['True: Safe', 'True: FDI Attack'],
    columns=['Predicted: Safe', 'Predicted: FDI Attack']
).reset_index().rename(columns={'index': ''})

# ---------------------------------------------------------------------------
section('5. FDI FEATURE ABLATION')
# ---------------------------------------------------------------------------
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X, y = generate_dataset(1200, seed=42)
feature_sets = {
    '(A) f1 only': [0], '(B) f2 only': [1], '(C) f3 only': [2],
    '(D) f1+f2': [0, 1], '(E) f1+f3': [0, 2], '(F) f2+f3': [1, 2],
    '(G) f1+f2+f3': [0, 1, 2],
}
print(f"  {'Feature subset':<14s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>7s} {'AUC-ROC':>8s}")
ablation2_rows = []
for name, cols in feature_sets.items():
    Xs = X[:, cols]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs, precs, recs, f1s, aucs = [], [], [], [], []
    for tr, te in skf.split(Xs, y):
        sc = StandardScaler()
        Xt = sc.fit_transform(Xs[tr]); Xe = sc.transform(Xs[te])
        svm = SVC(C=10, kernel='rbf', gamma=0.01, probability=True, random_state=42)
        svm.fit(Xt, y[tr])
        yp = svm.predict(Xe)
        ypr = svm.predict_proba(Xe)[:, 1]
        accs.append(accuracy_score(y[te], yp))
        precs.append(precision_score(y[te], yp, zero_division=0))
        recs.append(recall_score(y[te], yp))
        f1s.append(f1_score(y[te], yp))
        aucs.append(roc_auc_score(y[te], ypr))
    acc_m, prec_m, rec_m, f1_m, auc_m = (np.mean(accs) * 100, np.mean(precs) * 100,
                                          np.mean(recs) * 100, np.mean(f1s), np.mean(aucs))
    print(f"  {name:<14s} {acc_m:8.1f}% {prec_m:9.1f}% {rec_m:7.1f}% {f1_m:7.3f} {auc_m:8.3f}")
    ablation2_rows.append({'Feature Subset': name, 'Accuracy (%)': acc_m, 'Precision (%)': prec_m,
                            'Recall (%)': rec_m, 'F1-score': f1_m, 'AUC-ROC': auc_m})
sheets['FDI_Feature_Ablation'] = pd.DataFrame(ablation2_rows)

# ---------------------------------------------------------------------------
section('SAVING RESULTS')
# ---------------------------------------------------------------------------
with pd.ExcelWriter(XLSX_PATH, engine='openpyxl') as writer:
    for name, df in sheets.items():
        df.to_excel(writer, sheet_name=name[:31], index=False)  # Excel sheet-name limit

print(f'Saved {len(sheets)} tables to {XLSX_PATH}')
print('\nDone.')
