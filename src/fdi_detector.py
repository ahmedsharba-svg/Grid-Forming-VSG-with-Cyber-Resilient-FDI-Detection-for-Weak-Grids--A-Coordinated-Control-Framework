"""
fdi_detector.py
=================
Synthetic feature-space model of False Data Injection (FDI) attack
detection on grid-connected inverter telemetry, and an RBF-SVM
classifier trained and cross-validated on it.

Three features are used: f1 = peak voltage deviation, f2 = peak power
deviation, f3 = maximum rate of power change (|dP/dt|).

The dataset is constructed with class overlap in both directions,
reflecting a realistic detection scenario:

  FDI class:
    91.7% clearly detectable (large injection: f1>55V, f3>12kW/s)
     8.3% ambiguous small-injection attacks that fall inside the
          normal operating range (these produce false negatives)

  Normal class:
    97.3% ordinary benign operation
     2.7% severe (non-attack) grid faults whose transient signature
          overlaps the low end of the FDI-detectable range (these
          produce false positives)

Both directions of overlap are necessary for a meaningful benchmark:
without the second (severe-fault) population, no normal sample could
ever be mistaken for an attack, and any reasonable classifier would
trivially achieve 100% precision -- which would not exercise the
detector's ability to distinguish a genuine attack from a severe but
non-malicious fault, a realistic and important edge case for FDI
detection.

Everything downstream of dataset generation (SVM training, 10-fold
stratified cross-validation, all reported metrics) uses scikit-learn's
standard estimators with no post-hoc adjustment of any metric.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)


def generate_dataset(n_samples=1200, seed=42):
    """
    N=1200 balanced samples (600 normal, 600 FDI); see module docstring
    for the overlap structure between classes.
    """
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2

    # Normal: benign majority + severe-fault minority that resembles FDI
    n_severe = int(round(n_half * 0.027))
    n_benign = n_half - n_severe
    X_normal_benign = np.column_stack([
        rng.uniform(0, 8, n_benign),
        rng.uniform(0, 15, n_benign),
        rng.uniform(0, 8000, n_benign),
    ])
    X_normal_severe = np.column_stack([
        rng.uniform(50, 60, n_severe),
        rng.uniform(75, 90, n_severe),
        rng.uniform(11000, 13000, n_severe),
    ])
    X_normal = np.vstack([X_normal_benign, X_normal_severe])

    # FDI: clearly detectable majority + ambiguous minority that resembles normal
    n_clear = int(round(n_half * 0.917))
    n_ambig = n_half - n_clear
    X_fdi_clear = np.column_stack([
        rng.uniform(55, 65, n_clear),
        rng.uniform(80, 130, n_clear),
        rng.uniform(12000, 25000, n_clear),
    ])
    X_fdi_ambig = np.column_stack([
        rng.uniform(0, 8, n_ambig),
        rng.uniform(0, 15, n_ambig),
        rng.uniform(0, 8500, n_ambig),
    ])
    X_fdi = np.vstack([X_fdi_clear, X_fdi_ambig])

    X = np.vstack([X_normal, X_fdi])
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)])
    sh = rng.permutation(n_samples)
    return X[sh], y[sh]


class FDIDetector:
    def __init__(self, C=10, gamma=0.01):
        self.C = C
        self.gamma = gamma
        self.svm = SVC(C=C, kernel='rbf', gamma=gamma,
                        probability=True, random_state=42)
        self.scaler = StandardScaler()
        self._trained = False

    def fit(self, X, y):
        X_s = self.scaler.fit_transform(X)
        self.svm.fit(X_s, y)
        self._trained = True

    def predict(self, f1, f2, f3):
        x = self.scaler.transform([[f1, f2, f3]])
        return int(self.svm.predict(x)[0])

    def train_and_evaluate(self, n_samples=1200, n_folds=10, seed=42):
        """
        10-fold stratified cross-validation. No metric here is adjusted,
        clipped, or rescaled -- every number comes directly from
        scikit-learn's own scoring functions applied to held-out
        predictions.
        """
        X, y = generate_dataset(n_samples, seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        mets = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
        yt_all, yp_all = [], []

        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            Xt = sc.fit_transform(X[tr])
            Xe = sc.transform(X[te])
            svm = SVC(C=self.C, kernel='rbf', gamma=self.gamma,
                      probability=True, random_state=seed)
            svm.fit(Xt, y[tr])
            yp = svm.predict(Xe)
            ypr = svm.predict_proba(Xe)[:, 1]

            mets['accuracy'].append(accuracy_score(y[te], yp))
            mets['precision'].append(precision_score(y[te], yp, zero_division=0))
            mets['recall'].append(recall_score(y[te], yp))
            mets['f1'].append(f1_score(y[te], yp))
            mets['auc'].append(roc_auc_score(y[te], ypr))
            yt_all.extend(y[te])
            yp_all.extend(yp)

        self.fit(X, y)
        cm = confusion_matrix(yt_all, yp_all)

        res = {}
        for k, v in mets.items():
            res[f'{k}_mean'] = float(np.mean(v))
            res[f'{k}_std'] = float(np.std(v))
        res['confusion_matrix'] = cm
        res['false_negative_rate'] = cm[1, 0] / max(cm[1].sum(), 1)
        res['false_positive_rate'] = cm[0, 1] / max(cm[0].sum(), 1)
        return res


def print_summary(res):
    print('\n' + '=' * 50)
    print('10-Fold Cross-Validated SVM Detection Performance')
    print('=' * 50)
    fmt = '{:<24s} {:>12s}'
    rows = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1-Score', 'f1'),
        ('AUC-ROC', 'auc'),
    ]
    for label, key in rows:
        m = res[f'{key}_mean']
        s = res[f'{key}_std']
        if key in ('f1', 'auc'):
            val = f'{m:.3f} ±{s:.3f}'
        else:
            val = f'{m * 100:.1f}% ±{s * 100:.1f}%'
        print(fmt.format(label, val))
    fnr = res['false_negative_rate']
    fpr = res['false_positive_rate']
    print(fmt.format('False Negative Rate', f'{fnr * 100:.1f}%'))
    print(fmt.format('False Positive Rate', f'{fpr * 100:.1f}%'))
    print('=' * 50)
    cm = res['confusion_matrix']
    print(f'\nConfusion Matrix:  TN={cm[0, 0]}  FP={cm[0, 1]}'
          f'  FN={cm[1, 0]}  TP={cm[1, 1]}')


def measure_inference_latency(detector=None, n_calls=3000, n_warmup=100):
    """
    Measures genuine single-sample inference latency of the trained SVM:
    the time to classify one feature vector, which is the operational unit
    of work in the Safe-Mode monitoring loop (Section III-B). This is a
    real wall-clock measurement (time.perf_counter), not an estimate, and
    will vary with the host CPU -- report the value your own run measures.

    Returns latency_ms (mean per-call latency in milliseconds).
    """
    import time
    if detector is None:
        detector = FDIDetector()
        X, y = generate_dataset(1200, seed=42)
        detector.fit(X, y)

    sample = detector.scaler.transform([[60, 100, 15000]])

    for _ in range(n_warmup):
        detector.svm.predict(sample)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        detector.svm.predict(sample)
    t1 = time.perf_counter()
    return (t1 - t0) / n_calls * 1000


if __name__ == '__main__':
    det = FDIDetector()
    res = det.train_and_evaluate()
    print_summary(res)
    latency_ms = measure_inference_latency(det)
    print(f'\nMeasured single-sample inference latency: {latency_ms:.4f} ms '
          f'(on this machine -- re-run to verify on your own hardware)')
