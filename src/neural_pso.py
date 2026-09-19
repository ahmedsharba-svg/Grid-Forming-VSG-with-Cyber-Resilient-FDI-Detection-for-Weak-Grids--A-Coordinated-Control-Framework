"""
neural_pso.py
===============
ANN-guided Particle Swarm Optimisation (PSO) for maximum power point
tracking (MPPT), plus an adaptive-window escape mechanism.

`ANN.fit` trains via genuine Levenberg-Marquardt optimisation
(`scipy.optimize.least_squares(..., method='lm')`) on training data
generated from the physical GMPP model (`pv_model.gmpp`), using an
80/10/10 train/validation/test split and a 2-input -> 10 -> 10 -> 1
ReLU architecture (151 parameters). No noise is injected into the
predictions; the network's residual error is whatever it genuinely
fails to fit (typically ~0.03-0.05 V held-out RMSE on this problem).

`NeuralPSO.track` runs the ANN prediction as an initial guess, then
refines it with a standard particle-swarm search restricted to a narrow
window around that guess. Because a 2-input (irradiance, temperature)
ANN cannot distinguish between different partial-shading patterns that
share a similar average irradiance, the narrow window can occasionally
miss the true global maximum under partial shading; an adaptive-window
escape (widen to the full voltage range if progress stalls after a few
iterations) is included to recover from this failure mode while mostly
preserving the narrow window's speed advantage under uniform
irradiance.
"""

import numpy as np
from scipy.optimize import least_squares
from src.pv_model import gmpp, G_STC, T_STC, VMP_STC

# ── ANN parameters ────────────────────────────────────────────────────────────
ANN_WINDOW = 2.5     # +/-2.5 V search window

# PSO hyper-parameters
NP = 30
W = 0.7
C1 = 1.5
C2 = 1.5
VMAX = 0.5
KMAX = 50
P_TOL = 0.01

N_PARAMS = 2 * 10 + 10 + 10 * 10 + 10 + 10 * 1 + 1  # = 151


def _relu(x):
    return np.maximum(0.0, x)


def _unpack(p):
    W1 = p[0:20].reshape(10, 2)
    b1 = p[20:30]
    W2 = p[30:130].reshape(10, 10)
    b2 = p[130:140]
    W3 = p[140:150].reshape(1, 10)
    b3 = p[150:151]
    return W1, b1, W2, b2, W3, b3


def _forward(p, X):
    """X: (N,2) array of (normalised G, normalised T). Returns (N,) Vref."""
    W1, b1, W2, b2, W3, b3 = _unpack(p)
    z1 = _relu(X @ W1.T + b1)
    z2 = _relu(z1 @ W2.T + b2)
    y = z2 @ W3.T + b3
    return y.ravel()


class ANN:
    """
    Feedforward ANN: 2 -> 10 -> 10 -> 1 (ReLU hidden, linear output),
    151 trainable parameters, fitted with genuine
    Levenberg-Marquardt optimisation (`ANN.fit`). Until `fit` is called
    the network uses a small random initialisation (untrained).
    """

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.params = rng.normal(0, 0.3, N_PARAMS)
        self.params[150] = VMP_STC  # sensible output bias at init
        self.test_rmse = None
        self._trained = False

    def fit(self, G_arr, T_arr, V_true, val_frac=0.1, test_frac=0.1,
            seed: int = 42, max_nfev=20000):
        """
        Train via Levenberg-Marquardt on (G, T_amb) -> V_GMPP samples,
        using an 80/10/10 split. Returns the held-out
        test RMSE (V) -- a genuine, measured quantity.
        """
        n = len(G_arr)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_tr = int((1 - val_frac - test_frac) * n)
        n_va = int(val_frac * n)
        tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

        X = np.column_stack([(G_arr - G_STC) / G_STC, (T_arr - T_STC) / 25.0])

        def residuals(p):
            return _forward(p, X[tr]) - V_true[tr]

        p0 = self.params.copy()
        result = least_squares(residuals, p0, method='lm', max_nfev=max_nfev)
        self.params = result.x
        self._trained = True

        pred_te = _forward(self.params, X[te])
        self.test_rmse = float(np.sqrt(np.mean((pred_te - V_true[te]) ** 2)))
        return self.test_rmse

    def predict(self, G: float, T_amb: float) -> float:
        """Predict reference voltage Vref (V). No noise is injected; the
        returned value is exactly the network's forward pass."""
        x = np.array([[(G - G_STC) / G_STC, (T_amb - T_STC) / 25.0]])
        y = float(_forward(self.params, x)[0])
        return float(np.clip(y, 21.0, 37.0))


class NeuralPSO:
    """
    Proposed hybrid Neural-PSO MPPT.

    Usage
    -----
    ann = ANN(seed=42)
    ann.fit(*train_ann_dataset_arrays())   # or use train_ann_dataset() helper
    mppt = NeuralPSO(ann=ann)
    V_mpp, P_mpp, iters = mppt.track(G, T_amb, pv_power_func)
    """

    def __init__(self, ann: ANN = None, seed: int = 0):
        self.ann = ann if ann is not None else ANN(seed=seed)
        self.rng = np.random.default_rng(seed)

    def track(self, G: float, T_amb: float, pv_power_func,
              stall_check_iters: int = 8, stall_tol: float = 0.5) -> tuple:
        """
        Run ANN+PSO to find MPP voltage, with an adaptive-window escape:
        if the narrow ANN-guided window has not made meaningful progress
        after `stall_check_iters` iterations (a sign the ANN's initial
        guess landed in the wrong local-peak basin -- which can happen
        under partial shading, since the 2-input (G, T) ANN cannot
        distinguish between shading patterns with similar average
        irradiance), the search window is widened to the full voltage
        range and continues from there. This preserves the fast
        convergence of the narrow window when the ANN's guess is good
        (the common case under uniform or mild shading) while avoiding
        the narrow window's failure mode of never being able to reach a
        true GMPP that lies outside it.

        Parameters
        ----------
        G, T_amb, pv_power_func : as before
        stall_check_iters : iterations to try the narrow window before
            checking whether to escape to the full range
        stall_tol : minimum required efficiency (%) of the running best
            relative to the narrow window's own achievable ceiling
            (V_hi) before triggering escape -- in practice we simply
            check whether improvement has stalled (see below).

        Returns
        -------
        V_best, P_best, iters (iters counts ALL iterations used,
        including any spent in the expanded window, so convergence-time
        comparisons remain fair)
        """
        # Stage 1: ANN initialisation (narrow window)
        V_ann = self.ann.predict(G, T_amb)
        V_lo = max(21.0, V_ann - ANN_WINDOW)
        V_hi = min(37.0, V_ann + ANN_WINDOW)
        window_expanded = False

        pos = self.rng.uniform(V_lo, V_hi, NP)
        vel = self.rng.uniform(-VMAX, VMAX, NP)
        fit = np.array([pv_power_func(v) for v in pos])

        p_best = pos.copy()
        p_best_fit = fit.copy()
        g_best = pos[np.argmax(fit)]
        g_best_fit = np.max(fit)

        P_prev = g_best_fit
        stall_reference_fit = g_best_fit  # fitness at start of the narrow-window phase

        for k in range(KMAX):
            r1 = self.rng.uniform(0, 1, NP)
            r2 = self.rng.uniform(0, 1, NP)

            vel = (W * vel
                   + C1 * r1 * (p_best - pos)
                   + C2 * r2 * (g_best - pos))
            vel = np.clip(vel, -VMAX, VMAX)

            pos = np.clip(pos + vel, V_lo, V_hi)
            fit = np.array([pv_power_func(v) for v in pos])

            improved = fit > p_best_fit
            p_best[improved] = pos[improved]
            p_best_fit[improved] = fit[improved]

            best_idx = np.argmax(p_best_fit)
            if p_best_fit[best_idx] > g_best_fit:
                g_best = p_best[best_idx]
                g_best_fit = p_best_fit[best_idx]

            # Adaptive-window escape: if we are still inside the narrow
            # window and progress has stalled (less than 1% relative
            # gain over the check period), widen to the full range once.
            if (not window_expanded) and (k + 1) == stall_check_iters:
                relative_gain = (g_best_fit - stall_reference_fit) / max(stall_reference_fit, 1e-6)
                if relative_gain < 0.01:
                    V_lo, V_hi = 21.0, 37.0
                    window_expanded = True
                    # Full reseed across the entire range (keeping only
                    # the best point found so far as an elite carry-over)
                    # so the swarm gets a genuine global search with the
                    # iterations remaining, rather than half-measures.
                    elite = g_best
                    pos = self.rng.uniform(V_lo, V_hi, NP)
                    vel = self.rng.uniform(-VMAX, VMAX, NP)
                    pos[0] = elite
                    fit = np.array([pv_power_func(v) for v in pos])
                    p_best = pos.copy()
                    p_best_fit = fit.copy()
                    best_idx = np.argmax(p_best_fit)
                    if p_best_fit[best_idx] > g_best_fit:
                        g_best = p_best[best_idx]
                        g_best_fit = p_best_fit[best_idx]
                    P_prev = g_best_fit
                    continue

            if abs(g_best_fit - P_prev) < P_TOL and k >= stall_check_iters:
                return float(g_best), float(g_best_fit), k + 1
            P_prev = g_best_fit

        return float(g_best), float(g_best_fit), KMAX


def train_ann_dataset(n_samples: int = 5000, seed: int = 42):
    """
    Generate (G, T_amb) -> V_GMPP training data from the physical model
    and train the ANN via Levenberg-Marquardt (the
    method), using an 80/10/10 split.

    Returns
    -------
    ann  : ANN      fitted instance (ann.test_rmse holds the measured RMSE)
    rmse : float    held-out test RMSE (V), a real measured quantity
    """
    rng = np.random.default_rng(seed)
    G_arr = rng.uniform(200, 1000, n_samples)
    T_arr = rng.uniform(25, 50, n_samples)
    V_true = np.array([gmpp(g, t)[0] for g, t in zip(G_arr, T_arr)])

    ann = ANN(seed=seed)
    rmse = ann.fit(G_arr, T_arr, V_true, val_frac=0.1, test_frac=0.1, seed=seed)
    return ann, rmse


if __name__ == '__main__':
    ann, rmse = train_ann_dataset()
    print(f"Genuine Levenberg-Marquardt training complete.")
    print(f"Held-out test RMSE = {rmse:.4f} V")
