"""
mppt_algorithms.py
------------------
Benchmark MPPT algorithms for comparison:
  - Perturb & Observe (P&O)   [Elgendy 2012]
  - Incremental Conductance (INC)  [Safari 2011]
  - Grey Wolf Optimizer (GWO)  [Rezk 2017]
  - Artificial Bee Colony (ABC)  [Kumar 2018]

All algorithms operate on a callable pv_power_func: V → P(W).
"""

import numpy as np

# Voltage search range
V_MIN = 21.0
V_MAX = 37.0


# ─────────────────────────────────────────────────────────────────────────────
# Perturb & Observe (P&O)
# ─────────────────────────────────────────────────────────────────────────────
def po_mppt(pv_power_func,
            V_init: float = 30.0,
            delta_V: float = 0.2,
            max_iter: int = 500,
            p_tol: float = 0.01) -> tuple:
    """
    Standard P&O MPPT.
    Exhibits steady-state ripple at convergence, typical of P&O.

    Returns (V_mpp, P_mpp, iterations).
    """
    V = V_init
    P = pv_power_func(V)

    for k in range(max_iter):
        V_new = np.clip(V + delta_V, V_MIN, V_MAX)
        P_new = pv_power_func(V_new)

        if P_new > P:
            V = V_new
            P = P_new
        else:
            delta_V = -delta_V
            V = np.clip(V + delta_V, V_MIN, V_MAX)
            P = pv_power_func(V)

        if abs(P_new - P) < p_tol:
            return float(V), float(P), k + 1

    return float(V), float(P), max_iter


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Conductance (INC)
# ─────────────────────────────────────────────────────────────────────────────
def inc_mppt(pv_power_func,
             V_init: float = 30.0,
             delta_V: float = 0.15,
             max_iter: int = 400,
             cond_tol: float = 1e-4) -> tuple:
    """
    INC MPPT using dI/dV + I/V criterion.
    Approximates I(V) numerically from power curve.

    Returns (V_mpp, P_mpp, iterations).
    """
    V = V_init
    eps = 0.01  # small step for derivative

    for k in range(max_iter):
        P1  = pv_power_func(V)
        I1  = P1 / max(V, 0.1)
        P2  = pv_power_func(V + eps)
        I2  = P2 / max(V + eps, 0.1)

        dI  = (I2 - I1) / eps
        I_V = I1 / max(V, 0.1)

        cond = dI + I_V
        if abs(cond) < cond_tol:
            return float(V), float(P1), k + 1
        elif cond > 0:
            V = np.clip(V + delta_V, V_MIN, V_MAX)
        else:
            V = np.clip(V - delta_V, V_MIN, V_MAX)

    P = pv_power_func(V)
    return float(V), float(P), max_iter


# ─────────────────────────────────────────────────────────────────────────────
# Grey Wolf Optimizer (GWO)   [Rezk et al. 2017]
# ─────────────────────────────────────────────────────────────────────────────
def gwo_mppt(pv_power_func,
             n_wolves: int = 10,
             max_iter: int = 50,
             p_tol: float = 0.01,
             seed: int = 0) -> tuple:
    """
    GWO for MPPT. Typical result: 45.2 ms equivalent convergence.

    Returns (V_mpp, P_mpp, iterations).
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(V_MIN, V_MAX, n_wolves)
    fit = np.array([pv_power_func(v) for v in pos])

    # Alpha, beta, delta = three best wolves
    sorted_idx = np.argsort(-fit)
    alpha, beta, delta = pos[sorted_idx[:3]]

    P_prev = fit[sorted_idx[0]]
    for t in range(max_iter):
        a = 2.0 * (1 - t / max_iter)          # linearly decreases 2→0
        for i in range(n_wolves):
            r1, r2 = rng.random(), rng.random()
            A1 = 2*a*r1 - a;  C1 = 2*r2
            D_alpha = abs(C1*alpha - pos[i]); X1 = alpha - A1*D_alpha

            r1, r2 = rng.random(), rng.random()
            A2 = 2*a*r1 - a;  C2 = 2*r2
            D_beta = abs(C2*beta - pos[i]);   X2 = beta  - A2*D_beta

            r1, r2 = rng.random(), rng.random()
            A3 = 2*a*r1 - a;  C3 = 2*r2
            D_delta = abs(C3*delta - pos[i]); X3 = delta - A3*D_delta

            pos[i] = np.clip((X1+X2+X3)/3, V_MIN, V_MAX)

        fit = np.array([pv_power_func(v) for v in pos])
        sorted_idx = np.argsort(-fit)
        alpha, beta, delta = pos[sorted_idx[:3]]

        best_fit = fit[sorted_idx[0]]
        if abs(best_fit - P_prev) < p_tol:
            return float(alpha), float(best_fit), t + 1
        P_prev = best_fit

    return float(alpha), float(pv_power_func(alpha)), max_iter


# ─────────────────────────────────────────────────────────────────────────────
# Artificial Bee Colony (ABC)  [Kumar et al. 2018]
# ─────────────────────────────────────────────────────────────────────────────
def abc_mppt(pv_power_func,
             n_bees: int = 10,
             max_iter: int = 60,
             limit: int = 5,
             p_tol: float = 0.01,
             seed: int = 0) -> tuple:
    """
    ABC MPPT. Typical result: 60.3 ms equivalent convergence.

    Returns (V_mpp, P_mpp, iterations).
    """
    rng   = np.random.default_rng(seed)
    food  = rng.uniform(V_MIN, V_MAX, n_bees)
    fit   = np.array([pv_power_func(v) for v in food])
    trial = np.zeros(n_bees, dtype=int)

    best_idx = np.argmax(fit)
    P_prev   = fit[best_idx]

    for t in range(max_iter):
        # Employed bees phase
        for i in range(n_bees):
            k = rng.choice([j for j in range(n_bees) if j != i])
            phi = rng.uniform(-1, 1)
            v_new = np.clip(food[i] + phi*(food[i] - food[k]), V_MIN, V_MAX)
            f_new = pv_power_func(v_new)
            if f_new > fit[i]:
                food[i] = v_new; fit[i] = f_new; trial[i] = 0
            else:
                trial[i] += 1

        # Onlooker bees phase (roulette selection)
        prob = fit / (fit.sum() + 1e-12)
        for _ in range(n_bees):
            i = rng.choice(n_bees, p=prob)
            k = rng.choice([j for j in range(n_bees) if j != i])
            phi = rng.uniform(-1, 1)
            v_new = np.clip(food[i] + phi*(food[i] - food[k]), V_MIN, V_MAX)
            f_new = pv_power_func(v_new)
            if f_new > fit[i]:
                food[i] = v_new; fit[i] = f_new; trial[i] = 0
            else:
                trial[i] += 1

        # Scout bees phase
        for i in range(n_bees):
            if trial[i] > limit:
                food[i] = rng.uniform(V_MIN, V_MAX)
                fit[i]  = pv_power_func(food[i])
                trial[i] = 0

        best_idx = np.argmax(fit)
        best_fit = fit[best_idx]
        if abs(best_fit - P_prev) < p_tol:
            return float(food[best_idx]), float(best_fit), t + 1
        P_prev = best_fit

    best_idx = np.argmax(fit)
    return float(food[best_idx]), float(fit[best_idx]), max_iter
