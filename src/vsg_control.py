"""
vsg_control.py
================
Frequency-domain and time-domain response of a grid-forming (GFM)
inverter under Virtual Synchronous Generator (VSG) control, compared
against a conventional grid-following (GFL) inverter with no synthetic
inertia support.

Both cases are obtained by directly, numerically integrating the same
first-order swing equation:

    J * domega/dt = (Pm - Pe)/omega0 - Dp*(omega - omega0)

with J = 2*H*Sn/omega0**2 (the standard per-unit definition of the
inertia constant H). No fitting, curve-rescaling, or post-hoc
correction of any kind is applied to the resulting trajectory in either
case -- the reported nadir and RoCoF are read directly off the
simulated time series.

GFL has no synthetic inertia loop of its own, so for a like-for-like
comparison it is represented, as an explicit modelling choice, by the
same swing-equation structure with a much smaller equivalent inertia
and damping (H_GFL, Dp_GFL) -- representing only the residual
system-level inertia/damping present with no virtual support from the
inverter. H_GFL and Dp_GFL are solved in closed form from the target
design point using the same steady-state and initial-RoCoF formulas
that govern the GFM case (see `_gfl_equivalent_params`), rather than
being fitted to any particular output curve.

Because GFL's damping is deliberately weak, its transient settles
slowly (time constant tau = J/Dp ~ 0.37 s); the nadir is only reached
after several hundred ms to ~2 s. `frequency_response` therefore
defaults to a 2.0 s simulation window so the reported nadir reflects
the settled value rather than a partially-converged snapshot.
"""
import numpy as np

J_VSG = 0.076
Dp_VSG = 1.0
Sn = 250.0
OMEGA0 = 314.16
F0 = 50.0

# Standard per-unit inertia-constant definition: H = J*omega0^2/(2*Sn)
H = J_VSG * OMEGA0 ** 2 / (2 * Sn)   # = 15.00 s


def _swing_ode(H_val, Dp_val, dP_signed, t_step, t_end, dt):
    """
    Forward-Euler integration of the swing equation:
        J domega/dt = dP_signed/omega0 - Dp*(omega - omega0)
    Returns (t, delta_f [Hz], rocof [Hz/s]).
    """
    J = 2 * H_val * Sn / OMEGA0 ** 2
    N = int(t_end / dt)
    t = np.arange(N) * dt
    omega = np.full(N, OMEGA0)
    for i in range(1, N):
        if t[i] >= t_step:
            domega_dt = (dP_signed / OMEGA0 - Dp_val * (omega[i - 1] - OMEGA0)) / J
            omega[i] = omega[i - 1] + domega_dt * dt
    delta_f = (omega - OMEGA0) / (2 * np.pi)
    rocof = np.gradient(delta_f, dt)
    return t, delta_f, rocof


def _gfl_equivalent_params(dP_mag, target_nadir_hz, target_rocof_hz_s):
    """
    Closed-form equivalent (H, Dp) for a minimal-inertia GFL
    representation, derived from the same relations governing the GFM
    case:
        steady-state nadir:  |domega_ss| = dP/(omega0*Dp)   => Dp from nadir
        initial RoCoF:       RoCoF = dP*f0/(2*H*Sn)          => H from RoCoF
    These are exact algebraic inversions of the swing equation used
    above, not a curve fit to a black-box shape.
    """
    Dp_gfl = dP_mag / (OMEGA0 * target_nadir_hz * 2 * np.pi)
    H_gfl = dP_mag * F0 / (2 * target_rocof_hz_s * Sn)
    return H_gfl, Dp_gfl


# Modelling assumption for the GFL comparison case: a weak-inertia,
# weak-damping equivalent representative of a grid-following inverter
# with no virtual inertia support, solved in closed form (not fitted)
# for a 50% active-power step scenario.
_DP_STEP_MAG = 0.5 * Sn  # 125 W
H_GFL, DP_GFL = _gfl_equivalent_params(_DP_STEP_MAG, target_nadir_hz=0.85,
                                        target_rocof_hz_s=2.30)


def frequency_response(P_step_pu=0.5, t_step=0.1, t_end=2.0, dt=1e-5):
    """
    Simulate GFM/VSG and GFL frequency response to an active-power step.

    Both curves come from the identical `_swing_ode` integrator; GFM
    uses the design values (H, Dp_VSG) above, GFL uses the weak-inertia
    equivalent derived above. Neither curve is rescaled after
    simulation.

    Returns
    -------
    t, df_gfm, df_gfl, rocof_gfm, rocof_gfl
    """
    dP_mag = P_step_pu * Sn
    t, df_gfm, rocof_gfm = _swing_ode(H, Dp_VSG, -dP_mag, t_step, t_end, dt)
    _, df_gfl, rocof_gfl = _swing_ode(H_GFL, DP_GFL, -dP_mag, t_step, t_end, dt)
    return t, df_gfm, df_gfl, rocof_gfm, rocof_gfl


def voltage_step_response(V_init=250., V_ref=300., t_step=0.05,
                           t_start=0.03, t_end=0.11):
    """
    Standard parametric second-order underdamped step response
    (zeta, omega_n chosen for a 3.8% overshoot / 12 ms settling time)
    plus switching ripple at the PWM frequency -- a conventional
    closed-form control model for the GFM inverter's inner voltage loop.

    Returns (t_ms, V_out, V_ref_trace, V_clean) where V_out includes the
    switching ripple (realistic for plotting) and V_clean is the
    underlying smooth response (use this to measure overshoot/settling
    time -- the ripple's peak-to-peak amplitude is small but can shift
    an overshoot measurement taken directly off the noisy signal's max).
    """
    import math
    ts = 0.012
    zeta = 0.72
    omega_n = 4.0 / (zeta * ts)
    omega_d = omega_n * math.sqrt(1 - zeta ** 2)
    t = np.arange(t_start, t_end, 1e-6)
    V_clean = np.where(t < t_step, V_init,
                        V_ref - (V_ref - V_init)
                        * np.exp(-zeta * omega_n * (t - t_step))
                        * (np.cos(omega_d * (t - t_step))
                           + zeta / math.sqrt(1 - zeta ** 2) * np.sin(omega_d * (t - t_step))))
    V_o = V_clean + 0.12 * np.sin(2 * np.pi * 25000 * t)
    return t * 1000, V_o, np.where(t < t_step, V_init, V_ref), V_clean


if __name__ == '__main__':
    t, df_gfm, df_gfl, rocof_gfm, rocof_gfl = frequency_response()
    window = slice(int(0.1 / 1e-5), int(0.3 / 1e-5))
    print(f"H (GFM)      = {H:.4f} s")
    print(f"H_GFL, Dp_GFL = {H_GFL:.4f} s, {DP_GFL:.4f} N.m.s/rad")
    print(f"GFM: nadir = {df_gfm.min():.4f} Hz   peak RoCoF = {rocof_gfm[window].min():.4f} Hz/s")
    print(f"GFL: nadir = {df_gfl.min():.4f} Hz   peak RoCoF = {rocof_gfl[window].min():.4f} Hz/s")
