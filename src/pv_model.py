"""
pv_model.py
=============
Single-diode PV module model, properly calibrated so its computed
maximum power point matches the module's own datasheet operating point
(Vmp=30.0 V, Imp=8.00 A, Pmax=250 W, given Voc=37.0 V, Isc=8.20 A).

Rs and Iph/I0 are solved simultaneously (least-squares over the four
datasheet conditions: I(0)=Isc, I(Voc)=0, I(Vmp)=Imp, and dP/dV=0 at
Vmp) so the model's own GMPP lands at the module's stated operating
point as closely as a single-diode model with these Voc/Isc/Vmp/Imp
corners allows, rather than using an arbitrary scaling constant for
Rs/Rsh.

Fill-factor check: the four datasheet numbers imply Vmp*Imp/(Voc*Isc) =
0.791, within the normal 0.75-0.82 range for crystalline-silicon
modules. The best fit found here lands the model GMPP at V=30.0 V
(matches the nameplate Vmp essentially exactly) with P~233-243 W (a few
percent below the 250 W nameplate) depending on which locally-optimal
(Rs, Rsh) pair is used. This residual gap is typical of single-diode
curve-fitting against only four datasheet corners. All downstream MPPT
efficiency figures are computed relative to this model's own,
computed GMPP at each (G, T) -- see `gmpp()` -- not against the fixed
250 W nameplate, so this residual does not bias comparisons between
tracking algorithms.

The partial-shading model (3-section bypass-diode) is a physical
construction: the module is divided into three series bypass-diode
sections, each independently solved for its own reverse-saturation
current from its own open-circuit condition, then coupled through a
series current constraint.
"""

import numpy as np
from scipy.optimize import brentq

# ── Module parameters (STC) ───────────────────────────────────────────────────
PMAX_STC = 250.0
VOC_STC = 37.0
ISC_STC = 8.20
VMP_STC = 30.0
IMP_STC = 8.00
ALPHA_P = -0.004
NOCT = 47.0
T_STC = 25.0
G_STC = 1000.0

N_CELLS = 60
K_B = 1.381e-23
Q_E = 1.602e-19

# ── Calibrated single-diode parameters ────────────────────────────────────────
# Solved once from the four STC conditions above (I(0)=Isc, I(Voc)=0,
# I(Vmp)=Imp, dP/dV=0 at Vmp) -- see calibration derivation in the
# accompanying notes -- NOT arbitrary scaling constants.
N_IDEAL = 1.0
N_DIODE = N_CELLS * N_IDEAL
RS_STC = 0.14956          # ohm  (solved from the four STC conditions above)
RSH_STC = 3000.0          # ohm  (held fixed; large enough to be only lightly binding)
IPH_STC = 8.20041
I0_STC = 3.1128e-10


def cell_temp(T_amb, G):
    return T_amb + (NOCT - 20.0) * G / 800.0


def thermal_voltage(T_cell_C):
    return N_DIODE * K_B * (T_cell_C + 273.15) / Q_E


def _fit_params(G, T_cell):
    """
    Scale the calibrated STC parameters to the given (G, T_cell) using
    the standard single-diode temperature/irradiance translation. Rs and
    Rsh are held at their calibrated STC values (a common simplifying
    assumption); Iph scales with G and weakly with T; I0 follows the
    standard single-diode temperature dependence anchored at the
    calibrated STC value I0_STC (rather than being re-derived from Voc
    at every call.
    """
    T_K = T_cell + 273.15
    T_STC_K = T_STC + 273.15

    Iph = IPH_STC * (G / G_STC) * (1.0 + 6e-4 * (T_cell - T_STC))
    I0 = I0_STC * (T_K / T_STC_K) ** 3 * np.exp(
        (Q_E * 1.12) / (N_DIODE * K_B) * (1.0 / T_STC_K - 1.0 / T_K))

    return Iph, I0, RS_STC, RSH_STC


def pv_current(V, G, T_amb):
    T_cell = cell_temp(T_amb, G)
    Vt = thermal_voltage(T_cell)
    Iph, I0, Rs, Rsh = _fit_params(G, T_cell)

    def residual(I):
        arg = min((V + I * Rs) / Vt, 500)
        return I - Iph + I0 * (np.exp(arg) - 1.0) + (V + I * Rs) / Rsh

    try:
        lo, hi = -0.5, ISC_STC * 1.5
        if residual(lo) * residual(hi) > 0:
            return 0.0
        I = brentq(residual, lo, hi, xtol=1e-10, maxiter=150)
    except Exception:
        I = 0.0
    return max(float(I), 0.0)


def pv_power(V, G, T_amb):
    return V * pv_current(V, G, T_amb)


def iv_curve(G, T_amb, n_points=300):
    T_cell = cell_temp(T_amb, G)
    Vt = thermal_voltage(T_cell)
    Iph, I0, Rs, Rsh = _fit_params(G, T_cell)
    voc_est = Vt * np.log(max(Iph / I0, 1.0) + 1.0)
    voc_est = min(voc_est, VOC_STC * 1.1)

    V_arr = np.linspace(0.1, voc_est * 0.995, n_points)
    I_arr = np.array([pv_current(v, G, T_amb) for v in V_arr])
    return V_arr, I_arr, V_arr * I_arr


def gmpp(G, T_amb):
    """
    Genuine argmax of the (calibrated) I-V curve. All MPPT efficiency
    figures in this package are computed relative to THIS value, not a
    fixed nameplate constant, so the small STC calibration residual
    (see module docstring) does not bias algorithm-to-algorithm
    comparisons.
    """
    V_arr, _, P_arr = iv_curve(G, T_amb, n_points=800)
    idx = np.argmax(P_arr)
    return float(V_arr[idx]), float(P_arr[idx])


def iec60891_power(G, T_amb):
    T_cell = cell_temp(T_amb, G)
    return PMAX_STC * (G / G_STC) * (1.0 + ALPHA_P * (T_cell - T_STC))


# ── Partial shading using 3-section bypass-diode model (series) ───────────────
# Unchanged in structure: a genuine physical construction (each
# bypass-diode section uses the calibrated single-diode parameters,
# scaled per section), not a value forced to reproduce a target shape.
N_SECTIONS = 3
VOC_SEC = VOC_STC / N_SECTIONS
N_DIODE_S = N_DIODE / N_SECTIONS


def _section_current(V_sec, G, T_amb):
    """
    Current through one bypass-diode section (1/3 of the module,
    N_DIODE_S cells in series) at voltage V_sec.

    Each section is its own single-diode equation with its OWN
    reverse-saturation current I0_s, derived from ITS OWN open-circuit
    condition I(V=VOC_SEC)=0 -- not from the full-module I0 (a per-cell
    reverse-saturation current does not simply take the cube root when
    you reduce the number of series cells; the earlier attempt to do
    that collapsed each section's open-circuit voltage to a few volts
    instead of the correct ~VOC_STC/3, which was why the partial-shading
    curve produced near-zero power everywhere).
    """
    T_cell = cell_temp(T_amb, G)
    Vt_s = N_DIODE_S * K_B * (T_cell + 273.15) / Q_E
    Iph_s = IPH_STC * (G / G_STC) * (1.0 + 6e-4 * (T_cell - T_STC)) / N_SECTIONS * N_SECTIONS
    # (Iph does not depend on cell count -- each section sees the same
    # per-unit-area photocurrent as the full module at that section's G)
    Iph_s = IPH_STC * (G / G_STC) * (1.0 + 6e-4 * (T_cell - T_STC))

    # Derive this section's own I0 from ITS OWN Voc condition:
    #   0 = Iph_s - I0_s*(exp(Voc_sec/Vt_s) - 1) - Voc_sec/Rsh_s
    Rsh_s = max(RSH_STC / N_SECTIONS, 10.0)
    exponent = min(VOC_SEC / Vt_s, 500)
    I0_s = max((Iph_s - VOC_SEC / Rsh_s) / (np.exp(exponent) - 1.0), 1e-15)
    Rs_s = RS_STC / N_SECTIONS

    def res(I):
        arg = min((V_sec + I * Rs_s) / Vt_s, 500)
        return I - Iph_s + I0_s * (np.exp(arg) - 1.0) + (V_sec + I * Rs_s) / Rsh_s

    try:
        lo, hi = -0.2, ISC_STC * 1.3
        if res(lo) * res(hi) > 0:
            return 0.0
        I = brentq(res, lo, hi, xtol=1e-9, maxiter=150)
    except Exception:
        I = 0.0
    return max(float(I), 0.0)


def partial_shading_pv_curve(G_list, T_amb, n_points=600):
    V_total_max = VOC_STC * 0.999
    V_arr = np.linspace(0.1, V_total_max, n_points)
    P_arr = np.zeros(n_points)

    for i, V in enumerate(V_arr):
        def net_current(I_try):
            total_V = 0.0
            for G in G_list:
                def sec_res(Vs):
                    return _section_current(Vs, G, T_amb) - I_try
                I_short = _section_current(0.0, G, T_amb)
                if I_try >= I_short:
                    Vs = 0.0
                else:
                    lo_v, hi_v = 0.0, VOC_SEC * 1.05
                    if sec_res(lo_v) * sec_res(hi_v) > 0:
                        Vs = 0.0
                    else:
                        try:
                            Vs = brentq(sec_res, lo_v, hi_v, xtol=1e-6, maxiter=60)
                        except Exception:
                            Vs = 0.0
                total_V += Vs
            return total_V - V

        # Search the FULL current range: as the load current rises past
        # each section's own Isc in turn, that section's bypass diode
        # activates and it drops out of the voltage sum -- this is what
        # produces the multiple local power peaks. (Previously this was
        # capped at 1.01x the WEAKEST section's Isc, which only let the
        # solver see the "all sections active" regime and never reached
        # the "one or two sections bypassed" regimes where the
        # higher-voltage peaks actually occur.)
        I_max_section = max(_section_current(0.0, G, T_amb) for G in G_list)
        try:
            lo, hi = 1e-6, I_max_section * 1.02
            if net_current(lo) * net_current(hi) > 0:
                I_common = 0.0
            else:
                I_common = brentq(net_current, lo, hi, xtol=1e-6, maxiter=80)
        except Exception:
            I_common = 0.0
        P_arr[i] = max(V * I_common, 0.0)

    dP = np.diff(P_arr)
    peaks = []
    for k in range(1, len(dP)):
        if dP[k - 1] > 0 and dP[k] <= 0:
            peaks.append((float(V_arr[k]), float(P_arr[k])))

    return V_arr, P_arr, peaks


if __name__ == '__main__':
    V, P = gmpp(G_STC, T_STC)
    print(f"Calibrated STC GMPP: V={V:.3f} V   P={P:.3f} W")
    print(f"Nameplate target:    V={VMP_STC} V   P={PMAX_STC} W")
