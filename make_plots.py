"""
make_plots.py
===============
Generates plots of the key simulation and detection results directly
from the src/ modules: no image here is hand-drawn; each is a fresh
plot of freshly-computed data.

Usage:
    pip install -r requirements.txt
    python3 make_plots.py

Output (written to ./results/figures/):
    frequency_response.png    -- GFM/VSG vs. GFL: delta-f(t) and RoCoF(t)
    voltage_step_response.png -- GFM inverter voltage step response
    confusion_matrix.png      -- FDI detector confusion matrix (heatmap)
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from src.vsg_control import frequency_response, voltage_step_response
from src.fdi_detector import FDIDetector

OUT_DIR = os.path.join('results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Frequency and RoCoF response: GFL vs. GFM/VSG
# ---------------------------------------------------------------------------
def plot_frequency_response():
    t, df_gfm, df_gfl, rocof_gfm, rocof_gfl = frequency_response(t_end=1.2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    ax1.plot(t, df_gfl, color='#C62828', linestyle='--', linewidth=1.8, label='GFL (baseline)')
    ax1.plot(t, df_gfm, color='#1565C0', linewidth=1.8, label='GFM/VSG')
    ax1.axhline(df_gfm.min(), color='#1565C0', linewidth=0.6, linestyle=':')
    ax1.axhline(df_gfl.min(), color='#C62828', linewidth=0.6, linestyle=':')
    ax1.annotate(f'Nadir: {df_gfm.min():.3f} Hz', xy=(0.55, df_gfm.min()),
                 fontsize=8.5, color='#1565C0', va='bottom')
    ax1.annotate(f'Nadir: {df_gfl.min():.2f} Hz', xy=(0.55, df_gfl.min()),
                 fontsize=8.5, color='#C62828', va='top')
    ax1.set_ylabel('\u0394f (Hz)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(t, rocof_gfl, color='#C62828', linestyle='--', linewidth=1.5, label='GFL RoCoF')
    ax2.plot(t, rocof_gfm, color='#1565C0', linewidth=1.5, label='GFM/VSG RoCoF')
    ax2.axhline(-1.0, color='orange', linestyle='-.', linewidth=1.2, label='1.0 Hz/s reference limit')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('RoCoF (Hz/s)')
    ax2.legend(loc='lower right', fontsize=8.5)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'frequency_response.png')
    plt.savefig(path, dpi=200, facecolor='white')
    plt.close()
    print(f'Saved {path}   (nadir_gfm={df_gfm.min():.4f} Hz, '
          f'peak_rocof_gfm={rocof_gfm[int(0.1/1e-5):int(0.3/1e-5)].min():.4f} Hz/s)')


# ---------------------------------------------------------------------------
# Voltage step response of the GFM inverter
# ---------------------------------------------------------------------------
def plot_voltage_step_response():
    t_ms, V_out, V_ref, V_clean = voltage_step_response()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    steady = V_clean[-1]
    band = 0.02 * steady
    ax.axhspan(steady - band, steady + band, color='#C8E6C9', alpha=0.6, label='\u00b12% settling band')
    ax.plot(t_ms, V_ref, color='#C62828', linestyle='--', linewidth=1.3, label='Reference Vref')
    ax.plot(t_ms, V_out, color='#1565C0', linewidth=1.8, label='Actual Vout')

    # Overshoot is measured from the clean (ripple-free) response, since the
    # switching ripple's peak can otherwise shift the noisy signal's max
    # away from the underlying control response being characterised.
    overshoot = (V_clean.max() - steady) / (steady - V_clean[0]) * 100
    ax.annotate(f'Overshoot = {overshoot:.1f}%', xy=(t_ms[np.argmax(V_out)], V_out.max()),
                xytext=(5, 5), textcoords='offset points', color='#1565C0', fontsize=9)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (V)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'voltage_step_response.png')
    plt.savefig(path, dpi=200, facecolor='white')
    plt.close()
    print(f'Saved {path}   (overshoot={overshoot:.2f}%)')


# ---------------------------------------------------------------------------
# FDI detector confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix():
    det = FDIDetector(C=10, gamma=0.01)
    res = det.train_and_evaluate(n_samples=1200, n_folds=10, seed=42)
    cm = res['confusion_matrix']
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=1)

    labels = ['Safe (0)', 'FDI Attack (1)']
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm_pct[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]*100:.1f}%)',
                    ha='center', va='center', color=color, fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=200, facecolor='white')
    plt.close()
    print(f'Saved {path}   (TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]})')


if __name__ == '__main__':
    plot_frequency_response()
    plot_voltage_step_response()
    plot_confusion_matrix()
    print(f'\nAll plots written to ./{OUT_DIR}/')
