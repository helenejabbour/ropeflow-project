# src/Data_processing/08_preprocess.py
# ─────────────────────────────────────────────────────────────
# IMU preprocessing pipeline for rope flow data (M5Stick C Plus 1.1)
# Output: one clean CSV per device, ready for PINN trajectory estimation
#
# Pipeline:
#   1. Load & split by device
#   2. Remove duplicate timestamps (M5Stick ms-clock artifact)
#   3. Normalize timestamps (ms → s, zero-based)
#   4. Resample to uniform grid (FS_TARGET Hz)
#   5. Low-pass denoise (acc + gyro)
#   6. Madgwick filter → orientation quaternions + gravity removal
#   7. Double integration → naive velocity & position (with HP drift suppression)
#   8. Cycle detection on dominant gyro axis
#   9. Save processed CSV + 3-panel validation plot
# ─────────────────────────────────────────────────────────────

import os
import math
import glob
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

# ── Paths ─────────────────────────────────────────────────────
# Script lives at: ropeflow-project/src/Data_processing/08_preprocess.py
# Go up 3 levels:  Data_processing → src → ropeflow-project
ROOT          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW      = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS_PROC  = os.path.join(ROOT, "results", "Data_processing")
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(RESULTS_PROC, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────
FS_TARGET      = 50.0    # Hz  (input ~33 Hz, upsample to 50)
LOWPASS_HZ     = 12.0    # Hz  low-pass cutoff for denoising
HP_VEL_HZ      = 0.05    # Hz  high-pass on velocity to suppress DC drift
MADGWICK_BETA  = 0.05    # Madgwick gain
G_MS2          = 9.80665 # m/s² per g

# Cycle detection — rope flow expected at 1–3 Hz
CYCLE_BAND     = (0.8, 4.0)   # Hz bandpass for cycle detection
PEAK_MIN_DIST  = 0.25         # seconds minimum between peaks


# ── Helper functions ──────────────────────────────────────────

def butter(x, cutoff, fs, btype, order=4):
    """Apply a zero-phase Butterworth filter along axis 0."""
    ny  = 0.5 * fs
    wn  = np.array(cutoff) / ny if isinstance(cutoff, (list, tuple)) else cutoff / ny
    wn  = np.clip(wn, 1e-6, 1 - 1e-6)
    b, a = signal.butter(order, wn, btype=btype)
    return signal.filtfilt(b, a, x, axis=0)


def resample(df, t_in, fs):
    """Interpolate all numeric columns onto a uniform time grid at fs Hz."""
    t0, tf = t_in[0], t_in[-1]
    n      = max(2, int(math.ceil((tf - t0) * fs)))
    t_out  = np.linspace(t0, tf, n)
    out    = {"timestamp": t_out}
    for col in df.columns:
        if col == "timestamp":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = np.interp(t_out, t_in, df[col].astype(float).values)
    return pd.DataFrame(out), t_out


def rotate_vec(quats, vecs):
    """
    Rotate body-frame vectors to world frame using quaternions.
    q = [w, x, y, z],  v_world = q ⊗ v ⊗ q*
    Uses the Rodrigues formula: v' = v + 2w(q_v × v) + 2(q_v × (q_v × v))
    """
    w  = quats[:, 0:1]
    qv = quats[:, 1:]
    return vecs + 2 * w * np.cross(qv, vecs) + 2 * np.cross(qv, np.cross(qv, vecs))


# ── Processing steps ──────────────────────────────────────────

def run_madgwick(acc_g, gyr_degs, fs):
    """
    Run Madgwick filter on the full session.
    Returns quaternions (N,4) and world-frame linear acceleration (N,3) in m/s².
    Gravity is estimated from the first 2 s (assumed near-static at session start).
    """
    acc_ms2  = acc_g    * G_MS2
    gyr_rads = gyr_degs * (np.pi / 180.0)

    mad    = Madgwick(beta=MADGWICK_BETA, sampleperiod=1.0 / fs)
    q      = np.array([1.0, 0.0, 0.0, 0.0])
    quats  = np.zeros((len(acc_ms2), 4))
    for i in range(len(acc_ms2)):
        q = mad.updateIMU(q, gyr=gyr_rads[i], acc=acc_ms2[i])
        quats[i] = q

    acc_world = rotate_vec(quats, acc_ms2)

    # Estimate gravity from first 2 s
    n_warmup = int(2.0 * fs)
    g_est    = np.mean(acc_world[:n_warmup], axis=0)
    if abs(np.linalg.norm(g_est) - G_MS2) > 1.5:   # fallback if moving at start
        g_est = np.array([0.0, 0.0, G_MS2])

    acc_linear = acc_world - g_est
    return quats, acc_linear


def integrate(acc_linear, fs):
    """
    Double-integrate linear acceleration → velocity → position.
    High-pass filter on velocity suppresses DC integration drift.
    Returns vel (N,3) and pos (N,3) in m/s and m.
    """
    dt  = 1.0 / fs
    vel = np.cumsum(acc_linear, axis=0) * dt
    vel = butter(vel, HP_VEL_HZ, fs, btype='high', order=2)
    pos = np.cumsum(vel, axis=0) * dt
    return vel, pos


def detect_cycles(gyr_degs, fs):
    """
    Detect rope-flow cycles on the dominant gyro axis (highest variance).
    Bandpass in CYCLE_BAND, then find peaks adaptively.
    Returns peak indices and the bandpassed signal.
    """
    gyr_rads = gyr_degs * (np.pi / 180.0)
    dom      = int(np.argmax(np.var(gyr_rads, axis=0)))   # dominant axis
    sig      = gyr_rads[:, dom]
    sig_bp   = butter(sig - sig.mean(), CYCLE_BAND, fs, btype='band')

    min_dist   = int(PEAK_MIN_DIST * fs)
    peaks_all, _ = signal.find_peaks(sig_bp, distance=min_dist)
    thresh     = (np.percentile(sig_bp[peaks_all], 60)
                  if len(peaks_all) > 1 else np.std(sig_bp) * 0.6)
    peaks, _   = signal.find_peaks(sig_bp, height=thresh, distance=min_dist)
    return peaks, sig_bp, dom


def save_plot(t, acc_linear, pos, peaks, gyr_bp, dom, fname, dev):
    """3-panel validation plot: acceleration PSD, cycle detection, naive position."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"{fname} — Device {dev}", fontsize=11)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    labels = ['X', 'Y', 'Z']

    # Panel 1: PSD of world-frame acceleration
    ax = axes[0]
    for i in range(3):
        f, P = signal.welch(acc_linear[:, i], fs=FS_TARGET,
                            nperseg=min(256, len(acc_linear)))
        ax.semilogy(f, P, color=colors[i], lw=0.9, label=labels[i])
    ax.axvspan(*CYCLE_BAND, alpha=0.12, color='green', label='Rope-flow band')
    ax.set_xlim(0, FS_TARGET / 2)
    ax.set_title("PSD — World-frame linear acceleration (m/s²)")
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(fontsize=8)

    # Panel 2: Cycle detection on dominant gyro axis
    ax = axes[1]
    ax.plot(t, gyr_bp, color='#e67e22', lw=0.8,
            label=f'Gyro-{labels[dom]} bandpass (rad/s)')
    if len(peaks):
        ax.plot(t[peaks], gyr_bp[peaks], 'v', color='black', ms=5,
                label=f'{len(peaks)} peaks')
    if len(peaks) > 1:
        mean_period = np.mean(np.diff(t[peaks]))
        ax.set_title(f"Cycle detection — dominant axis {labels[dom]} | "
                     f"mean period {mean_period:.2f} s "
                     f"({1/mean_period:.2f} Hz)")
    else:
        ax.set_title(f"Cycle detection — dominant axis {labels[dom]}")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)

    # Panel 3: Naive position (drifted — PINN corrects this)
    ax = axes[2]
    for i in range(3):
        ax.plot(t, pos[:, i], color=colors[i], lw=0.8, label=labels[i])
    ax.set_title("Naive position (m) — drifted input for PINN")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS_PROC, f"{fname}_dev{dev}_validation.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────

def process_file(path):
    fname = os.path.splitext(os.path.basename(path))[0]
    print(f"\nProcessing: {fname}")

    df      = pd.read_csv(path)
    ts_col  = next((c for c in df.columns if c.lower() in ('timestamp','time','t','ts')), None)
    dev_col = next((c for c in df.columns if c.lower() in ('device','dev')), None)
    if ts_col is None:
        raise RuntimeError("No timestamp column found.")
    if dev_col is None:
        df['device'] = 'all'
        dev_col = 'device'

    for dev in sorted(df[dev_col].astype(str).unique()):
        print(f"\n  Device {dev}")
        d = df[df[dev_col].astype(str) == dev].copy()

        # Remove duplicate timestamps (M5Stick clock-resolution artifact)
        n_before = len(d)
        d = d.drop_duplicates(subset=[ts_col], keep='first').reset_index(drop=True)
        if len(d) < n_before:
            print(f"    Removed {n_before - len(d)} duplicate timestamps")

        d = d.sort_values(ts_col).reset_index(drop=True)

        # ms → seconds, zero-based
        t_raw  = d[ts_col].values.astype(np.float64)
        t_norm = (t_raw - t_raw[0]) * 1e-3
        d['timestamp'] = t_norm

        fs_in = 1.0 / np.median(np.diff(t_norm))
        print(f"    {len(d)} samples | {t_norm[-1]:.1f} s | {fs_in:.1f} Hz input")

        # Resample to uniform grid
        d, t = resample(d, t_norm, FS_TARGET)

        # Extract IMU
        acc_g    = d[['ax','ay','az']].values
        gyr_degs = d[['gx','gy','gz']].values

        # Denoise
        acc_g    = butter(acc_g,    LOWPASS_HZ, FS_TARGET, 'low')
        gyr_degs = butter(gyr_degs, LOWPASS_HZ, FS_TARGET, 'low')

        # Madgwick → quaternions + gravity-removed world-frame acc
        quats, acc_linear = run_madgwick(acc_g, gyr_degs, FS_TARGET)

        # Post-rotation denoise
        acc_linear = butter(acc_linear, LOWPASS_HZ, FS_TARGET, 'low')

        # Integration → velocity & position
        vel, pos = integrate(acc_linear, FS_TARGET)

        # Cycle detection
        peaks, gyr_bp, dom = detect_cycles(gyr_degs, FS_TARGET)
        if len(peaks) > 1:
            periods = np.diff(t[peaks])
            print(f"    Cycles: {len(peaks)} peaks | "
                  f"mean {np.mean(periods):.2f} s ({1/np.mean(periods):.2f} Hz) | "
                  f"CV {np.std(periods)/np.mean(periods):.2f}")
        else:
            print(f"    Cycles: {len(peaks)} peaks detected (check signal)")

        # Drift summary
        drift = np.array([np.polyfit(t, pos[:, i], 1)[0] for i in range(3)])
        print(f"    Drift rate (m/s²): X={drift[0]:+.4f}  Y={drift[1]:+.4f}  Z={drift[2]:+.4f}")

        # Save processed CSV
        out_df = pd.DataFrame({
            'timestamp': t,
            'qw': quats[:,0], 'qx': quats[:,1], 'qy': quats[:,2], 'qz': quats[:,3],
            'ax_w': acc_linear[:,0], 'ay_w': acc_linear[:,1], 'az_w': acc_linear[:,2],
            'vx': vel[:,0], 'vy': vel[:,1], 'vz': vel[:,2],
            'px': pos[:,0], 'py': pos[:,1], 'pz': pos[:,2],
        })
        out_path = os.path.join(DATA_PROCESSED, f"{fname}_device{dev}_processed.csv")
        out_df.to_csv(out_path, index=False)
        print(f"    Saved → {out_path}")

        # Validation plot
        save_plot(t, acc_linear, pos, peaks, gyr_bp, dom, fname, dev)


def main():
    files = sorted(glob.glob(os.path.join(DATA_RAW, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {DATA_RAW}")
    print(f"Found {len(files)} file(s). Processing first.")
    process_file(files[0])
    # To process all files:
    # for f in files: process_file(f)


if __name__ == "__main__":
    main()
