# ─────────────────────────────────────────────────────────────
# IMU preprocessing pipeline for rope flow data (M5Stick C Plus 1.1)
# Output: one clean CSV per device, ready for trajectory reconstruction
#
# Pipeline:
#   1. Load & split by device
#   2. Remove duplicate timestamps (M5Stick ms-clock artifact)
#   3. Normalize timestamps (ms → s, zero-based)
#   4. Resample to uniform grid (FS_TARGET Hz)
#   5. Low-pass denoise (acc + gyro)
#   6. Madgwick filter → orientation quaternions + gravity removal
#   7. Cycle detection on dominant gyro axis
#   8. Save processed CSV + 4-panel before/after validation plot
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
# Script lives at: ropeflow-project/src/Data_processing/
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

    # Subtract hardcoded gravity (world frame Z-up)
    acc_linear = acc_world - np.array([0.0, 0.0, G_MS2])
    return quats, acc_linear


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


def save_plot(t, acc_g_raw, acc_g_filt, acc_linear, gyr_bp, peaks, dom, fname, dev):
    """
    4-panel before/after validation plot.
      Panel 1 — Raw vs processed acceleration (time domain, first 10 s)
                 Shows how much high-frequency noise the low-pass removed.
      Panel 2 — Raw vs processed acceleration (frequency domain)
                 Confirms signal energy is preserved in the rope-flow band
                 and noise above the cutoff is attenuated.
      Panel 3 — Body-frame vs world-frame acceleration (time domain, first 10 s)
                 Shows effect of Madgwick rotation + gravity removal:
                 the ~1g gravity bias disappears and signal centres at zero.
      Panel 4 — Cycle detection on dominant gyro axis
                 Confirms detected peaks match the periodic rope-flow rhythm.
    """
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))
    fig.suptitle(f"Processing validation — {fname}  |  Device {dev}", fontsize=11)
    labels = ['X', 'Y', 'Z']
    t_mask = t <= min(10.0, t[-1])   # first 10 s for time-domain panels

    # ── Panel 1: Raw vs processed acc — time domain ───────────────────────
    ax = axes[0]
    ax.plot(t[t_mask], np.linalg.norm(acc_g_raw[t_mask]  * G_MS2, axis=1),
        color='lightgray', lw=0.9, label='Raw ||acc|| (m/s²)')
    ax.plot(t[t_mask], np.linalg.norm(acc_g_filt[t_mask] * G_MS2, axis=1),
        color='#e74c3c', lw=1.2, label='Filtered ||acc|| (m/s²)')
    ax.set_title("Raw vs processed acceleration — time domain (first 10 s)\n"
                 "High-frequency noise removed; periodic structure preserved")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m/s²")
    ax.legend(fontsize=8)

    # ── Panel 2: Raw vs processed acc — frequency domain ─────────────────
    ax = axes[1]
    nperseg = min(256, len(acc_g_raw))
    f_raw,  P_raw  = signal.welch(acc_g_raw[:, 0] * G_MS2, FS_TARGET, nperseg=nperseg)
    f_filt, P_filt = signal.welch(acc_g_filt[:, 0] * G_MS2, FS_TARGET, nperseg=nperseg)
    ax.semilogy(f_raw,  P_raw,  color='lightgray', lw=1.0, label='Raw acc-X')
    ax.semilogy(f_filt, P_filt, color='#e74c3c',   lw=1.2, label='Processed acc-X')
    ax.axvspan(*CYCLE_BAND, alpha=0.12, color='green',
               label=f'Rope-flow band ({CYCLE_BAND[0]}–{CYCLE_BAND[1]} Hz)')
    ax.axvline(LOWPASS_HZ, color='black', lw=1.0, linestyle='--',
               label=f'Low-pass cutoff ({LOWPASS_HZ} Hz)')
    ax.set_xlim(0, FS_TARGET / 2)
    ax.set_title("Raw vs processed acceleration — frequency domain\n"
                 "Signal energy in rope-flow band preserved; noise above cutoff attenuated")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(fontsize=8)

    # ── Panel 3: Body frame vs world frame — gravity removal ──────────────
    ax = axes[2]
    ax.plot(t[t_mask], acc_g_raw[t_mask, 0] * G_MS2,
            color='lightgray', lw=0.9, label='Body-frame acc-X (gravity included)')
    ax.plot(t[t_mask], acc_linear[t_mask, 0],
            color='#3498db', lw=1.2, label='World-frame acc-X (gravity removed)')
    ax.axhline(0, color='black', lw=0.6, linestyle='--')
    ax.set_title("Body-frame vs world-frame acceleration (first 10 s)\n"
                 "Gravity bias removed; signal centred around zero")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m/s²")
    ax.legend(fontsize=8)

    # ── Panel 4: Cycle detection ──────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, gyr_bp, color='#e67e22', lw=0.8,
            label=f'Gyro-{labels[dom]} bandpass (rad/s)')
    if len(peaks):
        ax.plot(t[peaks], gyr_bp[peaks], 'v', color='black', ms=5,
                label=f'{len(peaks)} peaks detected')
    if len(peaks) > 1:
        mean_period = np.mean(np.diff(t[peaks]))
        ax.set_title(f"Rope-flow cycle detection — dominant gyro axis: {labels[dom]}\n"
                     f"Mean period: {mean_period:.2f} s  ({1/mean_period:.2f} Hz)")
    else:
        ax.set_title(f"Rope-flow cycle detection — dominant gyro axis: {labels[dom]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("rad/s")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS_PROC, f"{fname}_dev{dev}_validation.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved → {out}")


# ── Column normalisation (defined here, outside process_file) ─
COLUMN_ALIASES = {
    'timestamp': ['timestamp', 'time', 't', 'ts', 'time_ms', 'time_us'],
    'device':    ['device', 'dev', 'sensor', 'id'],
    'ax': ['ax', 'accel_x', 'acc_x', 'a_x', 'AccX'],
    'ay': ['ay', 'accel_y', 'acc_y', 'a_y', 'AccY'],
    'az': ['az', 'accel_z', 'acc_z', 'a_z', 'AccZ'],
    'gx': ['gx', 'gyro_x', 'gyr_x', 'g_x', 'GyroX'],
    'gy': ['gy', 'gyro_y', 'gyr_y', 'g_y', 'GyroY'],
    'gz': ['gz', 'gyro_z', 'gyr_z', 'g_z', 'GyroZ'],
}

def resolve_columns(df):
    """Rename columns to standard names based on known aliases. Drop sync_time."""
    rename = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in cols_lower:
                rename[cols_lower[alias.lower()]] = standard
                break
    df = df.rename(columns=rename)
    df = df.drop(columns=[c for c in df.columns if 'sync' in c.lower()], errors='ignore')
    return df


# ── Main ──────────────────────────────────────────────────────

def process_file(path):
    fname = os.path.splitext(os.path.basename(path))[0]
    print(f"\nProcessing: {fname}")

    df = pd.read_csv(path)
    df = resolve_columns(df)

    if 'timestamp' not in df.columns:
        raise RuntimeError("No timestamp column found — check COLUMN_ALIASES.")
    if 'device' not in df.columns:
        df['device'] = 'all'

    for dev in sorted(df['device'].astype(str).unique()):
        print(f"\n  Device {dev}")
        d = df[df['device'].astype(str) == dev].copy()

        # Remove duplicate timestamps (M5Stick clock-resolution artifact)
        n_before = len(d)
        d = d.drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
        if len(d) < n_before:
            print(f"    Removed {n_before - len(d)} duplicate timestamps")

        d = d.sort_values('timestamp').reset_index(drop=True)

        # Normalize timestamps to zero-based seconds (internal processing always in seconds)
        t_raw = d['timestamp'].values.astype(np.float64)
        t_raw = t_raw - t_raw.min()   # handles negative timestamps

        # Auto-detect unit from median inter-sample interval
        dt_med = np.median(np.diff(t_raw))
        if dt_med > 1e4:
            scale = 1e-6          # microseconds → seconds
        elif dt_med > 5:
            scale = 1e-3          # milliseconds → seconds
        else:
            scale = 1.0           # already seconds
        t_norm = t_raw * scale
        print(f"    Timestamp unit detected: {'us' if scale==1e-6 else 'ms' if scale==1e-3 else 's'}")

        fs_in = 1.0 / np.median(np.diff(t_norm))
        print(f"    {len(d)} samples | {t_norm[-1]:.1f} s | {fs_in:.1f} Hz input")

        # Resample to uniform grid
        d['timestamp'] = t_norm
        d, t = resample(d, t_norm, FS_TARGET)

        # Extract IMU
        acc_g   = d[['ax','ay','az']].values
        gyr_raw = d[['gx','gy','gz']].values

        # Auto-detect gyro units from median magnitude
        gyr_med = np.median(np.linalg.norm(gyr_raw, axis=1))
        if gyr_med > 10.0:
            gyr_degs = gyr_raw
            print(f"    Gyro unit: deg/s (median magnitude {gyr_med:.1f})")
        else:
            gyr_degs = np.degrees(gyr_raw)
            print(f"    Gyro unit: rad/s → converted to deg/s (median magnitude {gyr_med:.2f})")

        # Denoise — keep raw copy for validation plot
        acc_g_raw = acc_g.copy()
        acc_g    = butter(acc_g,    LOWPASS_HZ, FS_TARGET, 'low')
        acc_g_filt = acc_g.copy()      
        gyr_degs   = butter(gyr_degs, LOWPASS_HZ, FS_TARGET, 'low')

        # Madgwick → quaternions + gravity-removed world-frame acc
        quats, acc_linear = run_madgwick(acc_g, gyr_degs, FS_TARGET)

        # Post-rotation denoise
        acc_linear = butter(acc_linear, LOWPASS_HZ, FS_TARGET, 'low')

        # Cycle detection
        peaks, gyr_bp, dom = detect_cycles(gyr_degs, FS_TARGET)
        if len(peaks) > 1:
            periods = np.diff(t[peaks])
            print(f"    Cycles: {len(peaks)} peaks | "
                  f"mean {np.mean(periods):.2f} s ({1/np.mean(periods):.2f} Hz) | "
                  f"CV {np.std(periods)/np.mean(periods):.2f}")
        else:
            print(f"    Cycles: {len(peaks)} peaks detected (check signal)")

        # Save processed CSV — timestamp_ms for granularity, all other units unchanged
        out_df = pd.DataFrame({
            'timestamp_ms': t * 1e3,
            'qw': quats[:,0], 'qx': quats[:,1], 'qy': quats[:,2], 'qz': quats[:,3],
            'ax_w': acc_linear[:,0], 'ay_w': acc_linear[:,1], 'az_w': acc_linear[:,2],
        })
        out_path = os.path.join(DATA_PROCESSED, f"{fname}_device{dev}_processed.csv")
        out_df.to_csv(out_path, index=False)
        print(f"    Saved → {out_path}")

        # Validation plot
        save_plot(t, acc_g_raw, acc_g_filt, acc_linear, gyr_bp, peaks, dom, fname, dev)


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