
# IMU preprocessing pipeline — batch processing all sessions
# Output: one CSV per device per session (e.g. session_device0_processed.csv, session_device1_processed.csv)
#
#   - Processes ALL raw CSV files (app-data + unified-data)
#   - Saves each device as a separate CSV file
#   - Uses session folder name for unified-data files (preserves pattern/subject labels)
#   - Generates a summary table at the end with per-session statistics

import os
import sys
import math
import glob
import json
import numpy as np

# Fix Windows console encoding for Unicode characters (µ, ‖, etc.)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

# ── Paths ─────────────────────────────────────────────────────
ROOT           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW       = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS_PROC   = os.path.join(ROOT, "results", "Data_processing")
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(RESULTS_PROC,   exist_ok=True)

# ── Parameters ────────────────────────────────────────────────
FS_TARGET       = 50.0     # Hz — target uniform sampling rate
LOWPASS_HZ      = 6.0      # Hz — cutoff (rope flow: 1–3 Hz; 6 Hz retains 2nd harmonic)
MADGWICK_BETA   = 0.1      # convergence gain (higher = more accelerometer correction)
G_MS2           = 9.80665  # m/s² per g
STATIC_WINDOW_S = 2.0      # seconds assumed static at start for gyro bias estimation
CYCLE_BAND             = (0.8, 4.0)   # Hz — bandpass shown in PSD plot only
PEAK_MIN_DIST          = 0.4      # seconds — min time between peaks (0.4 s × 50 Hz = 20 samples, matches validated diagnostic)
CYCLE_PROMINENCE_DEGS  = 50.0     # deg/s — min peak prominence
CYCLE_MIN_PEAK_DEGS    = 50.0     # deg/s — absolute minimum peak height (post-prominence filter)
GAP_WARN_MS     = 100.0    # ms  — gaps above this threshold are flagged


# ── Utility functions ─────────────────────────────────────────

def butter_filter(x, cutoff, fs, btype, order=4):
    """Zero-phase Butterworth filter applied along axis 0."""
    ny  = 0.5 * fs
    wn  = np.array(cutoff) / ny if isinstance(cutoff, (list, tuple)) else cutoff / ny
    wn  = np.clip(wn, 1e-6, 1 - 1e-6)
    b, a = signal.butter(order, wn, btype=btype)
    return signal.filtfilt(b, a, x, axis=0)


def resolve_timestamp_collisions(ts_raw):
    """
    Resolve M5Stick clock-resolution timestamp collisions without discarding data.

    The M5Stick C Plus 1.1 samples at ~33 Hz but has 1 ms timestamp resolution.
    When two consecutive samples land within the same millisecond they receive
    identical timestamps. Fix: spread runs of identical timestamps evenly across
    one median inter-sample interval. Preserves monotonicity and all data.

    Returns monotonic timestamp array of the same length as ts_raw.
    """
    ts     = ts_raw.copy().astype(np.float64)
    nonzero_diffs = np.diff(ts)
    nonzero_diffs = nonzero_diffs[nonzero_diffs > 0]
    dt_med = np.median(nonzero_diffs) if len(nonzero_diffs) > 0 else 1.0

    n_collisions = 0
    i = 0
    while i < len(ts):
        j = i + 1
        while j < len(ts) and ts[j] == ts[i]:
            j += 1
        run_len = j - i
        if run_len > 1:
            for k in range(run_len):
                ts[i + k] = ts[i] + k * dt_med / run_len
            n_collisions += run_len - 1
        i = j

    return ts, n_collisions


def check_gaps(t_seconds, threshold_s=GAP_WARN_MS / 1000.0):
    """Identify large gaps (>threshold) that will be linearly interpolated."""
    diffs = np.diff(t_seconds)
    large = np.where(diffs > threshold_s)[0]
    return [(t_seconds[i], diffs[i] * 1000) for i in large]


def resample_uniform(df, t_in, fs):
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


def rotate_to_world(quats, vecs):
    """Rotate body-frame vectors to world frame via Rodrigues formula."""
    w  = quats[:, 0:1]
    qv = quats[:, 1:]
    return vecs + 2 * w * np.cross(qv, vecs) + 2 * np.cross(qv, np.cross(qv, vecs))


def gravity_vector_from_quaternion(quats):
    """Instantaneous gravity in world frame from orientation quaternion."""
    qw, qx, qy, qz = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    gx = 2 * G_MS2 * (qx*qz - qw*qy)
    gy = 2 * G_MS2 * (qy*qz + qw*qx)
    gz =     G_MS2 * (qw**2 - qx**2 - qy**2 + qz**2)
    return np.column_stack([gx, gy, gz])


# ── Processing steps ──────────────────────────────────────────

def estimate_gyro_bias(gyr_degs, window_s=STATIC_WINDOW_S):
    """
    Estimate gyroscope DC bias from the first window_s seconds.
    Skipped if mean std > 5 deg/s (sensor was already in motion).
    """
    n_static = min(int(window_s * FS_TARGET), len(gyr_degs) // 4)
    window   = gyr_degs[:n_static]
    bias     = window.mean(axis=0)
    std_mean = np.mean(np.std(window, axis=0))

    if std_mean > 5.0:
        print(f"    Gyro bias: skipped — sensor was in motion at recording start "
              f"(window std = {std_mean:.1f} deg/s). "
              f"Ensure sensor is stationary for the first {window_s:.0f} s.")
        return np.zeros(3), False

    print(f"    Gyro bias corrected: "
          f"[{bias[0]:.2f}, {bias[1]:.2f}, {bias[2]:.2f}] deg/s")
    return bias, True


def initial_quaternion_from_acc(acc_g_samples):
    """Compute initial quaternion aligning measured gravity with world Z-up."""
    g_mean = acc_g_samples.mean(axis=0)
    g_mean = g_mean / (np.linalg.norm(g_mean) + 1e-10)
    z      = np.array([0.0, 0.0, 1.0])
    axis   = np.cross(g_mean, z)
    s      = np.linalg.norm(axis)
    c      = np.dot(g_mean, z)
    if s < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis  = axis / s
    angle = np.arctan2(s, c)
    q = np.array([
        np.cos(angle / 2),
        axis[0] * np.sin(angle / 2),
        axis[1] * np.sin(angle / 2),
        axis[2] * np.sin(angle / 2),
    ])
    return q / np.linalg.norm(q)


def run_madgwick(acc_g, gyr_degs, fs):
    """
    Madgwick AHRS with improved initialisation and per-timestep gravity removal.
    Gravity vector is smoothed at 0.5 Hz before subtraction to decouple from
    fast orientation tracking errors.
    """
    acc_ms2  = acc_g    * G_MS2
    gyr_rads = gyr_degs * (np.pi / 180.0)

    n_init = min(int(0.5 * fs), len(acc_g))
    q      = initial_quaternion_from_acc(acc_g[:n_init])

    mad   = Madgwick(beta=MADGWICK_BETA, sampleperiod=1.0 / fs)
    quats = np.zeros((len(acc_ms2), 4))
    for i in range(len(acc_ms2)):
        q = mad.updateIMU(q, gyr=gyr_rads[i], acc=acc_ms2[i])
        quats[i] = q

    acc_world  = rotate_to_world(quats, acc_ms2)
    g_smoothed = butter_filter(gravity_vector_from_quaternion(quats), 0.5, fs, 'low')
    acc_linear = acc_world - g_smoothed

    bias = acc_linear.mean(axis=0)
    print(f"    Residual acceleration bias:  "
          f"X={bias[0]:.3f}  Y={bias[1]:.3f}  Z={bias[2]:.3f}  m/s²")

    return quats, acc_linear


def detect_cycles(gyr_degs, fs):
    """
    Detect rope-flow cycles from ||omega(t)|| — matches cycle_detection_pairing_v2:
    single Savgol pass, prominence + absolute-height filter, no period-ratio culling.
    """
    mag        = np.linalg.norm(gyr_degs, axis=1)
    mag_smooth = signal.savgol_filter(mag, window_length=15, polyorder=3)

    peaks, _ = signal.find_peaks(mag_smooth,
                                 distance=int(PEAK_MIN_DIST * fs),
                                 prominence=CYCLE_PROMINENCE_DEGS)
    peaks = peaks[mag_smooth[peaks] >= CYCLE_MIN_PEAK_DEGS]

    return peaks, mag_smooth


# ── Validation plot ───────────────────────────────────────────

def save_plot(t, acc_g_raw, acc_g_filt, acc_linear,
              mag_smooth, peaks, gaps, session_name, dev, gyr_filt):
    """5-panel processing validation figure."""
    fig, axes = plt.subplots(5, 1, figsize=(11, 16))
    fig.suptitle(
        f"Stage 1 — IMU preprocessing validation\n"
        f"Session: {session_name}   Device: {dev}   "
        f"fs = {FS_TARGET:.0f} Hz   low-pass = {LOWPASS_HZ:.0f} Hz",
        fontsize=10, y=1.01
    )
    axis_labels = ['X', 'Y', 'Z']
    colours     = ['#e74c3c', '#2ecc71', '#3498db']
    t10         = t <= min(10.0, t[-1])

    # Panel 1: filtering — time domain
    ax = axes[0]
    ax.plot(t[t10], np.linalg.norm(acc_g_raw[t10]  * G_MS2, axis=1), color='#cccccc', lw=1.0, label='Raw')
    ax.plot(t[t10], np.linalg.norm(acc_g_filt[t10] * G_MS2, axis=1), color='#e74c3c', lw=1.4, label=f'Filtered ({LOWPASS_HZ} Hz)')
    ax.set_ylabel('‖a‖  (m/s²)')
    ax.set_title('Acceleration magnitude — raw vs filtered (first 10 s)')
    ax.legend(fontsize=9)

    # Panel 2: filtering — frequency domain
    ax = axes[1]
    nperseg = min(256, len(acc_g_raw))
    f_, P_ = signal.welch(acc_g_raw[:,0] * G_MS2, FS_TARGET, nperseg=nperseg)
    ax.semilogy(f_, P_, color='#cccccc', lw=1.3, label='Raw (acc-X)')
    f_f, P_f = signal.welch(acc_g_filt[:,0] * G_MS2, FS_TARGET, nperseg=nperseg)
    ax.semilogy(f_f, P_f, color='#e74c3c', lw=1.3, label=f'Filtered (acc-X, {LOWPASS_HZ} Hz)')
    ax.axvspan(*CYCLE_BAND, alpha=0.12, color='#27ae60', label='Rope-flow band')
    ax.axvline(LOWPASS_HZ, color='black', lw=1.0, ls='--', label='Cutoff')
    ax.set_ylabel('PSD  (m²/s⁴/Hz)')
    ax.set_title('Power spectral density')
    ax.legend(fontsize=9)

    # Panel 3: world-frame linear acceleration
    ax = axes[2]
    for i in range(3):
        ax.plot(t, acc_linear[:, i], color=colours[i], lw=0.8, label=f'a_{axis_labels[i]}')
    ax.axhline(0, color='black', lw=0.6, ls='--')
    for t_start, gap_ms in gaps:
        ax.axvspan(t_start, t_start + gap_ms/1000, color='#f39c12', alpha=0.3)
    ax.set_ylabel('Linear acceleration (m/s²)')
    ax.set_title('World-frame linear acceleration (gravity removed)')
    ax.legend(fontsize=9, loc='upper right')

    # Panel 4: body-frame angular velocity
    ax = axes[3]
    for i in range(3):
        ax.plot(t, gyr_filt[:, i], color=colours[i], lw=0.8, label=f'g_{axis_labels[i]}')
    ax.set_ylabel('Angular velocity (deg/s)')
    ax.set_title('Filtered body-frame angular velocity (gx, gy, gz)')
    ax.legend(fontsize=9, loc='upper right')

    # Panel 5: cycle detection
    ax = axes[4]
    ax.plot(t, mag_smooth, color='#e67e22', lw=1.0, label='||omega|| smoothed')
    if len(peaks) > 0:
        ax.plot(t[peaks], mag_smooth[peaks], 'v', color='#2c3e50', ms=6, label=f'{len(peaks)} cycles')
    ax.set_ylabel('Magnitude (deg/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Rope-flow cycle detection')
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(RESULTS_PROC, f"{session_name}_dev{dev}_validation.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


# ── Column name resolution ────────────────────────────────────
COLUMN_ALIASES = {
    'timestamp': ['timestamp', 'time', 't', 'ts', 'time_ms', 'time_us', 'timestamp_ms'],
    'device':    ['device', 'dev', 'sensor', 'id', 'device_id'],
    'ax': ['ax', 'accel_x', 'acc_x', 'a_x', 'AccX'],
    'ay': ['ay', 'accel_y', 'acc_y', 'a_y', 'AccY'],
    'az': ['az', 'accel_z', 'acc_z', 'a_z', 'AccZ'],
    'gx': ['gx', 'gyro_x', 'gyr_x', 'g_x', 'GyroX'],
    'gy': ['gy', 'gyro_y', 'gyr_y', 'g_y', 'GyroY'],
    'gz': ['gz', 'gyro_z', 'gyr_z', 'g_z', 'GyroZ'],
}

def resolve_columns(df):
    """Rename raw column names to the standard set using COLUMN_ALIASES."""
    rename     = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in cols_lower:
                rename[cols_lower[alias.lower()]] = standard
                break
    df = df.rename(columns=rename)
    df = df.drop(columns=[c for c in df.columns if 'sync' in c.lower()],
                 errors='ignore')
    return df


# ── Session name resolution ───────────────────────────────────

def session_name_from_path(path):
    """
    Derive a human-readable session name from the file path.
    - unified-data: use the parent folder name (e.g. '20260303_174607_underhand_jo')
    - app-data: use the CSV filename without extension
    """
    parent = os.path.basename(os.path.dirname(path))
    fname  = os.path.splitext(os.path.basename(path))[0]
    if parent in ("app-data", "raw"):
        return fname
    # unified-data: parent folder is the session name, file is always 'imu_data'
    return parent


# ── Single-device processing ─────────────────────────────────

def process_device(d, dev, session_name):
    """
    Run the full preprocessing pipeline on one device's data.
    Returns (out_df, plot_args) or None if processing fails.
    """
    print(f"\n  Device {dev}")
    d = d.sort_values('timestamp').reset_index(drop=True)

    if len(d) < 20:
        print(f"    Skipped — only {len(d)} samples (need >= 20)")
        return None

    # ── Timestamp collision resolution ───────────────────────
    ts_raw, n_collisions = resolve_timestamp_collisions(
        d['timestamp'].values.astype(np.float64))
    if n_collisions > 0:
        print(f"    Clock collisions: {n_collisions} timestamps spread within "
              f"their ms window — all {len(d)} samples retained")
    d['timestamp'] = ts_raw

    # ── Normalise to zero-based seconds; auto-detect unit ────
    t_raw  = d['timestamp'].values.astype(np.float64)
    t_raw  = t_raw - t_raw.min()
    nonzero_diffs = np.diff(t_raw)
    nonzero_diffs = nonzero_diffs[nonzero_diffs > 0]
    dt_med = np.median(nonzero_diffs) if len(nonzero_diffs) > 0 else 1.0
    scale  = 1e-6 if dt_med > 1e4 else (1e-3 if dt_med > 5 else 1.0)
    unit   = 'µs' if scale == 1e-6 else ('ms' if scale == 1e-3 else 's')
    t_norm = t_raw * scale
    fs_in  = 1.0 / (dt_med * scale)

    print(f"    Input:  {len(d)} samples  |  {t_norm[-1]:.1f} s  |  "
          f"{fs_in:.1f} Hz  |  timestamps in {unit}")
    print(f"    Output: resampled to {FS_TARGET:.0f} Hz")

    # ── Flag large recording gaps ────────────────────────────
    gaps = check_gaps(t_norm)
    if gaps:
        total_gap_ms = sum(g[1] for g in gaps)
        print(f"    Recording gaps: {len(gaps)} gaps > {GAP_WARN_MS:.0f} ms  "
              f"(total {total_gap_ms:.0f} ms linearly interpolated — "
              f"marked in validation plot)")

    # ── Resample to uniform grid ─────────────────────────────
    d['timestamp'] = t_norm
    d, t = resample_uniform(d, t_norm, FS_TARGET)

    # ── Extract IMU arrays ───────────────────────────────────
    acc_g   = d[['ax','ay','az']].values
    gyr_raw = d[['gx','gy','gz']].values

    # Auto-detect gyro unit from signal magnitude
    gyr_med  = np.median(np.linalg.norm(gyr_raw, axis=1))
    gyr_degs = gyr_raw.copy() if gyr_med > 10.0 else np.degrees(gyr_raw)

    # ── Gyroscope bias correction ────────────────────────────
    gyr_bias, bias_valid = estimate_gyro_bias(gyr_degs)
    if bias_valid:
        gyr_degs -= gyr_bias

    # ── Low-pass filter (single pass before Madgwick) ────────
    acc_g_raw  = acc_g.copy()
    acc_g_filt = butter_filter(acc_g,    LOWPASS_HZ, FS_TARGET, 'low')
    gyr_filt   = butter_filter(gyr_degs, LOWPASS_HZ, FS_TARGET, 'low')

    # ── Madgwick AHRS + gravity removal ──────────────────────
    quats, acc_linear = run_madgwick(acc_g_filt, gyr_filt, FS_TARGET)

    # Post-rotation low-pass (removes HF noise introduced by rotation)
    acc_linear = butter_filter(acc_linear, LOWPASS_HZ, FS_TARGET, 'low')

    # ── Cycle detection ──────────────────────────────────────
    peaks, mag_smooth = detect_cycles(gyr_filt, FS_TARGET)

    n_cycles   = len(peaks)
    mean_freq  = None
    cycle_cv   = None
    if n_cycles > 1:
        periods   = np.diff(t[peaks])
        mean_freq = 1.0 / np.mean(periods)
        cycle_cv  = np.std(periods) / np.mean(periods)
        print(f"    Cycles: {n_cycles}  |  "
              f"mean period {np.mean(periods):.2f} s  "
              f"({mean_freq:.2f} Hz)  |  "
              f"CV {cycle_cv:.2f}")
    else:
        print(f"    Cycles: {n_cycles} detected — inspect validation plot")

    # ── Build output DataFrame ───────────────────────────────
    out_df = pd.DataFrame({
        'timestamp_ms': t * 1e3,
        'qw': quats[:,0], 'qx': quats[:,1],
        'qy': quats[:,2], 'qz': quats[:,3],
        'ax_w': acc_linear[:,0],
        'ay_w': acc_linear[:,1],
        'az_w': acc_linear[:,2],
        'gx': gyr_filt[:,0],
        'gy': gyr_filt[:,1],
        'gz': gyr_filt[:,2],
    })

    # ── Save validation plot ─────────────────────────────────
    save_plot(t, acc_g_raw, acc_g_filt, acc_linear,
              mag_smooth, peaks, gaps, session_name, dev, gyr_filt)

    stats = {
        'n_samples_raw': len(d),
        'n_samples_out': len(out_df),
        'duration_s':    t[-1],
        'fs_input':      fs_in,
        'n_gaps':        len(gaps),
        'n_cycles':      n_cycles,
        'mean_freq_hz':  mean_freq,
        'cycle_cv':      cycle_cv,
    }

    return out_df, stats


# ── Main processing function ─────────────────────────────────

def process_file(path):
    """
    Process one raw CSV file. Returns dict of per-device stats, or empty dict
    if the file has no valid data.
    """
    session_name = session_name_from_path(path)
    print(f"\n{'─'*60}")
    print(f"  {session_name}")
    print(f"{'─'*60}")

    df = pd.read_csv(path)
    df = resolve_columns(df)

    if 'timestamp' not in df.columns:
        print(f"  SKIPPED — no timestamp column found")
        return {}
    if 'device' not in df.columns:
        df['device'] = 0

    device_dfs = {}
    all_stats  = {}

    for dev in sorted(df['device'].astype(str).unique()):
        d = df[df['device'].astype(str) == dev].copy()
        result = process_device(d, dev, session_name)
        if result is not None:
            out_df, stats = result
            device_dfs[dev] = out_df
            all_stats[dev]  = stats

    # ── Save one CSV per device ───────────────────────────────
    if device_dfs:
        for dev in sorted(device_dfs.keys()):
            out_path = os.path.join(DATA_PROCESSED, f"{session_name}_device{dev}_processed.csv")
            device_dfs[dev].to_csv(out_path, index=False)
            print(f"\n  Saved:  {out_path}")

    return {session_name: all_stats}


# ── Entry point ───────────────────────────────────────────────

def main():
    # Collect all raw CSV files from both app-data and unified-data
    files = sorted(glob.glob(
        os.path.join(DATA_RAW, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {DATA_RAW}")

    print(f"Found {len(files)} CSV file(s) to process.\n")

    summary_rows = []
    n_success    = 0
    n_failed     = 0

    for i, path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}]", end="")
        try:
            session_stats = process_file(path)
            for session, dev_stats in session_stats.items():
                for dev, stats in dev_stats.items():
                    summary_rows.append({
                        'session':       session,
                        'device':        dev,
                        'samples_raw':   stats['n_samples_raw'],
                        'samples_out':   stats['n_samples_out'],
                        'duration_s':    round(stats['duration_s'], 1),
                        'fs_input_hz':   round(stats['fs_input'], 1),
                        'n_gaps':        stats['n_gaps'],
                        'n_cycles':      stats['n_cycles'],
                        'mean_freq_hz':  round(stats['mean_freq_hz'], 2) if stats['mean_freq_hz'] else None,
                        'cycle_cv':      round(stats['cycle_cv'], 3) if stats['cycle_cv'] else None,
                    })
            n_success += 1
        except Exception as e:
            print(f"\n  ERROR processing {path}: {e}")
            n_failed += 1

    # ── Print summary ────────────────────────────────────────
    print(f"\n\n{'═'*60}")
    print(f"  BATCH COMPLETE: {n_success} succeeded, {n_failed} failed "
          f"out of {len(files)} files")
    print(f"{'═'*60}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(DATA_PROCESSED, "_processing_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary table saved to: {summary_path}")
        print(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
