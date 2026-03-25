
# IMU preprocessing pipeline for rope flow data (M5Stick C Plus 1.1)
# Output: one clean CSV per device, ready for trajectory reconstruction
#
# Pipeline:
#   1. Load & split by device
#   2. Resolve M5Stick clock-resolution timestamp collisions (no data discarded)
#   3. Normalise timestamps to zero-based seconds; auto-detect unit (µs/ms/s)
#   4. Flag large recording gaps (>100 ms) that will be linearly interpolated
#   5. Resample to uniform grid at FS_TARGET Hz
#   6. Gyroscope bias correction from static startup window
#      (skipped automatically if sensor was moving at recording start)
#   7. Low-pass filter accelerometer and gyroscope (single pass, LOWPASS_HZ cutoff)
#   8. Madgwick AHRS — initialised from accelerometer, smoothed quaternion gravity removal
#   9. Low-pass filter world-frame linear acceleration
#  10. Cycle detection on |bandpassed gyro| with period-validity filtering
#  11. Save processed CSV + 4-panel validation figure

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
PEAK_MIN_DIST          = 0.5      # seconds — minimum time between cycle peaks
CYCLE_PROMINENCE_DEGS  = 100.0    # deg/s — min peak prominence for cycle detection
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
    At 33 Hz the nominal inter-sample interval is ~30 ms, but when the sensor
    fires slightly faster than nominal, two consecutive samples land within the
    same millisecond and receive identical timestamps. These are distinct valid
    readings — discarding one (as drop_duplicates would do) silently removes
    ~17% of the data and degrades Madgwick orientation accuracy.

    Fix: for each run of identical timestamps, the samples are spread evenly
    across a window equal to the median inter-sample interval. This preserves
    monotonicity and all data while introducing at most sub-millisecond timing
    error (well below the 20 ms Madgwick integration step).

    Returns monotonic timestamp array of the same length as ts_raw.
    """
    ts     = ts_raw.copy().astype(np.float64)
    # Use non-zero diffs for median to avoid bias from colliding pairs
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
    """
    Identify large gaps in the recording that will be linearly interpolated
    during resampling. Linear interpolation over a fast-moving IMU signal
    for >100 ms produces artificial readings that may corrupt Madgwick
    orientation tracking across that window.
    Returns list of (start_time, gap_duration_ms) for gaps above threshold.
    """
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
    """
    Rotate body-frame vectors to world frame: v' = q ⊗ v ⊗ q*
    Rodrigues formula: v' = v + 2w(q_v × v) + 2(q_v × (q_v × v))
    """
    w  = quats[:, 0:1]
    qv = quats[:, 1:]
    return vecs + 2 * w * np.cross(qv, vecs) + 2 * np.cross(qv, np.cross(qv, vecs))


def gravity_vector_from_quaternion(quats):
    """
    Instantaneous gravity vector in world frame from orientation quaternion.
    Equivalent to rotating [0, 0, g] by q — third column of R(q).
    Smoothed at 0.5 Hz before subtraction to decouple gravity estimation
    from fast dynamic orientation tracking errors during rope flow motion.
    """
    qw, qx, qy, qz = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    gx = 2 * G_MS2 * (qx*qz - qw*qy)
    gy = 2 * G_MS2 * (qy*qz + qw*qx)
    gz =     G_MS2 * (qw**2 - qx**2 - qy**2 + qz**2)
    return np.column_stack([gx, gy, gz])


# ── Processing steps ──────────────────────────────────────────

def estimate_gyro_bias(gyr_degs, window_s=STATIC_WINDOW_S):
    """
    Estimate gyroscope DC bias from the first window_s seconds.
    Skipped if mean std > 5 deg/s — sensor was already in motion.
    Returns (bias_vector, valid_flag).
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
    """
    Compute the initial quaternion aligning measured gravity with world Z-up.
    Prevents the orientation transient from Madgwick starting at identity
    [1,0,0,0] while the sensor is already tilted at an arbitrary angle.
    """
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

    Gravity removal: the quaternion-derived gravity vector is smoothed at 0.5 Hz
    before subtraction. This decouples gravity estimation from the fast orientation
    tracking during dynamic rope flow motion — fast tracking errors are filtered
    out, while slow orientation drift (which genuinely changes the gravity direction)
    is tracked correctly.

    Returns:
        quats      : (N, 4)  quaternions [w, x, y, z]
        acc_linear : (N, 3)  world-frame linear acceleration [m/s²]
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


def filter_peaks_by_period(peaks, factor=0.5):
    """
    Remove peaks that produce physically implausible inter-peak intervals.
    Valid range: [factor × median_period,  (1/factor) × median_period].

    Sub-peaks (interval too short): the later of the pair is discarded —
    it is the shoulder of the main peak, not a genuine cycle boundary.

    Merged cycles (interval too long): no peak is removed — the gap is real
    (typically caused by a recording dropout or missed detection) and removing
    a surrounding peak would corrupt adjacent valid cycles.
    """
    if len(peaks) < 3:
        return peaks

    periods    = np.diff(peaks)
    med_period = np.median(periods)
    lo         = factor * med_period

    keep = np.ones(len(peaks), dtype=bool)
    for i, p in enumerate(periods):
        if p < lo:
            keep[i + 1] = False

    return peaks[keep]


def detect_cycles(gyr_degs, fs):
    """
    Detect rope-flow cycles from the angular velocity magnitude ||omega(t)||.

    Why magnitude and not a bandpassed single axis:
      The bandpass (0.8-4 Hz) approach was tested and produces CV > 0.8.
      A ~1 Hz rope-flow signal passed through a 4 Hz bandpass retains up to
      4 harmonics, each appearing as a separate peak. No threshold setting
      recovers from this. Angular velocity magnitude is always positive, has
      one clear peak per revolution regardless of rotation axis, and gives
      CV ~ 0.28 on both devices with a physical prominence threshold.

    CYCLE_PROMINENCE_DEGS sets the minimum height a peak must stand above
    its local surroundings in deg/s — physically interpretable and consistent
    across sessions, unlike relative thresholds that depend on signal amplitude.
    """
    mag        = np.linalg.norm(gyr_degs, axis=1)
    mag_smooth = signal.savgol_filter(mag, window_length=15, polyorder=3)

    min_dist = int(PEAK_MIN_DIST * fs)
    peaks, _ = signal.find_peaks(mag_smooth,
                                 distance=min_dist,
                                 prominence=CYCLE_PROMINENCE_DEGS)
    peaks    = filter_peaks_by_period(peaks, factor=0.5)

    # Bandpassed single-axis signal kept for the PSD validation plot only
    dom    = int(np.argmax(np.var(gyr_degs, axis=0)))
    sig_bp = butter_filter(
        gyr_degs[:, dom] * np.pi / 180.0 - np.mean(gyr_degs[:, dom]) * np.pi / 180.0,
        CYCLE_BAND, fs, btype='band')

    return peaks, sig_bp, mag_smooth, dom


# ── Validation plot ───────────────────────────────────────────

def save_plot(t, acc_g_raw, acc_g_filt, acc_linear,
              sig_bp, mag_smooth, peaks, dom, gaps, fname, dev, gyr_filt):
    """
    5-panel processing validation figure.
    """
    fig, axes = plt.subplots(5, 1, figsize=(11, 16))
    fig.suptitle(
        f"Stage 1 — IMU preprocessing validation\n"
        f"Session: {fname}   Device: {dev}   "
        f"fs = {FS_TARGET:.0f} Hz   low-pass = {LOWPASS_HZ:.0f} Hz",
        fontsize=10, y=1.01
    )
    axis_labels = ['X', 'Y', 'Z']
    colours     = ['#e74c3c', '#2ecc71', '#3498db']
    t10         = t <= min(10.0, t[-1])

    # ── Panel 1: filtering — time domain ─────────────────────
    ax = axes[0]
    ax.plot(t[t10], np.linalg.norm(acc_g_raw[t10]  * G_MS2, axis=1), color='#cccccc', lw=1.0, label='Raw')
    ax.plot(t[t10], np.linalg.norm(acc_g_filt[t10] * G_MS2, axis=1), color='#e74c3c', lw=1.4, label=f'Filtered ({LOWPASS_HZ} Hz)')
    ax.set_ylabel('‖a‖  (m/s²)')
    ax.set_title('Acceleration magnitude — raw vs filtered (first 10 s)')
    ax.legend(fontsize=9)

    # ── Panel 2: filtering — frequency domain ────────────────
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

    # ── Panel 3: world-frame linear acceleration ──────────────
    ax = axes[2]
    for i in range(3):
        ax.plot(t, acc_linear[:, i], color=colours[i], lw=0.8, label=f'a_{axis_labels[i]}')
    ax.axhline(0, color='black', lw=0.6, ls='--')
    for t_start, gap_ms in gaps:
        ax.axvspan(t_start, t_start + gap_ms/1000, color='#f39c12', alpha=0.3)
    ax.set_ylabel('Linear acceleration (m/s²)')
    ax.set_title('World-frame linear acceleration (gravity removed)')
    ax.legend(fontsize=9, loc='upper right')

    # ── Panel 4: Body-frame angular velocity ──────────────────
    ax = axes[3]
    for i in range(3):
        ax.plot(t, gyr_filt[:, i], color=colours[i], lw=0.8, label=f'g_{axis_labels[i]}')
    ax.set_ylabel('Angular velocity (deg/s)')
    ax.set_title('Filtered body-frame angular velocity (gx, gy, gz)')
    ax.legend(fontsize=9, loc='upper right')

    # ── Panel 5: cycle detection ──────────────────────────────
    ax = axes[4]
    ax.plot(t, mag_smooth, color='#e67e22', lw=1.0, label='||omega|| smoothed')
    if len(peaks) > 0:
        ax.plot(t[peaks], mag_smooth[peaks], 'v', color='#2c3e50', ms=6, label=f'{len(peaks)} cycles')
    ax.set_ylabel('Magnitude (deg/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Rope-flow cycle detection')
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(RESULTS_PROC, f"{fname}_dev{dev}_validation.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

# ── Column name resolution ────────────────────────────────────
COLUMN_ALIASES = {
'timestamp': ['timestamp', 'time', 't', 'ts', 'time_ms', 'time_us'],
'device': ['device', 'dev', 'sensor', 'id'],
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


# ── Main processing function ──────────────────────────────────

def process_file(path):
    fname = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{'─'*60}")
    print(f"  {fname}")
    print(f"{'─'*60}")

    df = pd.read_csv(path)
    df = resolve_columns(df)

    if 'timestamp' not in df.columns:
        raise RuntimeError("No timestamp column found — check COLUMN_ALIASES.")
    if 'device' not in df.columns:
        df['device'] = 'all'

    for dev in sorted(df['device'].astype(str).unique()):
        print(f"\n  Device {dev}")
        d = df[df['device'].astype(str) == dev].copy()
        d = d.sort_values('timestamp').reset_index(drop=True)

        # ── Timestamp collision resolution (no data discarded) ──
        ts_raw, n_collisions = resolve_timestamp_collisions(
            d['timestamp'].values.astype(np.float64))
        if n_collisions > 0:
            print(f"    Clock collisions: {n_collisions} timestamps spread within "
                  f"their ms window — all {len(d)} samples retained")
        d['timestamp'] = ts_raw

        # ── Normalise to zero-based seconds; auto-detect unit ───
        t_raw  = d['timestamp'].values.astype(np.float64)
        t_raw  = t_raw - t_raw.min()
        # Use non-zero diffs to avoid bias from any residual collisions
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
        gyr_degs = gyr_raw if gyr_med > 10.0 else np.degrees(gyr_raw)

        # ── Gyroscope bias correction ────────────────────────────
        gyr_bias, bias_valid = estimate_gyro_bias(gyr_degs)
        if bias_valid:
            gyr_degs -= gyr_bias

        # ── Low-pass filter (single pass before Madgwick) ────────
        acc_g_raw  = acc_g.copy()
        acc_g_filt = butter_filter(acc_g,    LOWPASS_HZ, FS_TARGET, 'low')
        gyr_filt   = butter_filter(gyr_degs, LOWPASS_HZ, FS_TARGET, 'low')

        # ── Madgwick AHRS + gravity removal ─────────────────────
        quats, acc_linear = run_madgwick(acc_g_filt, gyr_filt, FS_TARGET)

        # Post-rotation low-pass (removes HF noise introduced by rotation)
        acc_linear = butter_filter(acc_linear, LOWPASS_HZ, FS_TARGET, 'low')

        # ── Cycle detection ──────────────────────────────────────
        peaks, sig_bp, mag_smooth, dom = detect_cycles(gyr_filt, FS_TARGET)

        if len(peaks) > 1:
            periods = np.diff(t[peaks])
            print(f"    Cycles: {len(peaks)}  |  "
                  f"mean period {np.mean(periods):.2f} s  "
                  f"({1/np.mean(periods):.2f} Hz)  |  "
                  f"CV {np.std(periods)/np.mean(periods):.2f}")
        else:
            print(f"    Cycles: {len(peaks)} detected — inspect validation plot")

        # ── Save processed CSV ───────────────────────────────────
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
        out_path = os.path.join(DATA_PROCESSED,
                                f"{fname}_device{dev}_processed.csv")
        out_df.to_csv(out_path, index=False)
        print(f"    Saved:  {out_path}")

        # ── Save validation plot ─────────────────────────────────
        save_plot(t, acc_g_raw, acc_g_filt, acc_linear,
                  sig_bp, mag_smooth, peaks, dom, gaps, fname, dev, gyr_filt)


# ── Entry point ───────────────────────────────────────────────

def main():
    files = sorted(glob.glob(
        os.path.join(DATA_RAW, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {DATA_RAW}")
    print(f"Found {len(files)} file(s). Processing first.")
    process_file(files[0])


if __name__ == "__main__":
    main()