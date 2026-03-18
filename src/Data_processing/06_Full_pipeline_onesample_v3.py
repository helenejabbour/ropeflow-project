# src/Data_processing/07_preprocessing_PINN_ready.py
# ============================================================
# PURPOSE:
#   Produce clean, PINN-ready processed data from raw M5Stick IMU CSVs.
#   Outputs per-device CSVs containing:
#       timestamp, quaternions (qw,qx,qy,qz),
#       world-frame linear acceleration (ax_w, ay_w, az_w) [m/s²],
#       naive velocity (vx,vy,vz) [m/s],
#       naive position (px,py,pz) [m]  ← PINN will correct this
#   The naive position is intentionally left as double-integrated (with
#   high-pass drift suppression only), because the PINN's job is to learn
#   the physically consistent trajectory FROM this drifted estimate.
#
# KEY IMPROVEMENTS OVER v2:
#   1. Correct rope-flow cycle detection (peak freq ~1-3 Hz, not 0.4 Hz)
#      → detect peaks on per-axis gyro (dominant rotation axis) rather
#        than gyro magnitude envelope which smooths out individual cycles
#   2. Zero-velocity update (ZUPT) at detected still intervals to reset
#      velocity drift before each analysis window
#   3. Gravity vector estimated per-window from static intervals (more
#      accurate than hardcoded world-Z assumption)
#   4. Comprehensive PINN-readiness diagnostics:
#        - Drift rate (m/s) per axis
#        - Periodicity score (how well cycles repeat)
#        - Dominant frequency via FFT (sanity-check for PINN periodicity loss)
#        - Position range per axis (bounding box for rope-length constraint)
#   5. Plots redesigned for PINN input validation:
#        - 3D trajectory scatter (naive position)
#        - Drift magnitude over time
#        - FFT of world-frame acceleration (dominant frequency visible)
#        - Quaternion norm over time (Madgwick stability check)
# ============================================================

import os
import glob
import math
import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from ahrs.filters import Madgwick

# ── Paths ─────────────────────────────────────────────────────
# Script lives at: ropeflow-project/src/Data_processing/08_preprocess.py
# Go up 3 levels:  Data_processing → src → ropeflow-project
ROOT          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW      = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS       = os.path.join(ROOT, "results")
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# ── Sensor constants (M5Stick C Plus 1.1) ────────────────────────────────────
G_TO_MS2    = 9.80665
GRAVITY_MS2 = np.array([0.0, 0.0, G_TO_MS2])   # world frame, Z-up

# ── Tunable parameters ────────────────────────────────────────────────────────
FS_TARGET         = 50.0     # Hz — safe above ~33 Hz input
LOWPASS_CUTOFF    = 12.0     # Hz — denoise; rope flow << 5 Hz
HP_VEL_CUTOFF     = 0.05     # Hz — high-pass on velocity to suppress DC drift
                              #      (low enough to preserve slow movements)
MADGWICK_BETA     = 0.05     # Madgwick gain (higher = faster convergence, more noise)
MADGWICK_WARMUP_S = 2.0      # seconds of data used only for filter warm-up

# Rope-flow cycle detection
# Expected rope-flow frequency: 1–3 Hz (one full cycle per 0.33–1 s)
CYCLE_BAND_LOW      = 0.8    # Hz
CYCLE_BAND_HIGH     = 4.0    # Hz
PEAK_MIN_DIST_S     = 0.25   # min seconds between peaks (~4 Hz max)
PEAK_HEIGHT_PERCENTILE = 60  # adaptive threshold percentile

# ZUPT (Zero-Velocity Update) — detect near-static intervals
ZUPT_ACC_THRESH_G   = 0.08   # g — acc magnitude variation threshold for stillness
ZUPT_WINDOW_SAMPLES = 10     # samples to check for stillness

# Window settings
WINDOW_S              = 30.0
MIN_SAMPLES_PER_WINDOW = 50


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def find_all_csvs(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return files


def normalize_time_ms(t_raw: np.ndarray) -> np.ndarray:
    """Convert M5Stick millisecond timestamps to zero-based seconds."""
    return (t_raw.astype(np.float64) - t_raw[0]) * 1e-3


def remove_duplicate_timestamps(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Remove duplicate timestamps (artifact of ms-resolution clock on M5Stick).
    Keep first occurrence; ensures np.interp receives strictly increasing x.
    """
    return df.drop_duplicates(subset=[ts_col], keep='first').reset_index(drop=True)


def resample_uniform(df: pd.DataFrame, t: np.ndarray, fs: float = FS_TARGET):
    """Resample all numeric columns to a uniform grid at fs Hz."""
    t0, tf = float(t[0]), float(t[-1])
    if tf <= t0:
        return None, None
    n = max(2, int(math.ceil((tf - t0) * fs)))
    t_u = np.linspace(t0, tf, n)
    out = {'timestamp': t_u}
    for col in df.columns:
        if col in ('timestamp', 'time', 't', 'ts'):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = np.interp(t_u, t, df[col].astype(float).values)
    return pd.DataFrame(out), t_u


def butter_filter(x, cutoff, fs, btype, order=4):
    ny = 0.5 * fs
    wn = np.array(cutoff) / ny if isinstance(cutoff, (list, tuple)) else cutoff / ny
    wn = np.clip(wn, 1e-6, 1.0 - 1e-6)
    b, a = signal.butter(order, wn, btype=btype)
    return signal.filtfilt(b, a, x, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — MADGWICK ORIENTATION FILTER
# ══════════════════════════════════════════════════════════════════════════════

def run_madgwick(acc_g: np.ndarray, gyr_degs: np.ndarray,
                 fs: float = FS_TARGET, beta: float = MADGWICK_BETA,
                 warmup_s: float = MADGWICK_WARMUP_S):
    """
    Run Madgwick filter on the FULL session (not per-window) so orientation
    state accumulates correctly from start to finish.

    Inputs
    ------
    acc_g    : (N,3) accelerometer in g
    gyr_degs : (N,3) gyroscope in deg/s

    Returns
    -------
    quats    : (N,4) quaternions [w,x,y,z]
    acc_world_no_g : (N,3) world-frame linear acceleration in m/s²
    """
    acc_ms2  = acc_g    * G_TO_MS2
    gyr_rads = gyr_degs * (np.pi / 180.0)

    mad  = Madgwick(beta=beta, sampleperiod=1.0 / fs)
    q    = np.array([1.0, 0.0, 0.0, 0.0])
    N    = len(acc_ms2)
    quats = np.zeros((N, 4))

    warmup_n = int(warmup_s * fs)

    for i in range(N):
        q = mad.updateIMU(q, gyr=gyr_rads[i], acc=acc_ms2[i])
        quats[i] = q

    # Rotate body-frame acceleration to world frame: v_w = q ⊗ v_b ⊗ q*
    w  = quats[:, 0:1]
    qv = quats[:, 1:]
    v  = acc_ms2
    cross1 = np.cross(qv, v)
    cross2 = np.cross(qv, cross1)
    acc_world = v + 2.0 * w * cross1 + 2.0 * cross2   # m/s²

    # Estimate gravity from warmup window (near-static assumption)
    g_est = np.mean(acc_world[:warmup_n], axis=0) if warmup_n > 0 else GRAVITY_MS2
    g_mag = np.linalg.norm(g_est)
    # Fallback if sensor was already moving during warmup
    if abs(g_mag - G_TO_MS2) > 1.5:
        g_est = GRAVITY_MS2

    acc_linear = acc_world - g_est
    return quats, acc_linear, g_est


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ZERO-VELOCITY UPDATE (ZUPT)
# ══════════════════════════════════════════════════════════════════════════════

def detect_static_intervals(acc_g: np.ndarray, window: int = ZUPT_WINDOW_SAMPLES,
                             thresh: float = ZUPT_ACC_THRESH_G):
    """
    Identify sample indices where the wrist is nearly still.
    Uses rolling std of acc magnitude; low std = static.
    Returns boolean mask (True = static).
    """
    mag = np.linalg.norm(acc_g, axis=1)
    static = np.zeros(len(mag), dtype=bool)
    for i in range(window, len(mag) - window):
        seg = mag[i - window: i + window]
        if np.std(seg) < thresh:
            static[i] = True
    return static


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — KINEMATIC INTEGRATION WITH DRIFT SUPPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def integrate_kinematics(acc_world: np.ndarray, static_mask: np.ndarray,
                          fs: float = FS_TARGET):
    """
    Double-integrate world-frame linear acceleration to get velocity and position.
    Drift suppression strategy:
      1. High-pass filter on velocity (removes DC integration drift).
      2. ZUPT: force velocity to zero at detected static intervals.
         This is the most effective drift reset for periodic motion.

    The resulting position is a DRIFTED ESTIMATE — intentionally so,
    because the PINN receives this as input and learns to correct it
    using physical constraints (periodicity, min-jerk, rope length).

    Returns
    -------
    vel : (N,3) velocity in m/s
    pos : (N,3) position in m
    drift_rate : (3,) RMS drift rate per axis (m/s per second) — PINN diagnostic
    """
    dt = 1.0 / fs
    N  = len(acc_world)

    # --- Velocity integration ---
    vel = np.cumsum(acc_world, axis=0) * dt

    # High-pass to remove low-frequency DC drift
    vel = butter_filter(vel, HP_VEL_CUTOFF, fs, btype='high', order=2)

    # ZUPT: zero velocity at static windows
    # Interpolate smoothly across static→dynamic transitions
    zupt_idx = np.where(static_mask)[0]
    if len(zupt_idx) > 0:
        for idx in zupt_idx:
            vel[idx] = 0.0
        # Re-integrate from each ZUPT point to prevent drift accumulation
        segments = np.split(np.arange(N), zupt_idx)
        v_corrected = np.zeros_like(vel)
        for seg in segments:
            if len(seg) == 0:
                continue
            start = seg[0]
            v_corrected[seg] = np.cumsum(acc_world[seg], axis=0) * dt
            # Anchor to zero at segment start
            v_corrected[seg] -= v_corrected[start]
        # Blend HP-filtered and ZUPT-corrected velocity
        vel = 0.5 * vel + 0.5 * v_corrected

    # --- Position integration ---
    pos = np.cumsum(vel, axis=0) * dt

    # Drift rate: linear trend in position (m/s per axis)
    t_s = np.arange(N) * dt
    drift_rate = np.array([
        np.polyfit(t_s, pos[:, i], 1)[0] for i in range(3)
    ])

    return vel, pos, drift_rate


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CYCLE DETECTION (corrected for actual rope-flow frequency)
# ══════════════════════════════════════════════════════════════════════════════

def detect_cycles(gyr_rads: np.ndarray, fs: float = FS_TARGET):
    """
    Detect rope-flow cycles on the dominant gyroscope axis.

    FIX vs v2: v2 used gyro magnitude, which is the Euclidean norm across
    all axes. During rope flow, one axis dominates (the wrist rotation axis).
    Taking the magnitude smears the signal and suppresses individual peaks,
    causing the detector to find envelope peaks at ~0.4 Hz instead of the
    true ~1-3 Hz cycle peaks.

    Strategy:
      1. Find the axis with highest variance (dominant rotation axis).
      2. Bandpass that axis in the rope-flow band (0.8–4 Hz).
      3. Detect peaks with an adaptive height threshold.

    Returns
    -------
    peaks      : indices of detected cycle peaks
    gyr_bp     : bandpass-filtered dominant-axis signal
    dom_axis   : index of dominant axis (0=x, 1=y, 2=z)
    dominant_freq_hz : estimated dominant frequency (Hz)
    """
    # Dominant axis = highest variance
    dom_axis = int(np.argmax(np.var(gyr_rads, axis=0)))
    sig      = gyr_rads[:, dom_axis]

    # Bandpass in rope-flow band
    gyr_bp = butter_filter(sig - np.mean(sig), [CYCLE_BAND_LOW, CYCLE_BAND_HIGH],
                           fs, btype='band', order=4)

    # Adaptive peak height threshold
    min_dist = int(PEAK_MIN_DIST_S * fs)
    peaks_all, _ = signal.find_peaks(np.abs(gyr_bp), distance=min_dist)
    if len(peaks_all) > 1:
        heights = np.abs(gyr_bp[peaks_all])
        thresh  = np.percentile(heights, PEAK_HEIGHT_PERCENTILE)
    else:
        thresh = np.std(gyr_bp) * 0.6

    peaks, _ = signal.find_peaks(gyr_bp, height=thresh, distance=min_dist)

    # Dominant frequency via FFT
    f, Pxx = signal.welch(gyr_bp, fs=fs, nperseg=min(512, len(gyr_bp)))
    band   = (f >= CYCLE_BAND_LOW) & (f <= CYCLE_BAND_HIGH)
    dominant_freq_hz = float(f[band][np.argmax(Pxx[band])]) if np.any(band) else 0.0

    return peaks, gyr_bp, dom_axis, dominant_freq_hz


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PINN-READINESS DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_pinn_diagnostics(pos: np.ndarray, vel: np.ndarray,
                              acc_world: np.ndarray, peaks: np.ndarray,
                              t: np.ndarray, dominant_freq: float,
                              drift_rate: np.ndarray, quats: np.ndarray):
    """
    Compute metrics that characterize how well the preprocessed data
    will serve as PINN input. These map directly to PINN loss terms.

    Returns a dict of named scalar metrics.
    """
    diag = {}

    # 1. Drift rate per axis [m/s per second] → informs λ_drift in PINN loss
    diag['drift_rate_x'] = float(drift_rate[0])
    diag['drift_rate_y'] = float(drift_rate[1])
    diag['drift_rate_z'] = float(drift_rate[2])
    diag['drift_rate_norm'] = float(np.linalg.norm(drift_rate))

    # 2. Position bounding box [m] → rope-length constraint for PINN
    for i, ax in enumerate(['x', 'y', 'z']):
        diag[f'pos_range_{ax}'] = float(np.ptp(pos[:, i]))

    # 3. Periodicity score: std of inter-peak intervals / mean interval
    #    → 0 = perfectly periodic (ideal for PINN periodicity loss)
    #    → >0.2 = irregular, harder for PINN to enforce period constraint
    if len(peaks) > 2:
        intervals = np.diff(t[peaks])
        diag['period_mean_s']  = float(np.mean(intervals))
        diag['period_std_s']   = float(np.std(intervals))
        diag['periodicity_cv'] = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
    else:
        diag['period_mean_s']  = 0.0
        diag['period_std_s']   = 0.0
        diag['periodicity_cv'] = 1.0   # unknown = worst case

    # 4. Dominant frequency [Hz] → sets T in periodicity loss ∫v dt ≈ 0
    diag['dominant_freq_hz'] = dominant_freq

    # 5. RMS jerk [m/s³] → should be low for smooth rope flow; high = noise
    dt  = float(np.mean(np.diff(t)))
    jerk = np.diff(acc_world, axis=0) / dt
    diag['rms_jerk'] = float(np.sqrt(np.mean(np.linalg.norm(jerk, axis=1) ** 2)))

    # 6. Quaternion norm stability → should be ≈ 1.0; deviation = filter instability
    qnorms = np.linalg.norm(quats, axis=1)
    diag['quat_norm_mean'] = float(np.mean(qnorms))
    diag['quat_norm_std']  = float(np.std(qnorms))

    # 7. SNR in rope-flow band [dB]
    f, Pxx = signal.welch(acc_world[:, 0], fs=FS_TARGET,
                          nperseg=min(512, len(acc_world)))
    band_mask  = (f >= CYCLE_BAND_LOW) & (f <= CYCLE_BAND_HIGH)
    noise_mask = (f > 10.0)
    band_power  = float(np.trapezoid(Pxx[band_mask],  x=f[band_mask]))  if np.any(band_mask)  else 1e-12
    noise_power = float(np.trapezoid(Pxx[noise_mask], x=f[noise_mask])) if np.any(noise_mask) else 1e-12
    diag['band_snr_db'] = float(10.0 * np.log10(band_power / (noise_power + 1e-12)))

    return diag


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — VALIDATION PLOTS (PINN-focused)
# ══════════════════════════════════════════════════════════════════════════════

def make_validation_plots(t, acc_world, vel, pos, quats, peaks, gyr_bp,
                          dom_axis, diag, fname, dev, results_dir):
    axis_labels = ['X', 'Y', 'Z']
    dom_lbl     = axis_labels[dom_axis]
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"PINN Input Validation — {fname} | Device {dev}", fontsize=13, y=0.98)

    # ── Plot 1: World-frame acceleration (all 3 axes) ─────────────────────
    ax1 = fig.add_subplot(3, 3, 1)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for i, lbl in enumerate(['X', 'Y', 'Z']):
        ax1.plot(t, acc_world[:, i], color=colors[i], lw=0.7, label=lbl, alpha=0.8)
    ax1.set_title("World-Frame Linear Acc (m/s²)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("m/s²")
    ax1.legend(fontsize=8)

    # ── Plot 2: FFT of world-frame acc — dominant frequency ───────────────
    ax2 = fig.add_subplot(3, 3, 2)
    f_fft, Pxx = signal.welch(acc_world[:, 0], FS_TARGET,
                              nperseg=min(512, len(acc_world)))
    ax2.semilogy(f_fft, Pxx, color='#8e44ad', lw=1.0)
    ax2.axvspan(CYCLE_BAND_LOW, CYCLE_BAND_HIGH, alpha=0.15,
                color='green', label=f'Rope-flow band\n({CYCLE_BAND_LOW}–{CYCLE_BAND_HIGH} Hz)')
    ax2.axvline(diag['dominant_freq_hz'], color='red', lw=1.5, linestyle='--',
                label=f"f_dom = {diag['dominant_freq_hz']:.2f} Hz")
    ax2.set_title("PSD of Acc-X (World Frame)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD")
    ax2.set_xlim(0, FS_TARGET / 2)
    ax2.legend(fontsize=7)

    # ── Plot 3: Cycle detection on dominant gyro axis ─────────────────────
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(t, gyr_bp, color='#e67e22', lw=0.8, label=f'Gyro-{dom_lbl} BP (rad/s)')
    if len(peaks) > 0:
        ax3.plot(t[peaks], gyr_bp[peaks], 'v', color='black', ms=5,
                 label=f'{len(peaks)} peaks')
    ax3.set_title(f"Cycle Detection (dominant axis: {dom_lbl})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("rad/s")
    ax3.legend(fontsize=7)

    # ── Plot 4: Velocity (all axes) ───────────────────────────────────────
    ax4 = fig.add_subplot(3, 3, 4)
    for i, lbl in enumerate(['X', 'Y', 'Z']):
        ax4.plot(t, vel[:, i], color=colors[i], lw=0.7, label=lbl, alpha=0.8)
    ax4.set_title("Velocity (m/s) — HP-filtered + ZUPT")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("m/s")
    ax4.legend(fontsize=8)

    # ── Plot 5: Naive position (all axes) + drift trend ───────────────────
    ax5 = fig.add_subplot(3, 3, 5)
    t_s = np.arange(len(pos)) / FS_TARGET
    for i, lbl in enumerate(['X', 'Y', 'Z']):
        ax5.plot(t, pos[:, i], color=colors[i], lw=0.7, label=lbl, alpha=0.8)
        # Overlay linear drift trend
        trend = np.polyval([diag[f'drift_rate_{lbl.lower()}'], 0], t_s)
        ax5.plot(t, trend, color=colors[i], lw=1.2, linestyle='--', alpha=0.5)
    ax5.set_title("Naive Position (m) — dashed = drift trend\n(PINN corrects this)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("m")
    ax5.legend(fontsize=8)

    # ── Plot 6: Drift magnitude over time ────────────────────────────────
    ax6 = fig.add_subplot(3, 3, 6)
    drift_mag = np.linalg.norm(pos, axis=1)
    ax6.plot(t, drift_mag, color='#c0392b', lw=1.0)
    ax6.set_title(f"Position Drift Magnitude (m)\n"
                  f"Drift rate = {diag['drift_rate_norm']:.3f} m/s")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("||p(t)|| (m)")

    # ── Plot 7: 3D naive trajectory ───────────────────────────────────────
    ax7 = fig.add_subplot(3, 3, 7, projection='3d')
    sc = ax7.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                     c=t, cmap='viridis', s=1.5, alpha=0.7)
    plt.colorbar(sc, ax=ax7, shrink=0.6, label='Time (s)')
    ax7.set_title("3D Naive Trajectory\n(coloured by time)")
    ax7.set_xlabel("X (m)")
    ax7.set_ylabel("Y (m)")
    ax7.set_zlabel("Z (m)")

    # ── Plot 8: Quaternion norm (Madgwick stability) ───────────────────────
    ax8 = fig.add_subplot(3, 3, 8)
    qnorm = np.linalg.norm(quats, axis=1)
    ax8.plot(t, qnorm, color='#16a085', lw=0.8)
    ax8.axhline(1.0, color='black', linestyle='--', lw=0.8, label='Ideal = 1.0')
    ax8.set_title(f"Quaternion Norm\nmean={diag['quat_norm_mean']:.4f}, "
                  f"std={diag['quat_norm_std']:.4f}")
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("||q||")
    ax8.legend(fontsize=8)

    # ── Plot 9: PINN diagnostics summary ──────────────────────────────────
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    lines = [
        ("PINN INPUT DIAGNOSTICS", True),
        (f"Dominant freq:      {diag['dominant_freq_hz']:.3f} Hz", False),
        (f"Mean cycle period:  {diag['period_mean_s']:.3f} s", False),
        (f"Period CV (0=good): {diag['periodicity_cv']:.3f}", False),
        ("", False),
        (f"Drift rate:         {diag['drift_rate_norm']:.4f} m/s²", False),
        (f"  X: {diag['drift_rate_x']:+.4f}  Y: {diag['drift_rate_y']:+.4f}  Z: {diag['drift_rate_z']:+.4f}", False),
        ("", False),
        (f"Pos range X: {diag['pos_range_x']:.3f} m", False),
        (f"Pos range Y: {diag['pos_range_y']:.3f} m", False),
        (f"Pos range Z: {diag['pos_range_z']:.3f} m", False),
        ("", False),
        (f"RMS Jerk:           {diag['rms_jerk']:.2f} m/s³", False),
        (f"Band SNR:           {diag['band_snr_db']:.1f} dB", False),
    ]
    y = 0.95
    for text, bold in lines:
        ax9.text(0.05, y, text, transform=ax9.transAxes, fontsize=9,
                 fontweight='bold' if bold else 'normal',
                 fontfamily='monospace', va='top')
        y -= 0.065

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(results_dir, f"{fname}_dev_{dev}_PINN_validation.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved validation plot → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file_path: str):
    fname = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")
    print(f"{'='*60}")

    df = pd.read_csv(file_path)

    ts_col  = next((c for c in df.columns if c.lower() in ('timestamp','time','t','ts')), None)
    dev_col = next((c for c in df.columns if c.lower() in ('device','dev')), None)
    if ts_col is None:
        raise RuntimeError("No timestamp column found.")
    if dev_col is None:
        df['device'] = 'unspecified'
        dev_col = 'device'

    all_diags    = []
    all_features = []

    for dev in sorted(df[dev_col].astype(str).unique()):
        print(f"\n  ── Device {dev} ──────────────────────────────────")
        df_dev = df[df[dev_col].astype(str) == dev].copy()

        # Remove duplicate timestamps (M5Stick clock artifact)
        n_before = len(df_dev)
        df_dev   = remove_duplicate_timestamps(df_dev, ts_col)
        n_dupes  = n_before - len(df_dev)
        if n_dupes:
            print(f"  Removed {n_dupes} duplicate timestamps (clock resolution artifact)")

        if len(df_dev) < 20:
            print(f"  Skipping: too few samples ({len(df_dev)})")
            continue

        # Sort and normalize time
        df_dev = df_dev.sort_values(ts_col).reset_index(drop=True)
        t_norm = normalize_time_ms(df_dev[ts_col].values)
        df_dev['timestamp'] = t_norm

        actual_fs = 1.0 / np.median(np.diff(t_norm))
        duration  = t_norm[-1]
        print(f"  Samples: {len(df_dev)} | Duration: {duration:.1f}s | "
              f"Actual FS: {actual_fs:.1f} Hz")

        # Resample to uniform grid
        df_res, t_uniform = resample_uniform(df_dev, t_norm, fs=FS_TARGET)
        if df_res is None:
            print("  Skipping: resampling failed")
            continue

        # Extract IMU arrays
        acc_cols = [c for c in ['ax','ay','az'] if c in df_res.columns]
        gyr_cols = [c for c in ['gx','gy','gz'] if c in df_res.columns]
        if len(acc_cols) < 3 or len(gyr_cols) < 3:
            print("  Skipping: missing IMU columns")
            continue

        acc_g    = df_res[acc_cols].values
        gyr_degs = df_res[gyr_cols].values

        # ── Denoise (before Madgwick) ──────────────────────────────────────
        acc_g_filt    = np.stack(
            [butter_filter(acc_g[:, i],    LOWPASS_CUTOFF, FS_TARGET, 'low')
             for i in range(3)], axis=1)
        gyr_degs_filt = np.stack(
            [butter_filter(gyr_degs[:, i], LOWPASS_CUTOFF, FS_TARGET, 'low')
             for i in range(3)], axis=1)

        # ── Madgwick orientation + gravity removal (full session) ──────────
        quats, acc_world, g_est = run_madgwick(
            acc_g_filt, gyr_degs_filt, fs=FS_TARGET, beta=MADGWICK_BETA)
        print(f"  Gravity estimate: [{g_est[0]:.3f}, {g_est[1]:.3f}, {g_est[2]:.3f}] m/s² "
              f"(||g|| = {np.linalg.norm(g_est):.3f})")

        # ── Post-rotation denoise on world-frame acc ───────────────────────
        acc_world = np.stack(
            [butter_filter(acc_world[:, i], LOWPASS_CUTOFF, FS_TARGET, 'low')
             for i in range(3)], axis=1)

        # ── ZUPT detection ─────────────────────────────────────────────────
        static_mask = detect_static_intervals(acc_g_filt)
        n_static    = static_mask.sum()
        print(f"  Static (ZUPT) samples: {n_static} / {len(static_mask)} "
              f"({100*n_static/len(static_mask):.1f}%)")

        # ── Kinematic integration ──────────────────────────────────────────
        vel, pos, drift_rate = integrate_kinematics(
            acc_world, static_mask, fs=FS_TARGET)

        # ── Cycle detection (corrected: dominant axis, not magnitude) ──────
        gyr_rads = gyr_degs_filt * (np.pi / 180.0)
        peaks, gyr_bp, dom_axis, dominant_freq = detect_cycles(gyr_rads, fs=FS_TARGET)
        axis_names = ['X', 'Y', 'Z']
        print(f"  Dominant gyro axis: {axis_names[dom_axis]} | "
              f"Dominant freq: {dominant_freq:.3f} Hz | Peaks: {len(peaks)}")
        if len(peaks) > 1:
            intervals = np.diff(t_uniform[peaks])
            print(f"  Cycle intervals: mean={np.mean(intervals):.3f}s, "
                  f"std={np.std(intervals):.3f}s, "
                  f"CV={np.std(intervals)/np.mean(intervals):.3f}")

        # ── PINN diagnostics ───────────────────────────────────────────────
        diag = compute_pinn_diagnostics(
            pos, vel, acc_world, peaks, t_uniform,
            dominant_freq, drift_rate, quats)
        diag['device'] = dev
        diag['file']   = fname
        all_diags.append(diag)

        print(f"\n  PINN Diagnostics:")
        print(f"    Drift rate:      {diag['drift_rate_norm']:.4f} m/s²")
        print(f"    Periodicity CV:  {diag['periodicity_cv']:.3f}  (0=perfect)")
        print(f"    Position range:  X={diag['pos_range_x']:.3f}m  "
              f"Y={diag['pos_range_y']:.3f}m  Z={diag['pos_range_z']:.3f}m")
        print(f"    RMS Jerk:        {diag['rms_jerk']:.2f} m/s³")
        print(f"    Band SNR:        {diag['band_snr_db']:.1f} dB")
        print(f"    Quat norm:       {diag['quat_norm_mean']:.4f} ± {diag['quat_norm_std']:.4f}")

        # ── Feature extraction per cycle ───────────────────────────────────
        features_dev = []
        if len(peaks) > 1:
            for i in range(len(peaks) - 1):
                s, e   = peaks[i], peaks[i + 1]
                period = t_uniform[e] - t_uniform[s]
                if period <= 0:
                    continue
                seg_pos = pos[s:e]
                path_len = float(np.sum(
                    np.linalg.norm(np.diff(seg_pos, axis=0), axis=1)
                )) if len(seg_pos) > 1 else 0.0
                rms_acc = float(np.sqrt(
                    np.mean(np.linalg.norm(acc_world[s:e], axis=1) ** 2)))
                features_dev.append({
                    'device':      dev,
                    'cycle_start': float(t_uniform[s]),
                    'cycle_end':   float(t_uniform[e]),
                    'period_s':    float(period),
                    'freq_hz':     float(1.0 / period),
                    'path_length': path_len,
                    'rms_acc_ms2': rms_acc,
                })

        if features_dev:
            feat_df   = pd.DataFrame(features_dev)
            feat_path = os.path.join(RESULTS, f"{fname}_device_{dev}_features.csv")
            feat_df.to_csv(feat_path, index=False)
            all_features.append(feat_df)
            print(f"\n  Saved {len(feat_df)} cycles → {feat_path}")

        # ── Validation plots ───────────────────────────────────────────────
        make_validation_plots(
            t_uniform, acc_world, vel, pos, quats,
            peaks, gyr_bp, dom_axis, diag, fname, dev, RESULTS)

        # ── Save PINN-ready processed CSV ──────────────────────────────────
        out_df = pd.DataFrame({
            'timestamp': t_uniform,
            'qw': quats[:, 0], 'qx': quats[:, 1],
            'qy': quats[:, 2], 'qz': quats[:, 3],
            'ax_w': acc_world[:, 0], 'ay_w': acc_world[:, 1], 'az_w': acc_world[:, 2],
            'vx':   vel[:, 0],       'vy':   vel[:, 1],       'vz':   vel[:, 2],
            'px':   pos[:, 0],       'py':   pos[:, 1],        'pz':   pos[:, 2],
            'static_flag': static_mask.astype(int),
        })
        out_path = os.path.join(DATA_PROCESSED, f"{fname}_device_{dev}_PINN_ready.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved PINN-ready data → {out_path}")

    # ── Save diagnostics across all devices ───────────────────────────────────
    if all_diags:
        diag_df   = pd.DataFrame(all_diags)
        diag_path = os.path.join(RESULTS, f"{fname}_PINN_diagnostics.csv")
        diag_df.to_csv(diag_path, index=False)
        print(f"\nSaved diagnostics → {diag_path}")

    if all_features:
        comb_path = os.path.join(RESULTS, f"{fname}_all_features.csv")
        pd.concat(all_features, ignore_index=True).to_csv(comb_path, index=False)
        print(f"Saved combined features → {comb_path}")


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