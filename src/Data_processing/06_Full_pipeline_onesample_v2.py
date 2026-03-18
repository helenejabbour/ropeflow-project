# src/06_Full_pipeline_onesample_v2.py

import os
import glob
import math
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

# ── Project architecture ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_RAW     = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS      = os.path.join(ROOT, "results")
for d in [DATA_PROCESSED, RESULTS]:
    os.makedirs(d, exist_ok=True)

# ── Sensor constants ──────────────────────────────────────────────────────────
# M5Stick C Plus 1.1 specifics:
#   - Timestamps  : milliseconds (Unix epoch ms)
#   - Accelerometer: g  (must multiply by 9.80665 for m/s²)
#   - Gyroscope   : deg/s (must convert to rad/s for Madgwick)
G_TO_MS2      = 9.80665          # 1 g in m/s²
GRAVITY_MS2   = np.array([0.0, 0.0, G_TO_MS2])   # gravity in m/s² (world frame, Z-up)

# ── Tunable parameters ────────────────────────────────────────────────────────
# Input rate is ~34.5 Hz per device; resample to 50 Hz (safe headroom above Nyquist)
FS_TARGET           = 50.0        # target processing rate (Hz)
LOWPASS_CUTOFF      = 15.0        # lowpass for denoising (Hz) — rope flow rarely > 5 Hz
BAND_LOW            = 0.5         # rope flow bandpass low (Hz)
BAND_HIGH           = 5.0         # rope flow bandpass high (Hz)
MADGWICK_BETA       = 0.041       # Madgwick filter gain (tune 0.01–0.1)
WINDOW_S            = 30.0        # analysis window length (s)
MIN_SAMPLES_PER_WINDOW = 50
PEAK_MIN_DISTANCE_S = 0.35        # min seconds between detected peaks
ADAPTIVE_PERCENTILE = 80
FALLBACK_STD_FACTOR = 0.8
HP_VEL_CUTOFF       = 0.1         # high-pass on velocity to remove DC drift (Hz)


# ── Utilities ─────────────────────────────────────────────────────────────────

def find_first_csv(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return files[0]


def normalize_time_ms(t_raw: np.ndarray) -> np.ndarray:
    """
    Convert raw M5Stick timestamps (Unix epoch, milliseconds) to
    seconds starting at 0.

    FIX vs v1: v1 used a heuristic `if dt > 500 -> scale=1e-6` which
    mis-classified ms timestamps (median dt ~7 ms for interleaved,
    ~29 ms per-device) and applied no scaling, treating ms as seconds.
    """
    t = t_raw.astype(np.float64)
    t_s = (t - t[0]) * 1e-3          # ms -> seconds, zero-based
    return t_s


def remove_duplicate_timestamps(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Remove rows with duplicate timestamps within each device.
    Duplicates break np.interp (requires strictly increasing x).
    Keep the first occurrence.
    """
    return df.drop_duplicates(subset=[ts_col], keep='first').reset_index(drop=True)


def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)


def resample_uniform_numeric(df: pd.DataFrame, t: np.ndarray, fs: float = FS_TARGET):
    """Resample all numeric columns to a uniform time grid at `fs` Hz."""
    t0, tf = float(t[0]), float(t[-1])
    if tf <= t0:
        return None, None
    n = max(2, int(math.ceil((tf - t0) * fs)))
    t_uniform = np.linspace(t0, tf, n)
    out = pd.DataFrame({'timestamp': t_uniform})
    for col in df.columns:
        if col in ('timestamp', 'time', 't', 'ts'):
            continue
        if is_numeric_series(df[col]):
            out[col] = np.interp(t_uniform, t, df[col].astype(float).values)
        else:
            out[col] = df[col].iloc[0]
    return out, t_uniform


def butter_lowpass_filter(x, cutoff=LOWPASS_CUTOFF, fs=FS_TARGET, order=4):
    ny = 0.5 * fs
    if cutoff >= ny:
        return x
    b, a = signal.butter(order, cutoff / ny, btype='low')
    return signal.filtfilt(b, a, x)


def butter_highpass_filter(x, cutoff=HP_VEL_CUTOFF, fs=FS_TARGET, order=2):
    ny = 0.5 * fs
    if cutoff >= ny:
        return x
    b, a = signal.butter(order, cutoff / ny, btype='high')
    return signal.filtfilt(b, a, x)


def bandpass_filter(x, low=BAND_LOW, high=BAND_HIGH, fs=FS_TARGET, order=4):
    ny = 0.5 * fs
    if low <= 0 or high >= ny or low >= high:
        return x
    b, a = signal.butter(order, [low / ny, high / ny], btype='band')
    return signal.filtfilt(b, a, x)


# ── Orientation and world-frame acceleration ──────────────────────────────────

def compute_madgwick_world_acc(acc_g: np.ndarray, gyr_degs: np.ndarray,
                               fs: float = FS_TARGET, beta: float = MADGWICK_BETA):
    """
    Compute orientation quaternions and gravity-removed world-frame acceleration.

    FIX vs v1:
      - acc input was in g; Madgwick expects m/s² -> convert here.
      - gyr input is deg/s from M5Stick; Madgwick expects rad/s -> convert here
        (v1 had a heuristic `if median(|gyr|) > 10 -> deg2rad` which would
         ALSO fire on rad/s data with large values, and would miss conversion
         on low-motion segments). Explicit conversion is always correct.
      - Quaternion rotation formula corrected to standard sandwich product:
        v_world = q ⊗ v_body ⊗ q*
        (v1 used an approximation that is only valid for unit vectors and
         small rotations; the full rotation is implemented below.)
      - Gravity removed AFTER rotating to world frame (in m/s²).

    Returns
    -------
    quats : (N,4) float — [w, x, y, z] unit quaternions
    acc_world_no_g : (N,3) float — linear acceleration in m/s² (world frame)
    """
    acc_ms2  = np.asarray(acc_g,    dtype=float) * G_TO_MS2   # g  -> m/s²
    gyr_rads = np.asarray(gyr_degs, dtype=float) * (np.pi / 180.0)  # deg/s -> rad/s

    mad   = Madgwick(beta=beta, sampleperiod=1.0 / fs)
    q     = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.zeros((len(acc_ms2), 4))

    for i in range(len(acc_ms2)):
        q = mad.updateIMU(q, gyr=gyr_rads[i], acc=acc_ms2[i])
        quats[i] = q

    # Full quaternion rotation: v_world = q ⊗ [0, v] ⊗ q*
    # Using the Rodrigues formula for efficiency:
    # v_rot = v + 2w(q_vec × v) + 2(q_vec × (q_vec × v))
    w   = quats[:, 0:1]          # (N,1)
    qv  = quats[:, 1:]           # (N,3)
    v   = acc_ms2

    cross1      = np.cross(qv, v)
    cross2      = np.cross(qv, cross1)
    acc_world   = v + 2.0 * w * cross1 + 2.0 * cross2   # same formula, now in m/s²

    acc_world_no_g = acc_world - GRAVITY_MS2
    return quats, acc_world_no_g


# ── Peak / cycle detection ────────────────────────────────────────────────────

def detect_cycles_adaptive(gyr_full: np.ndarray, fs: float = FS_TARGET,
                           percentile: int = ADAPTIVE_PERCENTILE,
                           fallback_factor: float = FALLBACK_STD_FACTOR):
    """
    Detect rope-flow cycles using gyroscope magnitude peaks.
    gyr_full is in rad/s after the fix; bandpass is applied in that domain.
    """
    gyr_mag = np.linalg.norm(gyr_full, axis=1)
    gyr_bp  = bandpass_filter(gyr_mag - np.mean(gyr_mag), fs=fs)

    try:
        peaks_all, _ = signal.find_peaks(gyr_bp, distance=int(PEAK_MIN_DISTANCE_S * fs))
        if len(peaks_all) == 0:
            raise ValueError("no peaks")
        heights = gyr_bp[peaks_all]
        thresh  = np.percentile(heights, percentile)
        if not np.isfinite(thresh) or thresh <= 0:
            raise ValueError("bad percentile")
    except Exception:
        thresh = np.std(gyr_bp) * fallback_factor

    peaks, props = signal.find_peaks(gyr_bp, height=thresh,
                                     distance=int(PEAK_MIN_DISTANCE_S * fs))
    return peaks, gyr_bp, thresh


# ── Band energy (SNR diagnostics) ─────────────────────────────────────────────

def compute_band_energy(x, fs=FS_TARGET, low=BAND_LOW, high=BAND_HIGH):
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (f >= low) & (f <= high)
    return np.trapezoid(Pxx[mask], x=f[mask]) if np.any(mask) else 0.0


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_first_file():
    file_path = find_first_csv(DATA_RAW)
    fname     = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nProcessing: {fname}")

    df = pd.read_csv(file_path)

    # ── Column detection ───────────────────────────────────────────────────
    ts_col = next((c for c in df.columns if c.lower() in ('timestamp', 'time', 't', 'ts')), None)
    if ts_col is None:
        raise RuntimeError("No timestamp column found in CSV")

    dev_col = next((c for c in df.columns if c.lower() in ('device', 'dev')), None)
    if dev_col is None:
        df['device'] = 'unspecified'
        dev_col = 'device'

    all_features = []

    for dev in sorted(df[dev_col].astype(str).unique()):
        df_dev = df[df[dev_col].astype(str) == dev].copy()

        # ── FIX: remove duplicates before any time-based operations ───────
        n_before = len(df_dev)
        df_dev = remove_duplicate_timestamps(df_dev, ts_col)
        n_dupes = n_before - len(df_dev)
        if n_dupes > 0:
            print(f"  Device {dev}: removed {n_dupes} duplicate timestamps")

        if len(df_dev) < 10:
            print(f"  Skipping device {dev}: too few samples ({len(df_dev)})")
            continue

        # ── FIX: normalize timestamps correctly (ms -> s) ─────────────────
        t_raw  = df_dev[ts_col].values.astype(float)
        t_norm = normalize_time_ms(t_raw)
        df_dev['timestamp'] = t_norm

        # Diagnostics
        actual_fs = 1.0 / np.median(np.diff(t_norm))
        print(f"  Device {dev}: {len(df_dev)} samples, "
              f"duration={t_norm[-1]:.1f}s, actual_fs={actual_fs:.1f} Hz")

        t0, tf = t_norm[0], t_norm[-1]
        window_starts    = np.arange(t0, tf, WINDOW_S)
        processed_windows = 0
        features_dev     = []
        v0 = np.zeros(3)
        p0 = np.zeros(3)

        for ws in window_starts:
            we  = ws + WINDOW_S
            win = df_dev[(df_dev['timestamp'] >= ws) &
                         (df_dev['timestamp'] < we)].reset_index(drop=True)
            if len(win) < MIN_SAMPLES_PER_WINDOW:
                print(f"    Window {ws:.0f}-{we:.0f}s: only {len(win)} samples, skipping")
                continue

            df_res, t_uniform = resample_uniform_numeric(
                win, win['timestamp'].values, fs=FS_TARGET)
            if df_res is None:
                continue

            # Save raw (before filtering) for SNR computation
            for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                if col in df_res.columns:
                    df_res[f"{col}_raw"] = df_res[col].values.copy()

            # Low-pass denoise (acc in g, gyr in deg/s — units don't matter here)
            for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                if col in df_res.columns and is_numeric_series(df_res[col]):
                    df_res[col] = butter_lowpass_filter(
                        df_res[col].values, cutoff=LOWPASS_CUTOFF, fs=FS_TARGET)

            acc_cols = [c for c in ['ax', 'ay', 'az'] if c in df_res.columns]
            gyr_cols = [c for c in ['gx', 'gy', 'gz'] if c in df_res.columns]

            if len(acc_cols) < 3 or len(gyr_cols) < 3:
                acc_g    = np.zeros((len(df_res), 3))
                gyr_degs = np.zeros((len(df_res), 3))
            else:
                acc_g    = df_res[acc_cols].values   # in g
                gyr_degs = df_res[gyr_cols].values   # in deg/s

            # ── Madgwick + gravity removal (FIX: units handled inside) ────
            quats, acc_world = compute_madgwick_world_acc(
                acc_g, gyr_degs, fs=FS_TARGET, beta=MADGWICK_BETA)

            # Secondary low-pass on world-frame acceleration
            for i in range(3):
                acc_world[:, i] = butter_lowpass_filter(
                    acc_world[:, i], cutoff=LOWPASS_CUTOFF, fs=FS_TARGET)

            # ── Kinematic integration with drift suppression ───────────────
            dt = 1.0 / FS_TARGET

            # Velocity: cumulative integral of world-frame linear acceleration
            vel_raw = np.cumsum(acc_world, axis=0) * dt + v0
            # High-pass to remove low-frequency integration drift
            vel_hp  = np.stack(
                [butter_highpass_filter(vel_raw[:, i], cutoff=HP_VEL_CUTOFF, fs=FS_TARGET)
                 for i in range(3)], axis=1)

            # Position: cumulative integral of de-drifted velocity
            pos = np.cumsum(vel_hp, axis=0) * dt + p0

            # Carry state to next window for continuity
            v0 = vel_hp[-1]
            p0 = pos[-1]

            # ── Cycle detection (on rad/s gyro magnitude) ─────────────────
            gyr_rads_full = gyr_degs * (np.pi / 180.0)
            peaks, gyr_bp, thresh = detect_cycles_adaptive(gyr_rads_full, fs=FS_TARGET)

            if len(peaks) > 1:
                for i in range(len(peaks) - 1):
                    s, e   = peaks[i], peaks[i + 1]
                    period = t_uniform[e] - t_uniform[s]
                    if period <= 0:
                        continue
                    seg_pos  = pos[s:e]
                    path_len = float(np.sum(np.linalg.norm(np.diff(seg_pos, axis=0), axis=1))
                                     ) if len(seg_pos) > 1 else 0.0
                    rms_acc  = float(np.sqrt(np.mean(
                        np.linalg.norm(acc_world[s:e], axis=1) ** 2)))
                    features_dev.append({
                        'device':       dev,
                        'window_start': float(ws),
                        'cycle_start':  float(t_uniform[s]),
                        'cycle_end':    float(t_uniform[e]),
                        'period_s':     float(period),
                        'freq_hz':      float(1.0 / period),
                        'path_length':  path_len,
                        'rms_acc':      rms_acc,
                    })

            # ── SNR diagnostics ───────────────────────────────────────────
            ax_raw  = df_res['ax_raw'].values  if 'ax_raw'  in df_res.columns else np.zeros(len(df_res))
            ax_filt = df_res['ax'].values      if 'ax'      in df_res.columns else np.zeros(len(df_res))
            noise_rms  = float(np.sqrt(np.mean((ax_raw - ax_filt) ** 2)))
            signal_rms = float(np.sqrt(np.mean(ax_filt ** 2)))
            snr_db     = float(20.0 * np.log10((signal_rms / (noise_rms + 1e-12)) + 1e-12))

            print(f"    Window {ws:.0f}-{we:.0f}s | "
                  f"peaks={len(peaks)} | thresh={thresh:.4f} rad/s | "
                  f"noise_rms={noise_rms:.4f} g | SNR={snr_db:.2f} dB")

            # ── Save processed window ─────────────────────────────────────
            out_df = pd.DataFrame({
                'timestamp': t_uniform,
                'qw': quats[:, 0], 'qx': quats[:, 1],
                'qy': quats[:, 2], 'qz': quats[:, 3],
                'ax_w': acc_world[:, 0], 'ay_w': acc_world[:, 1], 'az_w': acc_world[:, 2],
                'vx': vel_hp[:, 0], 'vy': vel_hp[:, 1], 'vz': vel_hp[:, 2],
                'px': pos[:, 0],    'py': pos[:, 1],    'pz': pos[:, 2],
            })
            out_path = os.path.join(DATA_PROCESSED,
                                    f"{fname}_device_{dev}_processed.csv")
            out_df.to_csv(out_path, mode='a', index=False,
                          header=not os.path.exists(out_path))

            # ── Validation plot (first window only) ───────────────────────
            if processed_windows == 0:
                fig, axes = plt.subplots(3, 1, figsize=(11, 9))

                # Time domain: ax raw vs filtered (in g)
                mask = t_uniform < min(5.0, t_uniform[-1])
                axes[0].plot(t_uniform[mask], ax_raw[mask],
                             color='lightgray', label='Raw (g)')
                axes[0].plot(t_uniform[mask], ax_filt[mask],
                             color='royalblue', label='Filtered (g)')
                axes[0].set_title(f"Ax — Time Domain (Device {dev})")
                axes[0].set_ylabel("Acceleration (g)")
                axes[0].legend()

                # Frequency domain
                f_raw,  p_raw  = signal.welch(ax_raw,  FS_TARGET,
                                              nperseg=min(256, len(ax_raw)))
                f_filt, p_filt = signal.welch(ax_filt, FS_TARGET,
                                              nperseg=min(256, len(ax_filt)))
                axes[1].semilogy(f_raw,  p_raw,  color='lightgray', label='Raw PSD')
                axes[1].semilogy(f_filt, p_filt, color='crimson',   label='Filtered PSD')
                axes[1].axvline(LOWPASS_CUTOFF, color='black',
                                linestyle='--', label=f'LP {LOWPASS_CUTOFF} Hz')
                axes[1].axvspan(BAND_LOW, BAND_HIGH, alpha=0.15,
                                color='green', label='Rope-flow band')
                axes[1].set_title("Frequency Domain")
                axes[1].set_ylabel("PSD")
                axes[1].legend()

                # Gyroscope magnitude + detected peaks
                gyr_mag_bp = np.linalg.norm(gyr_rads_full, axis=1)
                axes[2].plot(t_uniform, gyr_mag_bp,
                             color='darkorange', lw=0.8, label='|ω| (rad/s)')
                axes[2].plot(t_uniform[peaks], gyr_mag_bp[peaks],
                             'v', color='black', ms=6, label='Detected peaks')
                axes[2].set_title("Gyro Magnitude + Cycle Peaks")
                axes[2].set_ylabel("|ω| (rad/s)")
                axes[2].set_xlabel("Time (s)")
                axes[2].legend()

                plt.tight_layout()
                fig_path = os.path.join(
                    RESULTS, f"{fname}_dev_{dev}_w{int(ws)}_validation.png")
                plt.savefig(fig_path, dpi=150)
                plt.close()
                print(f"    Saved validation plot: {fig_path}")

            processed_windows += 1

        # ── Save features ─────────────────────────────────────────────────
        if features_dev:
            feat_df = pd.DataFrame(features_dev)
            feat_path = os.path.join(RESULTS, f"{fname}_device_{dev}_features.csv")
            feat_df.to_csv(feat_path, index=False)
            all_features.append(feat_df)
            print(f"  Saved {len(feat_df)} cycles for device {dev} -> {feat_path}")

    if all_features:
        combined_path = os.path.join(RESULTS, f"{fname}_all_features.csv")
        pd.concat(all_features, ignore_index=True).to_csv(combined_path, index=False)
        print(f"\nSaved combined features -> {combined_path}")


if __name__ == "__main__":
    process_first_file()