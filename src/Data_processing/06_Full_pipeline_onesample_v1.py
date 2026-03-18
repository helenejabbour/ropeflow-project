# src/06_Full_pipeline_onesample.py

import os
import glob
import math
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

# Project architecture
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
for d in [DATA_PROCESSED, RESULTS]:
    os.makedirs(d, exist_ok=True)

# Tunable parameters (change at top of file)
FS_TARGET = 100.0         # target processing sampling rate (Hz). Lower -> faster.
LOWPASS_CUTOFF = 20.0     # lowpass used for denoising (Hz)
BAND_LOW, BAND_HIGH = 0.5, 5.0  # rope flow band for energy & cycle detection (Hz)
MADGWICK_BETA = 0.041
GRAVITY = np.array([0.0, 0.0, 9.80665])
WINDOW_S = 30.0           # window length in seconds
MIN_SAMPLES_PER_WINDOW = 50
PEAK_MIN_DISTANCE_S = 0.35  # minimum seconds between peaks
ADAPTIVE_PERCENTILE = 80    # percentile for adaptive thresholding
FALLBACK_STD_FACTOR = 0.8   # fallback threshold factor if percentile fails

# Utilities
def find_first_csv(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return files[0]

def normalize_time(t_raw):
    t = t_raw.astype(float)
    if len(t) < 2:
        return t - t[0]
    dt = np.median(np.diff(t))
    if dt > 500:
        scale = 1e-6
    elif dt > 0.5:
        scale = 1e-3
    else:
        scale = 1.0
    t_s = (t * scale) - (t[0] * scale)
    return t_s

def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)

def resample_uniform_numeric(df, t, fs=FS_TARGET):
    t0, tf = float(t[0]), float(t[-1])
    if tf <= t0:
        return None, None
    n = max(2, int(math.ceil((tf - t0) * fs)))
    t_uniform = np.linspace(t0, tf, n)
    out = pd.DataFrame({'timestamp': t_uniform})
    for col in df.columns:
        if col in ['timestamp', 'time', 't', 'ts']:
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
    b, a = signal.butter(order, cutoff/ny, btype='low')
    return signal.filtfilt(b, a, x)

def bandpass_filter(x, low=BAND_LOW, high=BAND_HIGH, fs=FS_TARGET, order=4):
    ny = 0.5 * fs
    if low <= 0 or high >= ny or low >= high:
        return x
    b, a = signal.butter(order, [low/ny, high/ny], btype='band')
    return signal.filtfilt(b, a, x)

# Orientation and world acceleration
def compute_madgwick_world_acc(acc, gyr, fs=FS_TARGET, beta=MADGWICK_BETA):
    acc = np.asarray(acc, dtype=float)
    gyr = np.asarray(gyr, dtype=float)
    if np.nanmedian(np.abs(gyr)) > 10.0:
        gyr = np.deg2rad(gyr)
    mad = Madgwick(beta=beta, sampleperiod=1.0/fs)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.zeros((len(acc), 4))
    for i in range(len(acc)):
        q = mad.updateIMU(q, gyr=gyr[i], acc=acc[i])
        quats[i] = q
    w = quats[:, 0][:, None]
    qv = quats[:, 1:]
    cross1 = np.cross(qv, acc)
    cross2 = np.cross(qv, cross1)
    acc_world = acc + 2.0 * w * cross1 + 2.0 * cross2
    acc_world_minus_g = acc_world - GRAVITY
    return quats, acc_world_minus_g

# Peak detection with adaptive threshold
def detect_cycles_adaptive(gyr_full, fs=FS_TARGET, percentile=ADAPTIVE_PERCENTILE, fallback_factor=FALLBACK_STD_FACTOR):
    gyr_mag = np.linalg.norm(gyr_full, axis=1)
    gyr_bp = bandpass_filter(gyr_mag - np.mean(gyr_mag), fs=fs)
    try:
        peaks_all, props = signal.find_peaks(gyr_bp, distance=int(PEAK_MIN_DISTANCE_S * fs))
        if len(peaks_all) == 0:
            raise ValueError("no peaks found for percentile")
        heights = gyr_bp[peaks_all]
        thresh = np.percentile(heights, percentile)
        if not np.isfinite(thresh) or thresh <= 0:
            raise ValueError("percentile threshold invalid")
    except Exception:
        thresh = np.std(gyr_bp) * fallback_factor
    peaks, props = signal.find_peaks(gyr_bp, height=thresh, distance=int(PEAK_MIN_DISTANCE_S * fs))
    return peaks, gyr_bp, thresh

# Energy and SNR diagnostics
def compute_band_energy(x, fs=FS_TARGET, low=BAND_LOW, high=BAND_HIGH):
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (f >= low) & (f <= high)
    # Replaced deprecated np.trapz with np.trapezoid
    return np.trapezoid(Pxx[mask], x=f[mask]) if np.any(mask) else 0.0

# Main per-file pipeline (first CSV only)
def process_first_file():
    file_path = find_first_csv(DATA_RAW)
    fname = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing single file: {fname}")

    df = pd.read_csv(file_path)
    ts_col = next((c for c in df.columns if c.lower() in ('timestamp', 'time', 't', 'ts')), None)
    if ts_col is None:
        raise RuntimeError("No timestamp column found in CSV")
    dev_col = next((c for c in df.columns if c.lower() in ('device', 'dev')), None)
    if dev_col is None:
        df['device'] = 'unspecified'
        dev_col = 'device'

    all_features = []
    for dev in sorted(df[dev_col].astype(str).unique()):
        df_dev = df[df[dev_col].astype(str) == dev].reset_index(drop=True)
        if len(df_dev) < 10:
            print(f"Skipping device {dev}: too few samples")
            continue

        t_raw = df_dev[ts_col].values.astype(float)
        t_norm = normalize_time(t_raw)
        df_dev = df_dev.copy()
        df_dev['timestamp'] = t_norm

        t0, tf = t_norm[0], t_norm[-1]
        window_starts = np.arange(t0, tf, WINDOW_S)
        processed_windows = 0
        features_dev = []

        # Initialize kinematic carry-over states for the device
        v0 = np.zeros(3)
        p0 = np.zeros(3)

        for ws in window_starts:
            we = ws + WINDOW_S
            win = df_dev[(df_dev['timestamp'] >= ws) & (df_dev['timestamp'] < we)].reset_index(drop=True)
            if len(win) < MIN_SAMPLES_PER_WINDOW:
                continue

            df_res, t_uniform = resample_uniform_numeric(win, win['timestamp'].values, fs=FS_TARGET)
            if df_res is None:
                continue

            for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                if col in df_res.columns:
                    df_res[f"{col}_raw_interp"] = df_res[col].values.copy()

            for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                if col in df_res.columns and is_numeric_series(df_res[col]):
                    df_res[col] = butter_lowpass_filter(df_res[col].values, cutoff=LOWPASS_CUTOFF, fs=FS_TARGET)

            acc_cols = [c for c in ['ax', 'ay', 'az'] if c in df_res.columns]
            gyr_cols = [c for c in ['gx', 'gy', 'gz'] if c in df_res.columns]
            if len(acc_cols) < 3 or len(gyr_cols) < 3:
                acc = np.zeros((len(df_res), 3))
                gyr = np.zeros((len(df_res), 3))
            else:
                acc = df_res[acc_cols].values
                gyr = df_res[gyr_cols].values

            quats, acc_world = compute_madgwick_world_acc(acc, gyr, fs=FS_TARGET, beta=MADGWICK_BETA)

            for i in range(3):
                acc_world[:, i] = butter_lowpass_filter(acc_world[:, i], cutoff=LOWPASS_CUTOFF, fs=FS_TARGET)

            # Continuous Integration with state carry-over
            dt = 1.0 / FS_TARGET
            
            # 1. Integrate acceleration to velocity, adding initial state v0
            vel_raw = np.cumsum(acc_world, axis=0) * dt + v0
            
            hp_cut = 0.1
            if hp_cut < 0.5 * FS_TARGET:
                b, a = signal.butter(2, hp_cut / (0.5 * FS_TARGET), btype='high')
                vel_hp = signal.filtfilt(b, a, vel_raw, axis=0)
            else:
                vel_hp = vel_raw
                
            # 2. Integrate velocity to position, adding initial state p0
            pos = np.cumsum(vel_hp, axis=0) * dt + p0

            # 3. Update carry-over states for the next window
            v0 = vel_hp[-1]
            p0 = pos[-1]

            gyr_full = gyr if gyr.shape[1] == 3 else np.zeros_like(acc_world)
            peaks, gyr_bp, thresh = detect_cycles_adaptive(gyr_full, fs=FS_TARGET)
            
            if len(peaks) > 1:
                for i in range(len(peaks) - 1):
                    s, e = peaks[i], peaks[i + 1]
                    period = t_uniform[e] - t_uniform[s]
                    if period <= 0:
                        continue
                    seg_pos = pos[s:e]
                    path_len = float(np.sum(np.linalg.norm(np.diff(seg_pos, axis=0), axis=1))) if len(seg_pos) > 1 else 0.0
                    rms_acc = float(np.sqrt(np.mean(np.linalg.norm(acc_world[s:e], axis=1) ** 2)))
                    features_dev.append({
                        'device': dev,
                        'window_start': float(ws),
                        'cycle_start': float(t_uniform[s]),
                        'cycle_end': float(t_uniform[e]),
                        'period_s': float(period),
                        'freq_hz': float(1.0 / period),
                        'path_length': path_len,
                        'rms_acc': rms_acc
                    })

            ax_raw = df_res['ax_raw_interp'].values if 'ax_raw_interp' in df_res.columns else np.zeros(len(df_res))
            ax_filt = df_res['ax'].values if 'ax' in df_res.columns else np.zeros(len(df_res))
            energy_raw = compute_band_energy(ax_raw, fs=FS_TARGET, low=BAND_LOW, high=BAND_HIGH)
            energy_filt = compute_band_energy(ax_filt, fs=FS_TARGET, low=BAND_LOW, high=BAND_HIGH)
            residual = ax_raw - ax_filt
            noise_rms = float(np.sqrt(np.mean(residual ** 2)))
            signal_rms = float(np.sqrt(np.mean(ax_filt ** 2)))
            snr_db = float(20.0 * np.log10((signal_rms / (noise_rms + 1e-12)) + 1e-12))

            print(f"Device {dev} window {ws:.1f}-{we:.1f}s | peaks {len(peaks)} | thresh {thresh:.4f} | noise_rms {noise_rms:.4f} m/s^2 | SNR {snr_db:.2f} dB")

            out_df = pd.DataFrame({
                'timestamp': t_uniform,
                'qw': quats[:, 0], 'qx': quats[:, 1], 'qy': quats[:, 2], 'qz': quats[:, 3],
                'ax_w': acc_world[:, 0], 'ay_w': acc_world[:, 1], 'az_w': acc_world[:, 2],
                'vx': vel_hp[:, 0], 'vy': vel_hp[:, 1], 'vz': vel_hp[:, 2],
                'px': pos[:, 0], 'py': pos[:, 1], 'pz': pos[:, 2],
            })
            out_path = os.path.join(DATA_PROCESSED, f"{fname}_device_{dev}_processed.csv")
            header = not os.path.exists(out_path)
            out_df.to_csv(out_path, mode='a', index=False, header=header)

            if processed_windows == 0:
                fig, axes = plt.subplots(2, 1, figsize=(10, 7))
                mask = t_uniform < min(5.0, t_uniform[-1])
                axes[0].plot(t_uniform[mask], ax_raw[mask], color='lightgray', label='Raw (interp)')
                axes[0].plot(t_uniform[mask], ax_filt[mask], color='blue', label='Filtered')
                axes[0].set_title(f"Time Domain (Ax) - Device {dev}")
                axes[0].legend()
                f_raw, p_raw = signal.welch(ax_raw, FS_TARGET, nperseg=min(1024, len(ax_raw)))
                f_filt, p_filt = signal.welch(ax_filt, FS_TARGET, nperseg=min(1024, len(ax_filt)))
                axes[1].semilogy(f_raw, p_raw, color='lightgray', label='Raw PSD')
                axes[1].semilogy(f_filt, p_filt, color='red', label='Filtered PSD')
                axes[1].axvline(LOWPASS_CUTOFF, color='black', linestyle='--', label=f'LP cutoff {LOWPASS_CUTOFF}Hz')
                axes[1].set_title("Frequency Domain")
                axes[1].legend()
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS, f"{fname}_dev_{dev}_window_{int(ws)}_validation.png"), dpi=150)
                plt.close()

            processed_windows += 1

        if features_dev:
            features_df = pd.DataFrame(features_dev)
            features_df.to_csv(os.path.join(RESULTS, f"{fname}_device_{dev}_features.csv"), index=False)
            all_features.append(features_df)
            print(f"Saved features for device {dev} ({len(features_df)} cycles)")

    if all_features:
        pd.concat(all_features, ignore_index=True).to_csv(os.path.join(RESULTS, f"{fname}_all_features.csv"), index=False)
        print("Saved combined features CSV")

if __name__ == "__main__":
    process_first_file()