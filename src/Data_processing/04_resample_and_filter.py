import os
import glob
import math
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Project-relative paths
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(ROOT, "data", "raw")

FS_TARGET = 200.0  # Uniform sampling rate (Hz)
LOWPASS_CUTOFF = 20.0  # Hz

def find_first_csv(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files: raise FileNotFoundError("No CSV detected.")
    return files[0]

def normalize_time(t_raw):
    """Detects units and returns zero-indexed time in seconds."""
    t = t_raw.astype(float)
    dt_median = np.median(np.diff(t))
    scale = 1e-6 if dt_median > 500 else (1e-3 if dt_median > 0.5 else 1.0)
    return (t - t[0]) * scale

def resample_uniform(df, fs=FS_TARGET):
    """
    Interpolates numeric columns to a uniform time base.
    Preserves non-numeric metadata via first-entry broadcasting.
    """
    t_orig = df['timestamp'].values
    t_min, t_max = t_orig[0], t_orig[-1]
    
    # Define uniform time vector
    n_samples = max(2, int(math.ceil((t_max - t_min) * fs)))
    t_uniform = np.linspace(t_min, t_max, n_samples)
    
    resampled_data = {'timestamp': t_uniform}
    
    # Separate numeric and categorical processing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'timestamp': continue
        # Linear interpolation for spectral safety
        resampled_data[col] = np.interp(t_uniform, t_orig, df[col].values)
        
    df_res = pd.DataFrame(resampled_data)
    
    # Handle non-numeric metadata (e.g., 'device', 'label')
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        df_res[col] = df[col].iloc[0]
        
    return df_res

def butter_lowpass(data, cutoff, fs, order=4):
    """Zero-phase Butterworth filter to avoid group delay."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0: return data
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def run_resampling_test(df):
    ts_col = next(c for c in df.columns if c.lower() in ('timestamp','time','t','ts'))
    dev_col = next((c for c in df.columns if c.lower() in ('device','dev')), 'device')
    if 'device' not in df.columns: df['device'] = 'unspecified'

    unique_devs = sorted(df[dev_col].unique())
    
    for dev in unique_devs:
        # 1. Pre-process Time
        d = df[df[dev_col] == dev].copy().reset_index(drop=True)
        d['timestamp'] = normalize_time(d[ts_col].values)
        
        # 2. Resample to uniform FS_TARGET
        d_res = resample_uniform(d, fs=FS_TARGET)
        
        # 3. Apply Low-pass filter to Accel
        acc_cols = [c for c in ['ax', 'ay', 'az'] if c in d_res.columns]
        for col in acc_cols:
            d_res[col] = butter_lowpass(d_res[col].values, LOWPASS_CUTOFF, FS_TARGET)
            
        # 4. Visualization: Raw vs. Processed
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        
        # Raw Jittery Data
        axes[0].plot(d['timestamp'], d[acc_cols], alpha=0.5, linewidth=0.8)
        axes[0].set_title(f"Raw Input (Jittery) - Device {dev}")
        axes[0].set_ylabel("$m/s^2$")
        axes[0].grid(True, alpha=0.2)
        
        # Uniformly Resampled + Filtered
        axes[1].plot(d_res['timestamp'], d_res[acc_cols], linewidth=1.2)
        axes[1].set_title(f"Uniformly Resampled ({FS_TARGET}Hz) + Low-pass ({LOWPASS_CUTOFF}Hz)")
        axes[1].set_ylabel("$m/s^2$")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(acc_cols, loc='upper right')
        axes[1].grid(True, alpha=0.2)
        
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    try:
        csv_path = find_first_csv(DATA_RAW)
        df_full = pd.read_csv(csv_path)
        run_resampling_test(df_full)
    except Exception as e:
        print(f"Resampling failed: {e}")