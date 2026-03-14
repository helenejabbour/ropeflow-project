import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Project-relative paths
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(ROOT, "data", "raw")

def find_first_csv(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files: raise FileNotFoundError("No CSV detected.")
    return files[0]

def normalize_time(t_raw):
    t = t_raw.astype(float)
    dt_median = np.median(np.diff(t))
    scale = 1e-6 if dt_median > 500 else (1e-3 if dt_median > 0.5 else 1.0)
    return (t - t[0]) * scale, 1.0 / (dt_median * scale)

def quick_bandpass(x, fs, low=0.5, high=5.0):
    nyquist = 0.5 * fs
    high_limit = min(high, nyquist * 0.9)
    b, a = signal.butter(4, [low / nyquist, high_limit / nyquist], btype='band')
    return signal.filtfilt(b, a, x)

def run_adaptive_cycle_analysis(df):
    ts_col = next(c for c in df.columns if c.lower() in ('timestamp','time','t','ts'))
    dev_col = next((c for c in df.columns if c.lower() in ('device','dev')), 'device')
    if 'device' not in df.columns: df['device'] = 'unspecified'

    for dev in sorted(df[dev_col].unique()):
        d = df[df[dev_col] == dev].copy()
        t, fs = normalize_time(d[ts_col].values)
        
        gyr_cols = [c for c in ['gx','gy','gz'] if c in d.columns]
        if not gyr_cols: continue
        gyr_mag = np.linalg.norm(d[gyr_cols].values, axis=1)
        sig = quick_bandpass(gyr_mag - np.mean(gyr_mag), fs)

        # --- ADAPTIVE THRESHOLD LOGIC ---
        # Calculate rolling standard deviation (1-second window)
        window_size = int(fs) 
        rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std().fillna(np.std(sig)).values
        adaptive_height = rolling_std * 0.8  # 80% of local signal intensity

        # Peak detection with adaptive height array
        # min_dist set to 0.35s to prevent harmonic double-counting
        peaks, _ = signal.find_peaks(sig, height=adaptive_height, distance=int(0.35 * fs))
        
        freq = len(peaks) / (t[-1] - t[0])

        # Plotting with the Adaptive Threshold visualization
        plt.figure(figsize=(12, 5))
        plt.plot(t, sig, label='Bandpassed Gyro', color='black', alpha=0.6)
        plt.plot(t, adaptive_height, '--', color='orange', label='Adaptive Threshold', alpha=0.8)
        plt.plot(t[peaks], sig[peaks], 'rx', markersize=8, label=f'Cycles (f={freq:.2f}Hz)')
        
        plt.title(f"Adaptive Cycle Detection: Device {dev}")
        plt.ylabel("Filtered Amplitude")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True, alpha=0.15)

    plt.show()

if __name__ == "__main__":
    try:
        csv_file = find_first_csv(DATA_RAW)
        print(f"Loading full dataset: {csv_file}")
        df_full = pd.read_csv(csv_file, engine='c') 
        
        print(f"Successfully loaded {len(df_full)} rows.")
        run_adaptive_cycle_analysis(df_full)
        
    except Exception as e:
        print(f"Analysis failed: {e}")