import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import cumulative_trapezoid
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
    t_norm = (t - t[0]) * scale
    return t_norm, 1.0 / (dt_median * scale)

def integrate_with_drift_mitigation(acc_world, fs, hp_cutoff=0.1):
    """
    Performs double integration with high-pass filtering to remove 
    low-frequency integration drift.
    """
    dt = 1.0 / fs
    nyquist = 0.5 * fs
    
    # 1. First Integration: Accel -> Velocity
    # Using trapezoidal rule for better O(dt^2) accuracy
    vel = cumulative_trapezoid(acc_world, dx=dt, axis=0, initial=0)
    
    # 2. High-pass filter velocity to stop 'runaway' drift
    # This assumes the net displacement over long periods is zero (rhythmic movement)
    b, a = signal.butter(2, hp_cutoff / nyquist, btype='high')
    vel_hp = signal.filtfilt(b, a, vel, axis=0)
    
    # 3. Second Integration: Velocity -> Position
    pos = cumulative_trapezoid(vel_hp, dx=dt, axis=0, initial=0)
    
    # 4. Optional: High-pass position if baseline drift persists
    pos_hp = signal.filtfilt(b, a, pos, axis=0)
    
    return vel_hp, pos_hp

def run_integration_test(df):
    ts_col = next(c for c in df.columns if c.lower() in ('timestamp','time','t','ts'))
    dev_col = next((c for c in df.columns if c.lower() in ('device','dev')), 'device')
    if 'device' not in df.columns: df['device'] = 'unspecified'

    unique_devs = sorted(df[dev_col].unique())
    
    for dev in unique_devs:
        d = df[df[dev_col] == dev].copy().reset_index(drop=True)
        if len(d) < 100: continue 
        
        t, fs = normalize_time(d[ts_col].values)
        
        # Proxy for World Accel (In production, use the Madgwick-rotated output)
        acc_labels = [c for c in ['ax','ay','az'] if c in d.columns]
        acc_raw = d[acc_labels].values
        
        # Subtract mean (DC Bias removal) is the simplest form of drift mitigation
        acc_centered = acc_raw - np.mean(acc_raw, axis=0)
        
        # Double Integration
        vel, pos = integrate_with_drift_mitigation(acc_centered, fs)

        # Plotting
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        
        # Velocity Plot
        v_labels = ['vx', 'vy', 'vz']
        for i in range(3):
            axes[0].plot(t, vel[:, i], label=v_labels[i], alpha=0.8)
        axes[0].set_title(f"Velocity (High-Pass Filtered) - Device {dev}")
        axes[0].set_ylabel("m/s")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.2)

        # Position Plot
        p_labels = ['px', 'py', 'pz']
        for i in range(3):
            axes[1].plot(t, pos[:, i], label=p_labels[i], alpha=0.8)
        axes[1].set_title("Reconstructed Position (Z-Centered)")
        axes[1].set_ylabel("m")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.2)
        
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    try:
        csv_file = find_first_csv(DATA_RAW)
        df_full = pd.read_csv(csv_file)
        run_integration_test(df_full)
    except Exception as e:
        print(f"Integration test failed: {e}")