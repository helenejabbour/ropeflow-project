import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

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

def rotate_vector_by_quaternion(v, q):
    """
    Vectorized rotation of a 3D vector v by a quaternion q [w, x, y, z].
    Utilizes the formula: v' = v + 2w(qv x v) + 2(qv x (qv x v))
    """
    w = q[:, 0]
    qv = q[:, 1:]
    cross1 = np.cross(qv, v)
    cross2 = np.cross(qv, cross1)
    return v + 2 * w[:, np.newaxis] * cross1 + 2 * cross2

def run_madgwick_test(df):
    ts_col = next(c for c in df.columns if c.lower() in ('timestamp','time','t','ts'))
    dev_col = next((c for c in df.columns if c.lower() in ('device','dev')), 'device')
    if 'device' not in df.columns: df['device'] = 'unspecified'

    unique_devs = sorted(df[dev_col].unique())
    
    for dev in unique_devs:
        d = df[df[dev_col] == dev].copy().reset_index(drop=True)
        if len(d) < 10: continue 
        
        t, fs = normalize_time(d[ts_col].values)
        
        # Explicit column mapping to ensure legend accuracy
        acc_labels = [c for c in ['ax','ay','az'] if c in d.columns]
        gyr_labels = [c for c in ['gx','gy','gz'] if c in d.columns]
        
        acc = d[acc_labels].values
        gyr = d[gyr_labels].values
        
        # Check for deg/s vs rad/s (Madgwick requires rad/s)
        if np.nanmedian(np.abs(gyr)) > 5:
            gyr = np.radians(gyr)

        # Madgwick filter is O(N) sequential; optimized loop
        mad = Madgwick(beta=0.04, sampleperiod=1.0/fs)
        quats = np.zeros((len(d), 4))
        q = np.array([1.0, 0.0, 0.0, 0.0])

        print(f"Filtering Device {dev}...")
        for i in range(len(d)):
            q = mad.updateIMU(q, gyr=gyr[i], acc=acc[i])
            quats[i] = q

        # Gravity vector in m/s^2 (Standard Earth Gravity)
        # Ensure your raw 'acc' units match this scale. 
        g_earth = np.array([0, 0, 9.80665]) 
        acc_world = rotate_vector_by_quaternion(acc, quats) - g_earth

        # --- PLOTTING ---
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        
        # Subplot 1: Body Frame
        for i, label in enumerate(acc_labels):
            axes[0].plot(t, acc[:, i], label=label, alpha=0.8, linewidth=1.2)
        axes[0].set_title(f"Body-Frame Acceleration: Device {dev}")
        axes[0].set_ylabel("$m/s^2$")
        axes[0].legend(loc='upper right', frameon=True)
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Subplot 2: World Frame
        w_labels = ['wx', 'wy', 'wz']
        for i, label in enumerate(w_labels):
            axes[1].plot(t, acc_world[:, i], label=label, alpha=0.8, linewidth=1.2)
        axes[1].set_title("World-Frame Linear Acceleration (Gravity Subtracted)")
        axes[1].set_ylabel("$m/s^2$")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(loc='upper right', frameon=True)
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    try:
        csv_file = find_first_csv(DATA_RAW)
        print(f"Ingesting full file: {csv_file}")
        df_full = pd.read_csv(csv_file)
        run_madgwick_test(df_full)
    except Exception as e:
        print(f"Error in Madgwick pipeline: {e}")