import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(ROOT, "data", "raw")

def find_first_csv(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError("No CSV detected in data/raw/")
    return files[0]

def find_timestamp_column(df):
    candidates = ['timestamp','time','t','ts','Time','Timestamp']
    for name in candidates:
        if name in df.columns: return name
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[0] if not numeric_cols.empty else None

def normalize_time(t_raw):
    t = t_raw.astype(float)
    dt_median = np.median(np.diff(t))
    
    if dt_median > 500:     scale = 1e-6 # us to s
    elif dt_median > 0.5:   scale = 1e-3 # ms to s
    else:                   scale = 1.0  # already s
    
    return (t - t[0]) * scale

def plot_per_device(df, ts_col):
    df['t_norm'] = normalize_time(df[ts_col].values)
    
    dev_col = next((c for c in df.columns if c.lower() in ['device', 'dev']), None)
    if dev_col is None:
        df['dev_temp'] = 'unspecified'
        dev_col = 'dev_temp'

    unique_devs = df[dev_col].unique()
    
    for dev in unique_devs:
        # Create a specific figure for this device
        d = df[df[dev_col] == dev]
        t = d['t_norm'].values
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Accel Plot
        acc_cols = [c for c in ['ax', 'ay', 'az'] if c in d.columns]
        for col in acc_cols:
            axes[0].plot(t, d[col], label=col, linewidth=1)
        axes[0].set_ylabel('Accel (Raw)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Gyro Plot
        gyro_cols = [c for c in ['gx', 'gy', 'gz'] if c in d.columns]
        for col in gyro_cols:
            axes[1].plot(t, d[col], label=col, linewidth=1)
        axes[1].set_ylabel('Gyro (Raw)')
        axes[1].set_xlabel('Time (s)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Raw Data Test: Device {dev}")
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    try:
        csv_path = find_first_csv(DATA_RAW)
        print(f"Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        
        ts_col = find_timestamp_column(df)
        if not ts_col:
            raise ValueError("Could not identify timestamp column.")
            
        plot_per_device(df, ts_col)
        
    except Exception as e:
        print(f"Error during execution: {e}")