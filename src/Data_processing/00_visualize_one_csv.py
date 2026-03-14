# src/visualize_by_device.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(ROOT, "data", "raw")
OUT_FOLDER = os.path.join(ROOT, "results")
os.makedirs(OUT_FOLDER, exist_ok=True)

# If your CSVs are very large, set MAX_ROWS to None to read full file.
MAX_ROWS = 20000  # set to None to read entire file

def find_timestamp_column(df):
    for name in ['timestamp','time','t','ts','Time','Timestamp']:
        if name in df.columns:
            return name
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            return c
    return None

def find_imu_columns(df):
    lower = {c.lower(): c for c in df.columns}
    mapping = {k: lower.get(k, None) for k in ['ax','ay','az','gx','gy','gz']}
    return mapping

def find_device_column(df):
    for name in ['device','dev','id','device_id']:
        if name in df.columns:
            return name
    # try any column named like device in lowercase
    for c in df.columns:
        if c.lower() == 'device':
            return c
    return None

def load_first_csv_recursive(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return files[0]

def read_and_prepare(path, max_rows=MAX_ROWS):
    df = pd.read_csv(path, nrows=max_rows)
    ts_col = find_timestamp_column(df)
    if ts_col is None:
        raise ValueError("No timestamp column found")
    t = df[ts_col].astype(float).values
    # detect ms vs s
    if len(t) > 1:
        dt = np.median(np.diff(t))
        if dt < 0.01:
            t = t * 1e-3
    df['timestamp'] = t
    return df

def plot_device(df_device, device_value, out_folder, src_name):
    cols = find_imu_columns(df_device)
    t = df_device['timestamp'].values
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # accelerometer
    if all(cols[k] is not None for k in ['ax','ay','az']):
        axes[0].plot(t, df_device[cols['ax']], label='ax')
        axes[0].plot(t, df_device[cols['ay']], label='ay')
        axes[0].plot(t, df_device[cols['az']], label='az')
        axes[0].set_ylabel('Acceleration (raw)')
        axes[0].legend()
        axes[0].grid(True)
    else:
        axes[0].text(0.5, 0.5, "No accelerometer columns found", ha='center', va='center')
    # gyroscope
    if all(cols[k] is not None for k in ['gx','gy','gz']):
        axes[1].plot(t, df_device[cols['gx']], label='gx')
        axes[1].plot(t, df_device[cols['gy']], label='gy')
        axes[1].plot(t, df_device[cols['gz']], label='gz')
        axes[1].set_ylabel('Gyro (raw)')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, "No gyroscope columns found", ha='center', va='center')
    axes[1].set_xlabel('Time (s)')
    title = f"IMU signals device={device_value} from {src_name}"
    fig.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.96])
    fname = f"imu_device_{device_value}.png"
    out_path = os.path.join(out_folder, fname)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main():
    csv_path = load_first_csv_recursive(DATA_RAW)
    print("Using CSV:", csv_path)
    df = read_and_prepare(csv_path)
    device_col = find_device_column(df)
    if device_col is None:
        print("No device column found. Plotting entire file as device=all.")
        out = plot_device(df, "all", OUT_FOLDER, os.path.basename(csv_path))
        print("Saved:", out)
        return
    # ensure device values are simple strings
    df[device_col] = df[device_col].astype(str)
    unique_devices = sorted(df[device_col].unique())
    saved_paths = []
    for dev in unique_devices:
        df_dev = df[df[device_col] == dev].reset_index(drop=True)
        if df_dev.empty:
            continue
        out = plot_device(df_dev, dev, OUT_FOLDER, os.path.basename(csv_path))
        saved_paths.append(out)
        print("Saved:", out)
    if not saved_paths:
        print("No device-specific plots created.")
    else:
        print("All plots saved to:", OUT_FOLDER)

if __name__ == "__main__":
    main()