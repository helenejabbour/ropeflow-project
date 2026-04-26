import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUT_DIR = REPO_ROOT / "results" / "Visualization"

CONFIG = {
    "FS": 100.0,
    "CYCLE_PROMINENCE_DEGS": 30.0,
    "CYCLE_SAVGOL_WINDOW": 15,
    "CYCLE_SAVGOL_POLYORDER": 3,
    "CYCLE_MIN_PEAK_DEGS": 0.0,
    "CYCLE_MIN_PERIOD_S": 0.2,
    "CYCLE_MAX_PERIOD_S": 2.0,
    "MIN_CYCLE_SAMPLES": 10,
}


def discover_processed_pairs(processed_dir):
    d0_files = sorted(glob.glob(os.path.join(processed_dir, "*_device0_processed.csv")))
    pairs = []
    for d0_path in d0_files:
        d1_path = d0_path.replace("_device0_", "_device1_")
        if not os.path.isfile(d1_path):
            continue
        stem = os.path.basename(d0_path).replace("_device0_processed.csv", "")
        pairs.append((d0_path, d1_path, stem))
    return pairs


def detect_cycles(t, omega, fs=50.0):
    """
    V10 cycle detector:
    - smoother ||omega|| curve in deg/s
    - rejects low-magnitude peaks
    - 1 peak = 1 cycle = 1 pattern
    """
    mag_deg = np.linalg.norm(omega, axis=1) * (180.0 / np.pi)
    n = len(mag_deg)
    if n < 7:
        return [], mag_deg, np.array([], dtype=int)

    win = int(CONFIG.get("CYCLE_SAVGOL_WINDOW", 21))
    if win % 2 == 0:
        win += 1
    max_odd = n if (n % 2 == 1) else (n - 1)
    win = max(5, min(win, max_odd))

    poly = int(CONFIG.get("CYCLE_SAVGOL_POLYORDER", 3))
    poly = max(1, min(poly, win - 2))

    mag_smooth = savgol_filter(mag_deg, window_length=win, polyorder=poly, mode="interp")
    mag_smooth = savgol_filter(mag_smooth, window_length=win, polyorder=poly, mode="interp")

    peaks, _ = find_peaks(
        mag_smooth,
        distance=max(1, int(CONFIG["CYCLE_MIN_PERIOD_S"] * fs)),
        prominence=CONFIG["CYCLE_PROMINENCE_DEGS"],
    )
    peaks = np.array([int(p) for p in peaks if mag_smooth[p] >= CONFIG["CYCLE_MIN_PEAK_DEGS"]], dtype=int)

    if len(peaks) == 0:
        return [], mag_smooth, peaks

    cycles = []
    for i, p in enumerate(peaks):
        left = 0 if i == 0 else (peaks[i - 1] + p) // 2
        right = (len(t) - 1) if i == len(peaks) - 1 else (p + peaks[i + 1]) // 2
        if right <= left:
            continue
        if (right - left) < CONFIG["MIN_CYCLE_SAMPLES"]:
            continue
        cycles.append((left, right))

    return cycles, mag_smooth, peaks


def plot_cycle_detection(csv_path, entry_name, device=0, save_name=None):
    path = csv_path

    df = pd.read_csv(path)
    t = df["timestamp_ms"].values / 1000.0
    omega = df[["gx", "gy", "gz"]].values * (np.pi / 180.0)
    cycles, mag_smooth, peaks = detect_cycles(t, omega, fs=CONFIG["FS"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, mag_smooth, color="#f28e2b", lw=1.4, label=r"$||\omega||$ smoothed")
    if len(peaks) > 0:
        ax.scatter(
            t[peaks],
            mag_smooth[peaks],
            marker="v",
            s=58,
            color="#2f4858",
            label=f"{len(peaks)} cycles",
        )
    ax.axhline(
        CONFIG["CYCLE_MIN_PEAK_DEGS"],
        color="#666666",
        ls="--",
        lw=1.1,
        label=f"min peak = {CONFIG['CYCLE_MIN_PEAK_DEGS']:.0f} deg/s",
    )

    ax.set_title(f"Rope-flow cycle detection | {entry_name} | device{device}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude (deg/s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_name is None:
        safe_name = entry_name.replace("/", "_").replace("\\", "_")
        save_name = f"fig_cycle_detection_{safe_name}_device{device}.png"
    out = OUTPUT_DIR / save_name
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    return {
        "entry": entry_name,
        "device": int(device),
        "n_peaks": int(len(peaks)),
        "n_cycles": int(len(cycles)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = discover_processed_pairs(str(DATA_PROCESSED))
    if not pairs:
        print("No processed device0/device1 pairs found. Skipping cycle-detection plots.")
        return

    print(f"Generating cycle-detection graphs for {len(pairs)} processed pairs...")
    for d0_path, d1_path, stem in pairs:
        diag_d0 = plot_cycle_detection(d0_path, stem, device=0)
        diag_d1 = plot_cycle_detection(d1_path, stem, device=1)
        print(diag_d0)
        print(diag_d1)


if __name__ == "__main__":
    main()
