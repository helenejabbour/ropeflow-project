import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

matplotlib.use("Agg")


# Same cycle-detection parameters as src/Visualization/cycle_detection.py
CONFIG = {
    "FS": 100.0,
    "CYCLE_PROMINENCE_DEGS": 100.0,
    "CYCLE_SAVGOL_WINDOW": 21,
    "CYCLE_SAVGOL_POLYORDER": 3,
    "CYCLE_MIN_PEAK_DEGS": 200.0,
    "CYCLE_MIN_PERIOD_S": 0.1,
    "CYCLE_MAX_PERIOD_S": 3.0,
    "MIN_CYCLE_SAMPLES": 10,
}


def detect_cycles(t, omega, fs=100.0):
    """
    V10 cycle detector:
    - smoother ||omega|| curve in deg/s
    - rejects low-magnitude peaks
    - 1 peak = 1 cycle = 1 pattern
    """
    mag_deg = np.linalg.norm(omega, axis=1) * (180.0 / np.pi)
    n = len(mag_deg)
    if n < 7:
        return [], mag_deg, np.array([], dtype=int), mag_deg

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
        height=CONFIG["CYCLE_MIN_PEAK_DEGS"],
    )

    if len(peaks) == 0:
        return [], mag_smooth, peaks, mag_deg

    cycles = []
    for i, p in enumerate(peaks):
        left = 0 if i == 0 else (peaks[i - 1] + p) // 2
        right = (len(t) - 1) if i == len(peaks) - 1 else (p + peaks[i + 1]) // 2
        if right <= left:
            continue
        if (right - left) < CONFIG["MIN_CYCLE_SAMPLES"]:
            continue
        cycles.append((left, right))

    return cycles, mag_smooth, peaks, mag_deg


def load_device(processed_csv_path):
    df = pd.read_csv(processed_csv_path)
    t = df["timestamp_ms"].values / 1000.0
    omega = df[["gx", "gy", "gz"]].values * (np.pi / 180.0)
    cycles, mag_smooth, peaks, mag_raw = detect_cycles(t, omega, fs=CONFIG["FS"])
    return t, mag_raw, mag_smooth, peaks, cycles


def cycle_stats(t, mag_smooth, peaks, cycles, device_name):
    periods = np.array([t[e] - t[s] for s, e in cycles], dtype=float)
    peak_vals = mag_smooth[peaks] if len(peaks) else np.array([], dtype=float)
    return {
        "device": device_name,
        "num_cycles": int(len(cycles)),
        "mean_period_s": float(np.mean(periods)) if len(periods) else np.nan,
        "std_period_s": float(np.std(periods)) if len(periods) else np.nan,
        "mean_peak_omega_deg_s": float(np.mean(peak_vals)) if len(peak_vals) else np.nan,
    }


def draw_cycle_annotations(ax, t, mag_smooth, cycles, color):
    boundaries = sorted({idx for s, e in cycles for idx in (s, e)})
    for b in boundaries:
        ax.axvline(t[b], color=color, ls="--", lw=0.9, alpha=0.55)

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    y_text = y_max - 0.08 * y_span

    for i, (s, e) in enumerate(cycles, start=1):
        t_mid = 0.5 * (t[s] + t[e])
        duration = t[e] - t[s]
        label = f"C{i} ({duration:.2f}s)"
        ax.text(
            t_mid,
            y_text,
            label,
            fontsize=7,
            ha="center",
            va="top",
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 1.5},
        )


def plot_session(session_name, processed_dir, output_path):
    d0_path = processed_dir / f"{session_name}_device0_processed.csv"
    d1_path = processed_dir / f"{session_name}_device1_processed.csv"
    if not d0_path.exists() or not d1_path.exists():
        raise FileNotFoundError(
            "Processed CSV pair not found:\n"
            f"  {d0_path}\n"
            f"  {d1_path}"
        )

    t0, raw0, smooth0, peaks0, cycles0 = load_device(d0_path)
    t1, raw1, smooth1, peaks1, cycles1 = load_device(d1_path)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    ax0, ax1, ax2 = axes

    # Top: D0
    ax0.plot(t0, raw0, color="#9ecae1", lw=1.0, alpha=0.7, label="D0 raw ||ω||")
    ax0.plot(t0, smooth0, color="#08519c", lw=1.8, label="D0 smoothed ||ω||")
    if len(peaks0):
        ax0.scatter(t0[peaks0], smooth0[peaks0], color="red", s=20, zorder=3, label="D0 peaks")
    draw_cycle_annotations(ax0, t0, smooth0, cycles0, color="#08519c")
    ax0.set_ylabel("||ω|| (deg/s)")
    ax0.set_title("D0 (left hand)")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8)

    # Middle: D1
    ax1.plot(t1, raw1, color="#fdae6b", lw=1.0, alpha=0.7, label="D1 raw ||ω||")
    ax1.plot(t1, smooth1, color="#a63603", lw=1.8, label="D1 smoothed ||ω||")
    if len(peaks1):
        ax1.scatter(t1[peaks1], smooth1[peaks1], color="red", s=20, zorder=3, label="D1 peaks")
    draw_cycle_annotations(ax1, t1, smooth1, cycles1, color="#a63603")
    ax1.set_ylabel("||ω|| (deg/s)")
    ax1.set_title("D1 (right hand)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    # Bottom placeholder
    ax2.set_title("Video frame at time t (insert manually)")
    ax2.text(
        0.5,
        0.5,
        "Video frame at time t\n(insert manually)",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
    )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_frame_on(True)

    fig.suptitle(
        f"{session_name} | detected cycles: D0={len(cycles0)}, D1={len(cycles1)}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    stats_rows = [
        cycle_stats(t0, smooth0, peaks0, cycles0, "D0"),
        cycle_stats(t1, smooth1, peaks1, cycles1, "D1"),
    ]
    stats_df = pd.DataFrame(stats_rows)
    print("\nCycle statistics")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Time-series + video placeholder visualization for one session.")
    parser.add_argument(
        "--session",
        default="20260121_160745_we_are_the_champions",
        help="Session stem used in processed CSV names.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    processed_dir = repo_root / "data" / "processed"
    output_path = repo_root / "results" / "Visualization" / f"timeseries_detail_{args.session}.png"

    plot_session(args.session, processed_dir, output_path)


if __name__ == "__main__":
    main()
