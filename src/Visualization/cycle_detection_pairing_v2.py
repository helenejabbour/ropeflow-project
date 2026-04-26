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
    "CYCLE_PROMINENCE_DEGS": 50.0,
    "CYCLE_SAVGOL_WINDOW": 15,
    "CYCLE_SAVGOL_POLYORDER": 3,
    "CYCLE_MIN_PEAK_DEGS": 50.0,
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


def merge_device_peaks(peaks_d0, peaks_d1, t_d0, t_d1, fs, gap_s=0.15):
    """
    Merge D0 and D1 peak indices into a single unified timeline.
    Deduplication uses peak_idx/fs for comparison (matches the validated diagnostic),
    but returns actual timestamps (t_d0/t_d1) for correct plot positioning.
    Returns (merged_actual_timestamps, merged_sources) where sources are 'D0', 'D1', or 'both'.
    """
    # (idx_time_for_dedup, actual_timestamp, source)
    tagged = [(p / fs, t_d0[p], "D0") for p in peaks_d0]
    tagged += [(p / fs, t_d1[p], "D1") for p in peaks_d1]
    if not tagged:
        return np.array([]), []

    tagged.sort(key=lambda x: x[0])
    all_idx_ts     = np.array([x[0] for x in tagged])
    all_actual_ts  = np.array([x[1] for x in tagged])
    all_src        = [x[2] for x in tagged]

    # Exact validated deduplication: compare idx/fs times, matching diagnostic
    accepted = [0]
    for i in range(1, len(all_idx_ts)):
        if all_idx_ts[i] - all_idx_ts[accepted[-1]] > gap_s:
            accepted.append(i)

    # Assign every candidate to its group, union sources
    group_sources = [set() for _ in accepted]
    a_idx = 0
    for i in range(len(all_idx_ts)):
        if a_idx + 1 < len(accepted) and i >= accepted[a_idx + 1]:
            a_idx += 1
        group_sources[a_idx].add(all_src[i])

    merged_ts = all_actual_ts[accepted]
    merged_sources = [
        "both" if len(s) > 1 else next(iter(s))
        for s in group_sources
    ]
    return merged_ts, merged_sources


def plot_merged_cycles(stem, t_d0, mag_d0, peaks_d0, t_d1, mag_d1, peaks_d1, merged_ts, merged_sources):
    n_d0 = len(peaks_d0)
    n_d1 = len(peaks_d1)
    n_merged = len(merged_ts)

    C_D0_LINE = "#f28e2b"
    C_D1_LINE = "#4e79a7"
    C_PEAK_MARKER = "#2f4858"
    C_TEAL = "#17becf"    # D0-only vertical
    C_ORANGE = "#ff7f0e"  # D1-only vertical
    C_PURPLE = "#9467bd"  # both vertical

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Top panel: D0
    ax0 = axes[0]
    ax0.plot(t_d0, mag_d0, color=C_D0_LINE, lw=1.4, label=r"D0 $||\omega||$ smoothed")
    if n_d0 > 0:
        ax0.scatter(t_d0[peaks_d0], mag_d0[peaks_d0], marker="v", s=58,
                    color=C_PEAK_MARKER, zorder=3, label=f"{n_d0} peaks")
    ax0.set_ylabel("Magnitude (deg/s)")
    ax0.legend(loc="upper right")
    ax0.grid(alpha=0.25)

    # Middle panel: D1
    ax1 = axes[1]
    ax1.plot(t_d1, mag_d1, color=C_D1_LINE, lw=1.4, label=r"D1 $||\omega||$ smoothed")
    if n_d1 > 0:
        ax1.scatter(t_d1[peaks_d1], mag_d1[peaks_d1], marker="v", s=58,
                    color=C_PEAK_MARKER, zorder=3, label=f"{n_d1} peaks")
    ax1.set_ylabel("Magnitude (deg/s)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.25)

    # Bottom panel: merged timeline
    ax2 = axes[2]
    ax2.plot(t_d0, mag_d0, color=C_D0_LINE, lw=1.2, alpha=0.7, label="D0")
    ax2.plot(t_d1, mag_d1, color=C_D1_LINE, lw=1.2, alpha=0.7, label="D1")

    n_d0_only = sum(1 for s in merged_sources if s == "D0")
    n_d1_only = sum(1 for s in merged_sources if s == "D1")
    n_both    = sum(1 for s in merged_sources if s == "both")

    color_map = {"D0": C_TEAL, "D1": C_ORANGE, "both": C_PURPLE}
    label_map = {
        "D0":   f"D0-only ({n_d0_only})",
        "D1":   f"D1-only ({n_d1_only})",
        "both": f"Both ({n_both})",
    }
    seen_src = set()
    for ts, src in zip(merged_ts, merged_sources):
        lbl = label_map[src] if src not in seen_src else None
        ax2.axvline(ts, color=color_map[src], ls="--", lw=1.0, alpha=0.85, label=lbl)
        seen_src.add(src)

    ax2.set_ylabel("Magnitude (deg/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        f"{stem} | D0: {n_d0} peaks, D1: {n_d1} peaks, Merged: {n_merged} unique peaks",
        fontsize=11,
    )
    plt.tight_layout()

    out = OUTPUT_DIR / f"fig_merged_cycles_{stem}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = discover_processed_pairs(str(DATA_PROCESSED))
    if not pairs:
        print("No processed device0/device1 pairs found. Skipping cycle-detection plots.")
        return

    print(f"Generating cycle-detection graphs for {len(pairs)} processed pairs...")
    summary = []
    for d0_path, d1_path, stem in pairs:
        # Load both devices
        df0 = pd.read_csv(d0_path)
        t_d0 = df0["timestamp_ms"].values / 1000.0
        omega_d0 = df0[["gx", "gy", "gz"]].values * (np.pi / 180.0)
        _, mag_d0, peaks_d0 = detect_cycles(t_d0, omega_d0, fs=CONFIG["FS"])

        df1 = pd.read_csv(d1_path)
        t_d1 = df1["timestamp_ms"].values / 1000.0
        omega_d1 = df1[["gx", "gy", "gz"]].values * (np.pi / 180.0)
        _, mag_d1, peaks_d1 = detect_cycles(t_d1, omega_d1, fs=CONFIG["FS"])

        merged_ts, merged_sources = merge_device_peaks(
            peaks_d0, peaks_d1, t_d0, t_d1, CONFIG["FS"]
        )
        n_merged = len(merged_ts)
        print(f"  D0 peaks: {len(peaks_d0)}, D1 peaks: {len(peaks_d1)}, Merged unique: {n_merged}")

        plot_merged_cycles(
            stem, t_d0, mag_d0, peaks_d0,
            t_d1, mag_d1, peaks_d1,
            merged_ts, merged_sources,
        )
        summary.append((stem, len(peaks_d0), len(peaks_d1), n_merged))

    # Summary table
    col_w = 45
    sep = "-" * (col_w + 24)
    print(f"\n{'Session':<{col_w}} {'D0':>6} {'D1':>6} {'Merged':>8}")
    print(sep)
    total_d0 = total_d1 = total_merged = 0
    for sess, n0, n1, nm in summary:
        print(f"{sess:<{col_w}} {n0:>6} {n1:>6} {nm:>8}")
        total_d0 += n0
        total_d1 += n1
        total_merged += nm
    print(sep)
    print(f"{'TOTAL':<{col_w}} {total_d0:>6} {total_d1:>6} {total_merged:>8}")


if __name__ == "__main__":
    main()
