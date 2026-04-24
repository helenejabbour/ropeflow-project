import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PROCESSED = (SCRIPT_DIR / ".." / ".." / "data" / "processed").resolve()
RESULTS_DIR = (SCRIPT_DIR / ".." / ".." / "results" / "Visualization").resolve()
OUT_PATH = RESULTS_DIR / "tsne_raw_cycles_hdbscan.png"

CONFIG = {
    "FS": 50.0,
    "CYCLE_PROMINENCE_DEGS": 100.0,
    "CYCLE_SAVGOL_WINDOW": 21,
    "CYCLE_SAVGOL_POLYORDER": 3,
    "CYCLE_MIN_PEAK_DEGS": 300.0,
    "CYCLE_MIN_PERIOD_S": 0.5,
    "CYCLE_MAX_PERIOD_S": 3.0,
    "TARGET_LEN": 64,
    "MIN_CYCLE_SAMPLES": 10,
}


def load_session(path_d0, path_d1):
    return pd.read_csv(path_d0), pd.read_csv(path_d1)


def extract_signals(df):
    t = df["timestamp_ms"].values / 1000.0
    acc = df[["ax_w", "ay_w", "az_w"]].values
    omega = df[["gx", "gy", "gz"]].values * (np.pi / 180.0)
    return t, acc, omega


def detect_cycles(t, omega, fs=50.0):
    """
    Same cycle logic as cycle_detection.py / V10:
    - two-pass Savitzky-Golay on ||omega||
    - peak thresholds in deg/s
    - 1 peak = 1 cycle with midpoint boundaries
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
        height=CONFIG["CYCLE_MIN_PEAK_DEGS"],
    )

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


def pair_cycles(t0, cyc0, t1, cyc1):
    paired0, paired1, used = [], [], set()
    for c0 in cyc0:
        best_i, best_overlap = -1, -1.0
        for i, c1 in enumerate(cyc1):
            if i in used:
                continue
            overlap = max(0.0, min(t0[c0[1]], t1[c1[1]]) - max(t0[c0[0]], t1[c1[0]]))
            if overlap > best_overlap:
                best_overlap, best_i = overlap, i
        if best_i >= 0 and best_overlap > 0:
            paired0.append(c0)
            paired1.append(cyc1[best_i])
            used.add(best_i)
    return paired0, paired1


def resample_cycle(signal, target_len):
    n = len(signal)
    if n < 2:
        if signal.ndim == 1:
            return np.zeros(target_len)
        return np.zeros((target_len, signal.shape[1]))
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    if signal.ndim == 1:
        return np.interp(x_new, x_old, signal)
    return np.column_stack([np.interp(x_new, x_old, signal[:, j]) for j in range(signal.shape[1])])


def build_cycle_matrix(A0, A1, om0, om1, s0, e0, s1, e1, target_len=64):
    state0 = np.column_stack([A0[s0:e0], om0[s0:e0]])  # (N0, 6)
    state1 = np.column_stack([A1[s1:e1], om1[s1:e1]])  # (N1, 6)
    r0 = resample_cycle(state0, target_len)
    r1 = resample_cycle(state1, target_len)
    # [D0(6ch), D1(6ch)] x 64 => (12, 64)
    return np.column_stack([r0, r1]).T


def discover_processed_pairs(processed_dir):
    csv_files = sorted(glob.glob(os.path.join(processed_dir, "*_device0_processed.csv")))
    pairs = []
    for d0_path in csv_files:
        d1_path = d0_path.replace("_device0_", "_device1_")
        if not os.path.exists(d1_path):
            continue
        session_name = os.path.basename(d0_path).replace("_device0_processed.csv", "")
        pairs.append((d0_path, d1_path, session_name))
    return pairs


def process_entry(entry):
    d0, d1, session_name = entry
    df0, df1 = load_session(d0, d1)
    t0, A0, om0 = extract_signals(df0)
    t1, A1, om1 = extract_signals(df1)

    cyc0, _, _ = detect_cycles(t0, om0, fs=CONFIG["FS"])
    cyc1, _, _ = detect_cycles(t1, om1, fs=CONFIG["FS"])
    p0, p1 = pair_cycles(t0, cyc0, t1, cyc1)

    vectors = []
    sessions = []
    for (s0, e0), (s1, e1) in zip(p0, p1):
        cm = build_cycle_matrix(A0, A1, om0, om1, s0, e0, s1, e1, CONFIG["TARGET_LEN"])
        vectors.append(cm.reshape(-1))  # 12*64 = 768
        sessions.append(session_name)
    return vectors, sessions

def main():
    print(f"DATA_PROCESSED: {DATA_PROCESSED}")

    sessions = discover_processed_pairs(str(DATA_PROCESSED))
    if not sessions:
        raise RuntimeError("No valid processed D0/D1 session pairs discovered.")

    X_list = []
    session_list = []
    for entry in sessions:
        vecs, s_names = process_entry(entry)
        if vecs:
            X_list.extend(vecs)
            session_list.extend(s_names)

    if not X_list:
        raise RuntimeError("No paired cycles extracted from discovered sessions.")

    X = np.vstack(X_list)  # (N_cycles, 768)
    session_names = np.array(session_list)

    print(f"\nExtracted cycles: {X.shape[0]}")
    print(f"Feature dimension: {X.shape[1]} (expected 768)")

    if X.shape[0] <= 30:
        raise RuntimeError(
            f"t-SNE perplexity=20 requires >30 samples, but only {X.shape[0]} cycles were found."
        )

    # 1. Z-Score Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Dimensionality Reduction (PCA)
    pca = PCA(n_components=0.95, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    print(
        f"\nPCA reduced: 768 -> {X_pca.shape[1]} "
        f"(explained variance={np.sum(pca.explained_variance_ratio_):.4f})"
    )

    # 3. Non-linear Manifold Projection (t-SNE)
    tsne = TSNE(n_components=2, perplexity=20, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_pca)

    # 4. Density-Based Clustering (HDBSCAN)
    # min_cluster_size dictates the smallest allowable grouping to be considered a distinct pattern
    clusterer = HDBSCAN(min_cluster_size=25, min_samples=5)
    cluster_labels = clusterer.fit_predict(X_tsne)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nHDBSCAN Results:")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise} ({(n_noise/len(cluster_labels))*100:.1f}%)")

    # 5. Visualization Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Isolate noise and valid clusters
    unique_labels = set(cluster_labels)
    cmap = matplotlib.colormaps.get_cmap("tab20")
    
    # Plot Noise (-1) first so it sits in the background
    if -1 in unique_labels:
        mask = cluster_labels == -1
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            s=10, alpha=0.3, color="gray", marker="x",
            label=f"Noise/Outliers ({n_noise})"
        )

    # Plot Discovered Clusters
    for k in unique_labels:
        if k == -1:
            continue
        mask = cluster_labels == k
        color = cmap(k / max(1, n_clusters))
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            s=20, alpha=0.85, color=color,
            label=f"Pattern {k} (n={np.sum(mask)})"
        )

    ax.set_title(f"Density-Based Structural Inference (Discovered: {n_clusters} Patterns)")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()