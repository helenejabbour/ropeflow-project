import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PROCESSED = (SCRIPT_DIR / ".." / ".." / "data" / "processed").resolve()
RESULTS_DIR = (SCRIPT_DIR / ".." / ".." / "results" / "Visualization").resolve()
OUT_PATH = RESULTS_DIR / "tsne_raw_cycles_kmeans.png"

CONFIG = {
  "FS": 50.0,
  "PEAK_PROM_DEGS": 100.0,
  "PEAK_SAVGOL_WINDOW": 21,
  "PEAK_SAVGOL_POLY": 3,
  "PEAK_MIN_DEGS": 50.0,
  "PEAK_MIN_PERIOD_S": 0.5,
  "PEAK_PAIR_MAX_DT_S": 0.25,
  "TARGET_LEN": 64,
}


def load_session(path_d0, path_d1):
  return pd.read_csv(path_d0), pd.read_csv(path_d1)


def extract_signals(df):
  t = df["timestamp_ms"].values / 1000.0
  acc = df[["ax_w", "ay_w", "az_w"]].values
  omega = df[["gx", "gy", "gz"]].values * (np.pi / 180.0)
  return t, acc, omega


def _smooth_mag_deg(omega_rad, cfg):
  mag_deg = np.linalg.norm(omega_rad, axis=1) * (180.0 / np.pi)
  n = len(mag_deg)
  if n < 7:
    return mag_deg

  win = int(cfg.get("PEAK_SAVGOL_WINDOW", 21))
  if win % 2 == 0:
    win += 1
  max_odd = n if (n % 2 == 1) else (n - 1)
  win = max(5, min(win, max_odd))

  poly = int(cfg.get("PEAK_SAVGOL_POLY", 3))
  poly = max(1, min(poly, win - 2))

  y = savgol_filter(mag_deg, window_length=win, polyorder=poly, mode="interp")
  y = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
  return y


def detect_cycle_peaks(omega_rad, fs, cfg):
  mag_smooth = _smooth_mag_deg(omega_rad, cfg)
  if len(mag_smooth) < 7:
    return np.array([], dtype=int), mag_smooth
  peaks, _ = find_peaks(
    mag_smooth,
    distance=max(1, int(cfg["PEAK_MIN_PERIOD_S"] * fs)),
    prominence=cfg["PEAK_PROM_DEGS"],
  )
  peaks = np.array([int(p) for p in peaks if mag_smooth[p] >= cfg["PEAK_MIN_DEGS"]], dtype=int)
  return peaks, mag_smooth


def pair_peaks_same_swing(t0, peaks0, t1, peaks1, max_dt_s):
  if len(peaks0) == 0 or len(peaks1) == 0:
    return []
  used, pairs = set(), []
  t1_peaks = t1[peaks1]
  for p0 in peaks0:
    d = np.abs(t1_peaks - t0[p0])
    for idx in np.argsort(d):
      p1 = int(peaks1[idx])
      if p1 in used:
        continue
      if d[idx] <= max_dt_s:
        used.add(p1)
        pairs.append((int(p0), p1))
        break
  return pairs


def extract_fixed_window(ch6, center_idx, window=64):
  half = window // 2
  start = int(center_idx) - half
  end = start + window
  out = np.zeros((6, window), dtype=np.float32)
  src_lo, src_hi = max(0, start), min(ch6.shape[0], end)
  if src_hi <= src_lo:
    return out
  dst_lo = src_lo - start
  out[:, dst_lo:dst_lo + (src_hi - src_lo)] = ch6[src_lo:src_hi].T
  return out


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

  peaks0, _ = detect_cycle_peaks(om0, CONFIG["FS"], CONFIG)
  peaks1, _ = detect_cycle_peaks(om1, CONFIG["FS"], CONFIG)
  pairs = pair_peaks_same_swing(t0, peaks0, t1, peaks1, CONFIG["PEAK_PAIR_MAX_DT_S"])

  ch0 = np.column_stack([A0, om0 * (180.0 / np.pi)])
  ch1 = np.column_stack([A1, om1 * (180.0 / np.pi)])
  vectors, sessions = [], []
  for p0, p1 in pairs:
    w0 = extract_fixed_window(ch0, p0, CONFIG["TARGET_LEN"])
    w1 = extract_fixed_window(ch1, p1, CONFIG["TARGET_LEN"])
    vectors.append(np.vstack([w0, w1]).reshape(-1))  # (12, 64) -> 768
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

  X = np.vstack(X_list) # (N_cycles, 768)
  session_names = np.array(session_list)

  print(f"\nExtracted cycles: {X.shape[0]}")
  print(f"Feature dimension: {X.shape[1]} (expected 768)")

  if X.shape[0] <= 30:
    raise RuntimeError(
      f"t-SNE perplexity=30 requires >30 samples, but only {X.shape[0]} cycles were found."
    )

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  pca = PCA(n_components=0.95, svd_solver="full")
  X_pca = pca.fit_transform(X_scaled)
  print(
    f"\nPCA reduced: 768 -> {X_pca.shape[1]} "
    f"(explained variance={np.sum(pca.explained_variance_ratio_):.4f})"
  )

  tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
  X_tsne = tsne.fit_transform(X_pca)

  kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
  cluster_labels = kmeans.fit_predict(X_tsne)

  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(12, 9))
  cmap = matplotlib.colormaps["tab20"]
  for k in range(12):
    mask = cluster_labels == k
    if not np.any(mask):
      continue
    color = cmap(k / 11.0) if 11 > 0 else cmap(0.0)
    ax.scatter(
      X_tsne[mask, 0],
      X_tsne[mask, 1],
      s=14,
      alpha=0.8,
      color=color,
      label=f"Cluster {k}",
    )

  ax.set_title("t-SNE of raw cycle matrices (12ch × 64t → 768D → PCA 95% var → t-SNE 2D)")
  ax.set_xlabel("t-SNE 1")
  ax.set_ylabel("t-SNE 2")
  ax.grid(alpha=0.25)
  ax.legend(loc="best", fontsize=8, ncol=2)
  plt.tight_layout()
  fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
  plt.close(fig)

  print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
  main()