import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from PyQt6.QtCore import QUrl, Qt
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    from PyQt6.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PyQt6 is required for this viewer. Install with:\n"
        "  pip install PyQt6"
    ) from exc


CONFIG = {
    "FS": 100.0,             # used for idx/fs dedup — matches validated diagnostic
    "PEAK_PROM_DEGS": 50.0,
    "PEAK_SAVGOL_WINDOW": 15,
    "PEAK_SAVGOL_POLY": 3,
    "PEAK_MIN_DEGS": 50.0,
    "PEAK_MIN_PERIOD_S": 0.2,
    "MERGE_GAP_S": 0.15,
}


def _smooth_mag_deg(omega_rad, cfg):
    mag_deg = np.linalg.norm(omega_rad, axis=1) * (180.0 / np.pi)
    n = len(mag_deg)
    if n < 7:
        return mag_deg

    win = int(cfg.get("PEAK_SAVGOL_WINDOW", 15))
    if win % 2 == 0:
        win += 1
    max_odd = n if (n % 2 == 1) else (n - 1)
    win = max(5, min(win, max_odd))

    poly = int(cfg.get("PEAK_SAVGOL_POLY", 3))
    poly = max(1, min(poly, win - 2))

    return savgol_filter(mag_deg, window_length=win, polyorder=poly, mode="interp")


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


def merge_device_peaks(peaks_d0, peaks_d1, t_d0, t_d1, fs=100.0, gap_s=0.15):
    """
    Union-merge D0/D1 peaks. Deduplication uses peak_idx/fs (matches validated diagnostic).
    Returns (merged_actual_timestamps, merged_sources) where sources are 'D0', 'D1', or 'both'.
    """
    tagged = [(p / fs, t_d0[p], "D0") for p in peaks_d0]
    tagged += [(p / fs, t_d1[p], "D1") for p in peaks_d1]
    if not tagged:
        return np.array([]), []
    tagged.sort(key=lambda x: x[0])
    all_idx_ts    = np.array([x[0] for x in tagged])
    all_actual_ts = np.array([x[1] for x in tagged])
    all_src       = [x[2] for x in tagged]

    accepted = [0]
    for i in range(1, len(all_idx_ts)):
        if all_idx_ts[i] - all_idx_ts[accepted[-1]] > gap_s:
            accepted.append(i)

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


def load_device(processed_csv_path):
    df = pd.read_csv(processed_csv_path)
    t = df["timestamp_ms"].values / 1000.0
    omega = df[["gx", "gy", "gz"]].values * (np.pi / 180.0)
    peaks, mag_smooth = detect_cycle_peaks(omega, CONFIG["FS"], CONFIG)
    mag_raw = np.linalg.norm(omega, axis=1) * (180.0 / np.pi)
    return t, mag_raw, mag_smooth, peaks


def cycle_stats(t, peaks, device_name):
    if len(peaks) < 2:
        return {"device": device_name, "num_peaks": int(len(peaks)),
                "mean_period_s": float("nan"), "std_period_s": float("nan")}
    periods = np.diff(t[peaks])
    return {
        "device": device_name,
        "num_peaks": int(len(peaks)),
        "mean_period_s": float(np.mean(periods)),
        "std_period_s": float(np.std(periods)),
    }


def find_video_path(session_name, preferred_video, search_dir):
    if preferred_video:
        candidate = Path(preferred_video).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Video not found: {candidate}")
        return candidate

    matches = sorted(search_dir.glob(f"{session_name}.*"))
    video_exts = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}
    for path in matches:
        if path.suffix.lower() in video_exts:
            return path.resolve()
    raise FileNotFoundError(
        f"No video found for session '{session_name}' in {search_dir}. "
        "Use --video to provide the file path."
    )


class SyncViewer(QMainWindow):
    def __init__(
        self,
        session_name,
        video_path,
        d0_data,
        d1_data,
        merged_ts,
        merged_sources,
        window_seconds=8.0,
        time_offset_s=0.0,
    ):
        super().__init__()
        self.session_name   = session_name
        self.video_path     = video_path
        self.window_seconds = float(window_seconds)

        self.t0, self.raw0, self.smooth0, self.peaks0 = d0_data
        self.t1, self.raw1, self.smooth1, self.peaks1 = d1_data
        self.merged_ts      = merged_ts
        self.merged_sources = merged_sources

        # Combined signal: interpolate both onto a shared grid, take pointwise max
        t_min = min(self.t0[0], self.t1[0])
        t_max = max(self.t0[-1], self.t1[-1])
        n_pts = max(len(self.t0), len(self.t1))
        self.t_combined = np.linspace(t_min, t_max, n_pts)
        i0 = np.interp(self.t_combined, self.t0, self.smooth0, left=0.0, right=0.0)
        i1 = np.interp(self.t_combined, self.t1, self.smooth1, left=0.0, right=0.0)
        self.combined = np.maximum(i0, i1)

        self.time_offset_box = QDoubleSpinBox()
        self.time_offset_box.setRange(-60.0, 60.0)
        self.time_offset_box.setDecimals(3)
        self.time_offset_box.setSingleStep(0.050)
        self.time_offset_box.setValue(float(time_offset_s))

        self.window_box = QDoubleSpinBox()
        self.window_box.setRange(1.0, 60.0)
        self.window_box.setDecimals(1)
        self.window_box.setSingleStep(0.5)
        self.window_box.setValue(self.window_seconds)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 0)

        self.play_btn  = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")

        n_d0     = len(self.peaks0)
        n_d1     = len(self.peaks1)
        n_merged = len(self.merged_ts)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout(rect=[0, 0, 1, 0.94])
        self.figure.suptitle(
            f"{self.session_name} | D0: {n_d0} peaks, D1: {n_d1} peaks, "
            f"Merged: {n_merged} unique",
            fontsize=10,
        )

        self._build_ui()
        self._draw_static_traces()
        self._connect_signals()
        self.player.setSource(QUrl.fromLocalFile(str(self.video_path)))

        self.setWindowTitle("Video + IMU Time-Series Sync Viewer")
        self.resize(1600, 900)

    def _build_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        info = QLabel(
            f"Session: {self.session_name} | Video: {self.video_path.name}", self
        )
        info.setStyleSheet("font-weight: 600;")
        layout.addWidget(info)

        controls = QHBoxLayout()
        controls.addWidget(self.play_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(QLabel("Timeline:", self))
        controls.addWidget(self.timeline_slider, stretch=1)
        controls.addWidget(QLabel("Window (s):", self))
        controls.addWidget(self.window_box)
        controls.addWidget(QLabel("Time offset (s):", self))
        controls.addWidget(self.time_offset_box)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.video_widget)
        splitter.addWidget(self.canvas)
        splitter.setSizes([800, 800])
        layout.addWidget(splitter, stretch=1)

    def _draw_static_traces(self):
        self.ax.clear()

        # Single combined curve: pointwise max of D0 and D1 smoothed ||omega||
        self.ax.plot(self.t_combined, self.combined, color="#2f4858", lw=1.5,
                     label="max(D0, D1) ||w||")

        # Peak markers on the combined curve
        if len(self.peaks0):
            y0 = np.interp(self.t0[self.peaks0], self.t_combined, self.combined)
            self.ax.scatter(self.t0[self.peaks0], y0,
                            s=18, c="#f28e2b", marker="v", zorder=3, label="D0 peaks")
        if len(self.peaks1):
            y1 = np.interp(self.t1[self.peaks1], self.t_combined, self.combined)
            self.ax.scatter(self.t1[self.peaks1], y1,
                            s=18, c="#4e79a7", marker="v", zorder=3, label="D1 peaks")

        # Merged peaks — vertical dashed lines colored by source
        C      = {"D0": "#17becf", "D1": "#ff7f0e", "both": "#9467bd"}
        counts = {k: sum(1 for s in self.merged_sources if s == k) for k in C}
        labels = {
            "D0":   f"D0-only ({counts['D0']})",
            "D1":   f"D1-only ({counts['D1']})",
            "both": f"Both ({counts['both']})",
        }
        seen = set()
        for ts, src in zip(self.merged_ts, self.merged_sources):
            lbl = labels[src] if src not in seen else None
            self.ax.axvline(ts, color=C[src], ls="--", lw=0.9, alpha=0.7, label=lbl)
            seen.add(src)

        self.ax.set_ylabel("||w|| (deg/s)")
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(alpha=0.25)
        self.ax.legend(loc="upper right", fontsize=7)

        t0_start = float(self.t_combined[0])
        self.cursor = self.ax.axvline(t0_start, color="black", lw=1.2)
        self.canvas.draw_idle()

    def _connect_signals(self):
        self.play_btn.clicked.connect(self.player.play)
        self.pause_btn.clicked.connect(self.player.pause)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.positionChanged.connect(self._on_position_changed)
        self.timeline_slider.sliderMoved.connect(self.player.setPosition)
        self.window_box.valueChanged.connect(self._on_window_changed)

    def _on_duration_changed(self, duration_ms):
        self.timeline_slider.setRange(0, int(duration_ms))

    def _set_axis_window(self, t_now, window_s):
        if len(self.t_combined) == 0:
            return
        t_lo, t_hi = float(self.t_combined[0]), float(self.t_combined[-1])
        half  = 0.5 * window_s
        left  = max(t_lo, t_now - half)
        right = min(t_hi, t_now + half)
        if right - left < window_s:
            if left <= t_lo:
                right = min(t_hi, left + window_s)
            elif right >= t_hi:
                left = max(t_lo, right - window_s)
        if right <= left:
            right = left + 0.1
        self.ax.set_xlim(left, right)

    def _on_position_changed(self, position_ms):
        if not self.timeline_slider.isSliderDown():
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(int(position_ms))
            self.timeline_slider.blockSignals(False)

        t_now    = (position_ms / 1000.0) + float(self.time_offset_box.value())
        window_s = float(self.window_box.value())

        self.cursor.set_xdata([t_now, t_now])
        self._set_axis_window(t_now, window_s)
        self.canvas.draw_idle()

    def _on_window_changed(self, _):
        self._on_position_changed(self.player.position())


def main():
    parser = argparse.ArgumentParser(
        description="Synchronized video + moving-window IMU time-series viewer (PyQt)."
    )
    parser.add_argument(
        "--session",
        default="20260121_160745_we_are_the_champions",
        help="Session stem for processed CSVs and default video lookup.",
    )
    parser.add_argument(
        "--video",
        default="",
        help="Optional explicit video path. If omitted, searches src/Visualization/<session>.*",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=8.0,
        help="Moving window width in seconds for the right-side plots.",
    )
    parser.add_argument(
        "--time-offset-s",
        type=float,
        default=0.0,
        help="Offset added to video time when mapping to IMU timeline (seconds).",
    )
    args = parser.parse_args()

    script_dir    = Path(__file__).resolve().parent
    repo_root     = script_dir.parents[1]
    processed_dir = repo_root / "data" / "processed"

    d0_path = processed_dir / f"{args.session}_device0_processed.csv"
    d1_path = processed_dir / f"{args.session}_device1_processed.csv"
    if not d0_path.exists() or not d1_path.exists():
        raise FileNotFoundError(
            "Processed CSV pair not found:\n"
            f"  {d0_path}\n"
            f"  {d1_path}"
        )

    video_path = find_video_path(args.session, args.video, script_dir)
    d0_data    = load_device(d0_path)
    d1_data    = load_device(d1_path)

    t0, _, _, peaks0 = d0_data
    t1, _, _, peaks1 = d1_data
    merged_ts, merged_sources = merge_device_peaks(
        peaks0, peaks1, t0, t1,
        fs=CONFIG["FS"], gap_s=CONFIG["MERGE_GAP_S"],
    )

    stats_df = pd.DataFrame([
        cycle_stats(t0, peaks0, "D0"),
        cycle_stats(t1, peaks1, "D1"),
    ])
    print("\nCycle statistics")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Merged unique peaks: {len(merged_ts)}")
    print(f"\nVideo: {video_path}")

    app = QApplication(sys.argv)
    viewer = SyncViewer(
        session_name=args.session,
        video_path=video_path,
        d0_data=d0_data,
        d1_data=d1_data,
        merged_ts=merged_ts,
        merged_sources=merged_sources,
        window_seconds=args.window_s,
        time_offset_s=args.time_offset_s,
    )
    viewer.show()
    viewer.player.play()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
