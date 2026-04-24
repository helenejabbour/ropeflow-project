import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from cycle_detection import CONFIG, detect_cycles

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


def load_device(processed_csv_path):
    df = pd.read_csv(processed_csv_path)
    t = df["timestamp_ms"].values / 1000.0
    omega = df[["gx", "gy", "gz"]].values * (np.pi / 180.0)
    cycles, mag_smooth, peaks = detect_cycles(t, omega, fs=CONFIG["FS"])
    mag_raw = np.linalg.norm(omega, axis=1) * (180.0 / np.pi)
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
        window_seconds=8.0,
        time_offset_s=0.0,
    ):
        super().__init__()
        self.session_name = session_name
        self.video_path = video_path
        self.window_seconds = float(window_seconds)

        self.t0, self.raw0, self.smooth0, self.peaks0, self.cycles0 = d0_data
        self.t1, self.raw1, self.smooth1, self.peaks1, self.cycles1 = d1_data

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

        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax0 = self.figure.add_subplot(211)
        self.ax1 = self.figure.add_subplot(212, sharex=self.ax0)
        self.figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.figure.suptitle(
            f"{self.session_name} | D0 cycles={len(self.cycles0)}, D1 cycles={len(self.cycles1)}",
            fontsize=11,
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
            f"Session: {self.session_name} | Video: {self.video_path.name}",
            self,
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

    def _draw_cycles(self, ax, t, cycles, color):
        boundaries = sorted({idx for s, e in cycles for idx in (s, e)})
        for b in boundaries:
            ax.axvline(t[b], color=color, ls="--", lw=0.8, alpha=0.35)

        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if y_max > y_min else 1.0
        y_text = y_max - 0.08 * y_span
        for i, (s, e) in enumerate(cycles, start=1):
            t_mid = 0.5 * (t[s] + t[e])
            duration = t[e] - t[s]
            ax.text(
                t_mid,
                y_text,
                f"C{i} ({duration:.2f}s)",
                fontsize=6,
                ha="center",
                va="top",
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 1.2},
            )

    def _draw_static_traces(self):
        self.ax0.clear()
        self.ax1.clear()

        self.ax0.plot(self.t0, self.raw0, color="#9ecae1", lw=1.0, alpha=0.7, label="D0 raw ||ω||")
        self.ax0.plot(self.t0, self.smooth0, color="#08519c", lw=1.7, label="D0 smoothed ||ω||")
        if len(self.peaks0):
            self.ax0.scatter(self.t0[self.peaks0], self.smooth0[self.peaks0], s=18, c="red", label="D0 peaks")
        self.ax0.set_title("D0 (left hand)")
        self.ax0.set_ylabel("||ω|| (deg/s)")
        self.ax0.grid(alpha=0.25)
        self.ax0.legend(loc="upper right", fontsize=8)

        self.ax1.plot(self.t1, self.raw1, color="#fdae6b", lw=1.0, alpha=0.7, label="D1 raw ||ω||")
        self.ax1.plot(self.t1, self.smooth1, color="#a63603", lw=1.7, label="D1 smoothed ||ω||")
        if len(self.peaks1):
            self.ax1.scatter(self.t1[self.peaks1], self.smooth1[self.peaks1], s=18, c="red", label="D1 peaks")
        self.ax1.set_title("D1 (right hand)")
        self.ax1.set_ylabel("||ω|| (deg/s)")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.grid(alpha=0.25)
        self.ax1.legend(loc="upper right", fontsize=8)

        self._draw_cycles(self.ax0, self.t0, self.cycles0, "#08519c")
        self._draw_cycles(self.ax1, self.t1, self.cycles1, "#a63603")

        self.cursor0 = self.ax0.axvline(self.t0[0], color="black", lw=1.2)
        self.cursor1 = self.ax1.axvline(self.t1[0], color="black", lw=1.2)
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

    def _set_axis_window(self, ax, t, t_now, window_s):
        if len(t) == 0:
            return
        half = 0.5 * window_s
        left = max(float(t[0]), t_now - half)
        right = min(float(t[-1]), t_now + half)

        if right - left < window_s:
            if left <= float(t[0]):
                right = min(float(t[-1]), left + window_s)
            elif right >= float(t[-1]):
                left = max(float(t[0]), right - window_s)
        if right <= left:
            right = left + 0.1

        ax.set_xlim(left, right)

    def _on_position_changed(self, position_ms):
        if not self.timeline_slider.isSliderDown():
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(int(position_ms))
            self.timeline_slider.blockSignals(False)

        t_now = (position_ms / 1000.0) + float(self.time_offset_box.value())
        window_s = float(self.window_box.value())

        self.cursor0.set_xdata([t_now, t_now])
        self.cursor1.set_xdata([t_now, t_now])

        self._set_axis_window(self.ax0, self.t0, t_now, window_s)
        self._set_axis_window(self.ax1, self.t1, t_now, window_s)
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

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
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
    d0_data = load_device(d0_path)
    d1_data = load_device(d1_path)

    stats_df = pd.DataFrame(
        [
            cycle_stats(d0_data[0], d0_data[2], d0_data[3], d0_data[4], "D0"),
            cycle_stats(d1_data[0], d1_data[2], d1_data[3], d1_data[4], "D1"),
        ]
    )
    print("\nCycle statistics")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nVideo: {video_path}")

    app = QApplication(sys.argv)
    viewer = SyncViewer(
        session_name=args.session,
        video_path=video_path,
        d0_data=d0_data,
        d1_data=d1_data,
        window_seconds=args.window_s,
        time_offset_s=args.time_offset_s,
    )
    viewer.show()
    viewer.player.play()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
