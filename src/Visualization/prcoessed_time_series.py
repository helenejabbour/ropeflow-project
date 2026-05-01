"""Plot processed IMU time series for a single session and device.

The script visualizes the 6 processed channels for a selected CSV file:
- linear acceleration: ax_w, ay_w, az_w
- angular velocity: gx, gy, gz

It shows the first 30 seconds by default and saves a side-by-side figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "processed" / "20260406_212408_experimental_jo div_device0_processed.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "Visualization"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_csv_path(value: str) -> Path:
	"""Resolve a file path or a processed-file stem to an existing CSV."""
	path = Path(value).expanduser()
	if path.suffix.lower() != ".csv":
		path = path.with_suffix(".csv")
	if not path.is_absolute():
		candidate = (REPO_ROOT / path).resolve()
		if candidate.exists():
			return candidate
		candidate = (REPO_ROOT / "data" / "processed" / path.name).resolve()
		if candidate.exists():
			return candidate
	if path.exists():
		return path.resolve()
	raise FileNotFoundError(f"Could not find processed CSV: {value}")


def load_processed_imu(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	required = ["timestamp_ms", "ax_w", "ay_w", "az_w", "gx", "gy", "gz"]
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	df = df.copy()
	df["time_s"] = (df["timestamp_ms"].astype(float) - float(df["timestamp_ms"].iloc[0])) / 1000.0
	return df


def plot_time_series(df: pd.DataFrame, csv_path: Path, seconds: float, output_dir: Path) -> Path:
	window = df[df["time_s"] <= seconds].copy()
	if window.empty:
		raise ValueError(f"No samples found in the first {seconds} seconds.")

	fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
	colors = {"x": "#e74c3c", "y": "#2ecc71", "z": "#3498db"}
	acc_cols = ("ax_w", "ay_w", "az_w")
	gyro_cols = ("gx", "gy", "gz")
	axis_labels = ("X", "Y", "Z")

	for row, (axis_label, acc_col, gyro_col) in enumerate(zip(axis_labels, acc_cols, gyro_cols)):
		ax_acc = axes[row, 0]
		ax_gyr = axes[row, 1]

		ax_acc.plot(window["time_s"], window[acc_col], color=colors[axis_label.lower()], linewidth=1.2)
		ax_acc.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
		ax_acc.set_ylabel(f"{acc_col}\n(m/s²)")
		ax_acc.set_title(f"Linear acceleration {axis_label}")
		ax_acc.grid(True, alpha=0.3)

		ax_gyr.plot(window["time_s"], window[gyro_col], color=colors[axis_label.lower()], linewidth=1.2)
		ax_gyr.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
		ax_gyr.set_ylabel(f"{gyro_col}\n(deg/s)")
		ax_gyr.set_title(f"Angular velocity {axis_label}")
		ax_gyr.grid(True, alpha=0.3)

	axes[2, 0].set_xlabel("Time (s)")
	axes[2, 1].set_xlabel("Time (s)")
	axes[0, 0].set_title("Linear acceleration X")
	axes[0, 1].set_title("Angular velocity X")

	session_name = csv_path.stem.replace("_device0_processed", "")
	fig.suptitle(f"Processed IMU signals | {session_name} | first {seconds:.0f} s", fontsize=13)
	fig.tight_layout(rect=[0, 0.02, 1, 0.95])

	out_path = output_dir / f"{csv_path.stem}_first_{int(seconds)}s_timeseries.png"
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot processed IMU time series for one processed CSV file.")
	parser.add_argument(
		"--csv",
		default=str(DEFAULT_DATA_FILE),
		help="Path or stem of the processed CSV file to plot.",
	)
	parser.add_argument(
		"--seconds",
		type=float,
		default=30.0,
		help="Number of seconds from the start of the recording to plot.",
	)
	parser.add_argument(
		"--output-dir",
		default=str(DEFAULT_OUTPUT_DIR),
		help="Directory where the figure will be saved.",
	)
	args = parser.parse_args()

	csv_path = resolve_csv_path(args.csv)
	output_dir = Path(args.output_dir).expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	df = load_processed_imu(csv_path)
	out_path = plot_time_series(df, csv_path, args.seconds, output_dir)

	print(f"Loaded: {csv_path}")
	print(f"Saved figure: {out_path}")


if __name__ == "__main__":
	main()
