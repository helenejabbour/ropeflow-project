# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase for **rope flow motion analysis** using dual M5Stick C Plus IMUs (wrist-mounted, ~98 Hz) and synchronized video. The pipeline extracts clean motion dynamics, discovers governing equations, reconstructs 3D trajectories, and classifies rope flow patterns.

## Architecture: two-stage pipeline

### Stage 1 — IMU preprocessing (`src/Data_processing/`)

The canonical pipeline is **`06_Full_pipeline_onesample_v6.py`** (latest). Earlier scripts (`00_`–`05_`, `v1`–`v5`) are iterative development/testing artifacts.

10-step workflow: load & split by device → timestamp collision resolution (M5Stick 1ms clock vs ~33 Hz sampling) → timestamp normalization → gap detection (>100 ms) → uniform resampling (50 Hz) → gyro bias correction (2 s static window) → Butterworth low-pass (6 Hz cutoff) → Madgwick AHRS (quaternion orientation + gravity subtraction) → cycle detection (||ω|| peaks, 100 deg/s prominence) → CSV + 5-panel validation plot output.

**Key physical parameters** (all physically motivated, not arbitrary):
- `FS_TARGET = 50` Hz, `LOWPASS_HZ = 6` Hz (retains 2nd harmonic for rope flow ~1–3 Hz)
- `MADGWICK_BETA = 0.1`, `STATIC_WINDOW_S = 2.0`, `GAP_WARN_MS = 100`

**Output columns**: `timestamp_ms, qw, qx, qy, qz, ax_w, ay_w, az_w, gx, gy, gz`

### Stage 2 — Equation discovery (`src/Equation_discovery/`)

Jupyter notebooks (`01_SINDY_PINN_DMD_V01–V03.ipynb`):
- **SINDy** — sparse identification of governing ODEs from processed acceleration
- **PINN** — physics-informed neural network for 3D position reconstruction
- **DMD + PCA** — dynamic mode decomposition for per-cycle feature extraction and pattern classification

## Data layout

- `data/raw/unified-data/` — 24 labeled sessions, each containing `imu_data.csv`, `metadata.json`, `sync_info.json`, optional `video.avi`
- `data/raw/app-data/` — earlier mobile app CSVs
- `data/processed/` — Stage 1 output: `*_device[0|1]_processed.csv`
- `results/` — validation plots and processing outputs

Each raw CSV has columns: `sync_time, timestamp_ms, device_id, ax, ay, az, gx, gy, gz`. Device 0 = left hand, device 1 = right hand.

## Running code

```bash
# Stage 1 — process a single recording
python src/Data_processing/06_Full_pipeline_onesample_v6.py

# Stage 2 — open notebooks in Jupyter
jupyter notebook src/Equation_discovery/01_SINDY_PINN_DMD_V03.ipynb
```

No `requirements.txt` exists yet. Key dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `ahrs`, `pysindy`, `scikit-learn`, `torch`.

## Conventions

- Scripts are numbered by pipeline order (`00_`–`06_`); notebooks are versioned (`V01`–`V03`). Always work on the **latest version**.
- All signal processing parameters must be **physically justified** (filter cutoffs tied to rope flow frequency bands, prominence thresholds tied to expected angular velocities).
- Every processing step should produce **validation plots** for visual inspection.
- Dual-sensor data is always split by `device_id` and processed independently.
