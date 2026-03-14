# Rope Flow IMU + Video Dataset

24 synchronized recordings of rope flow patterns captured with dual M5StickC Plus IMU sensors and video.

## Data Structure

Each recording folder contains:

| File | Description |
|------|-------------|
| `imu_data.csv` | Accelerometer + gyroscope at ~100Hz from two wrist-mounted sensors |
| `video.avi` (or `.mp4`) | Synchronized video of the session |
| `sync_info.json` | Video-IMU time alignment (frame timestamps in seconds) |
| `metadata.json` | Recording metadata (pattern, subject, duration) |

## IMU CSV Columns

| Column | Description |
|--------|-------------|
| `sync_time` | Seconds from recording start (use this for time alignment) |
| `timestamp_ms` | Raw device timestamp in milliseconds |
| `device_id` | 0 = left hand, 1 = right hand |
| `ax, ay, az` | Accelerometer (g) |
| `gx, gy, gz` | Gyroscope (deg/s) |

## Video Sync

`sync_info.json` contains a `video_frame_times` array where each entry is the time in seconds (from recording start) of the corresponding video frame. This shares the same time reference as `sync_time` in the CSV.

## Patterns

- **underhand** - Basic underhand rope swing
- **overhand** - Overhand rope swing
- **dragon_roll** - Figure-8 pattern
- **sneak_underhand / sneak_overhand** - Lower body engagement variants
- **race_and_chase** - Alternating fast patterns
- **cheetahs_tail** - Whipping tail motion
- **experimental** - Free-form / mixed patterns

## Subjects

- **jo** - Joe (experienced)
- **oli** - Oli (learning)
- **default / free** - Unnamed or free-form sessions

## Quick Start (Python)

```python
import pandas as pd
import numpy as np
import json

# Load IMU data
df = pd.read_csv('20260303_174607_underhand_jo/imu_data.csv')
left = df[df['device_id'] == 0]
right = df[df['device_id'] == 1]

# Acceleration magnitude
left_mag = np.sqrt(left['ax']**2 + left['ay']**2 + left['az']**2)

# Load video sync
with open('20260303_174607_underhand_jo/sync_info.json') as f:
    sync = json.load(f)
frame_times = np.array(sync['video_frame_times'])
```

See `manifest.json` for a machine-readable index of all recordings.
