# New Labeled Sessions — For Processing

These 19 sessions need to be run through the preprocessing pipeline
(`src/Data_processing/06_Full_pipeline_onesample_v6.py` or `07_Data_processing_denoising_all_v1.py`).

## What's here

Each folder contains `imu_data.csv`, `metadata.json`, `sync_info.json`,
and most importantly **per-segment movement labels** in `labels.json` or
`labels_corrected.json` (prefer `labels_corrected.json` when both exist).

These are **heterogeneous recordings** — the performer switches between
movements during the session. The label files contain timestamped segments
like:

```json
{"start": 10.5, "end": 15.2, "label": "underhand"}
```

## Label mapping

Some sessions use clean labels (`underhand`, `overhand`, `dragon_roll`,
`sneak_underhand`, `sneak_overhand`, `idle`).

Others use shorthand notation. Here's the mapping:

| Shorthand prefix | Movement class |
|---|---|
| U* (UR, UL0, UR0, UR-CW) | underhand |
| O* (OR, OL, OR2, OR3, OL2, OR-OSL, OR/OSL?) | overhand |
| US* (USR, USL) | sneak_underhand |
| OS* (OSR, OSL) | sneak_overhand |
| FB*, BF* (FB, FB2, BF2) | dragon_roll |
| CW, CCW | discard (rotation transitions) |
| Idle*, VQ* | discard (idle / unlabeled clusters) |

## Sessions with clean labels (8)

- `20260315_155448_experimental_jo` — underhand, overhand, dragon_roll
- `20260315_182722_experimental_jo` — underhand, overhand, dragon_roll
- `20260316_170025_experimental_oli` — underhand, overhand, dragon_roll
- `20260316_170310_experimental_oli` — underhand, overhand
- `20260316_171050_experimental_oli` — underhand, overhand, dragon_roll
- `20260316_171550_experimental_jo` — underhand, overhand, dragon_roll, sneak_overhand, sneak_underhand
- `20260316_172228_experimental_jo` — underhand, overhand, dragon_roll
- `20260405_173949_experimental_progression` — underhand, overhand, dragon_roll, sneak_overhand, sneak_underhand

## Sessions with shorthand labels (7)

- `20260315_150630_experimental_jo`
- `20260405_184437_experimental_bass-walk-down`
- `20260405_184729_experimental_pachelbel`
- `20260405_184905_experimental_pachelbel`
- `20260405_185331_experimental_birth-death`
- `20260405_185509_experimental_bass-walk-down`
- `20260406_212408_experimental_jo div`

## Sessions without labels (4) — cannot use for training

- `20260316_170613_experimental_jo`
- `20260405_184927_experimental_pachelbel`
- `20260405_185105_experimental_pachelbel`
- `20260406_211811_experimental_jo div`

## What to do

1. Process each session through the preprocessing pipeline (output: device0 + device1 CSVs)
2. For classification, we'll slice the processed data by the label timestamps to extract per-movement segments
3. Audio files (`.wav`) were excluded from this upload to save space — they're on Mounir's local machine if needed
4. One video (`20260406_212408` from data3) was excluded (279MB) — also on local machine
