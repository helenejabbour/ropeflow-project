
# Comprehensive label index builder — all raw data folders
#
# Covers three data sources:
#   1. data/raw/unified-data/   — session-level labels from metadata.json
#   2. data/raw/new-labeled-sessions/ — per-segment labels from labels*.json
#   3. data/raw/app-data/       — no usable labels (music filenames only), skipped
#
# Output: data/processed/labels_index.csv
# Columns: session, start_s, end_s, canonical_label, label_type, source_folder

import os
import sys
import json
import re
import pandas as pd

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────
ROOT           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED_DATA   = os.path.join(ROOT, "data", "raw", "unified-data")
NEW_LABELED    = os.path.join(ROOT, "data", "raw", "new-labeled-sessions")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)

OUTPUT_CSV = os.path.join(DATA_PROCESSED, "labels_index.csv")

# ── Canonical class set ────────────────────────────────────────
# All valid class names — anything not in this set (after mapping) is flagged.
CANONICAL_CLASSES = {
    "underhand",
    "overhand",
    "dragon_roll",
    "sneak_underhand",
    "sneak_overhand",
    "race_and_chase",
    "cheetahs_tail",
}

# unified-data sessions whose metadata.pattern is 'experimental' or otherwise
# non-informative — skipped for supervised training.
SKIP_PATTERNS = {"experimental"}

# ── Shorthand → canonical mapping (for new-labeled-sessions) ──
# Exact match checked first (handles already-canonical labels).
# Prefix rules ordered so US*/OS* match before U*/O*.

CANONICAL_EXACT = {
    "underhand":       "underhand",
    "overhand":        "overhand",
    "dragon_roll":     "dragon_roll",
    "sneak_underhand": "sneak_underhand",
    "sneak_overhand":  "sneak_overhand",
    "race_and_chase":  "race_and_chase",
    "cheetahs_tail":   "cheetahs_tail",
    "idle":            None,
}

PREFIX_RULES = [
    (re.compile(r"^us",   re.IGNORECASE), "sneak_underhand"),  # USR, USL
    (re.compile(r"^os",   re.IGNORECASE), "sneak_overhand"),   # OSR, OSL
    (re.compile(r"^u",    re.IGNORECASE), "underhand"),        # UR, UL0, UR0, …
    (re.compile(r"^o",    re.IGNORECASE), "overhand"),         # OR, OL, OR2, or, …
    (re.compile(r"^fb",   re.IGNORECASE), "dragon_roll"),      # FB, FB2
    (re.compile(r"^bf",   re.IGNORECASE), "dragon_roll"),      # BF2
    (re.compile(r"^cw$",  re.IGNORECASE), None),               # rotation transition
    (re.compile(r"^ccw$", re.IGNORECASE), None),               # rotation transition
    (re.compile(r"^idle", re.IGNORECASE), None),               # idle clusters
    (re.compile(r"^vq",   re.IGNORECASE), None),               # unlabeled clusters
]

NULL_END_FALLBACK_S = 2.0


def map_label(raw: str):
    """Return canonical label string, or None to discard."""
    raw = raw.strip()
    if raw.lower() in CANONICAL_EXACT:
        return CANONICAL_EXACT[raw.lower()]
    for pattern, canonical in PREFIX_RULES:
        if pattern.match(raw):
            return canonical
    print(f"  [WARN] unknown label '{raw}' — discarded")
    return None


# ── Source 1: unified-data (session-level) ────────────────────

def process_unified_data():
    """
    Read metadata.json from each unified-data session.
    Emit one segment [0, duration_sec] per session for sessions with a
    canonical (non-experimental) pattern label.
    """
    rows = []
    summary = []

    session_dirs = sorted([
        d for d in os.listdir(UNIFIED_DATA)
        if os.path.isdir(os.path.join(UNIFIED_DATA, d))
    ])

    print(f"\n{'─'*100}")
    print(f"  unified-data  ({len(session_dirs)} sessions)")
    print(f"{'─'*100}")
    print(f"  {'Session':<50} {'Pattern':<20} {'Status'}")

    for name in session_dirs:
        meta_path = os.path.join(UNIFIED_DATA, name, "metadata.json")
        if not os.path.isfile(meta_path):
            print(f"  {name:<50} {'—':<20} NO metadata.json")
            summary.append({"session": name, "status": "no_meta", "n_kept": 0})
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        pattern  = meta.get("pattern", "").strip()
        duration = meta.get("duration_sec")

        if pattern in SKIP_PATTERNS or not pattern:
            print(f"  {name:<50} {pattern:<20} skipped (unlabeled)")
            summary.append({"session": name, "status": "skipped", "n_kept": 0})
            continue

        if pattern not in CANONICAL_CLASSES:
            print(f"  {name:<50} {pattern:<20} [WARN] unknown pattern — skipped")
            summary.append({"session": name, "status": "unknown_pattern", "n_kept": 0})
            continue

        if duration is None or duration <= 0:
            print(f"  {name:<50} {pattern:<20} [WARN] missing duration — skipped")
            summary.append({"session": name, "status": "no_duration", "n_kept": 0})
            continue

        rows.append({
            "session":         name,
            "start_s":         0.0,
            "end_s":           round(float(duration), 4),
            "canonical_label": pattern,
            "label_type":      "session",
            "source_folder":   "unified-data",
        })
        print(f"  {name:<50} {pattern:<20} OK  [{0.0:.1f}s – {duration:.1f}s]")
        summary.append({"session": name, "status": "labeled", "n_kept": 1})

    return rows, summary


# ── Source 2: new-labeled-sessions (per-segment) ──────────────

def pick_label_file(session_dir: str):
    corrected = os.path.join(session_dir, "labels_corrected.json")
    plain     = os.path.join(session_dir, "labels.json")
    if os.path.isfile(corrected):
        return corrected, "labels_corrected.json"
    if os.path.isfile(plain):
        return plain, "labels.json"
    return None, None


def parse_label_file(label_path: str, session_name: str):
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    n_discarded = 0

    for seg in data.get("segments", []):
        raw_label = seg.get("label", "")
        start_s   = seg.get("start")
        end_s     = seg.get("end")

        if start_s is None:
            continue
        if end_s is None:
            end_s = start_s + NULL_END_FALLBACK_S

        canonical = map_label(raw_label)
        if canonical is None:
            n_discarded += 1
            continue

        rows.append({
            "session":         session_name,
            "start_s":         round(float(start_s), 4),
            "end_s":           round(float(end_s),   4),
            "canonical_label": canonical,
            "label_type":      "segment",
            "source_folder":   "new-labeled-sessions",
        })

    return rows, n_discarded


def process_new_labeled():
    rows = []
    summary = []

    session_dirs = sorted([
        d for d in os.listdir(NEW_LABELED)
        if os.path.isdir(os.path.join(NEW_LABELED, d))
    ])

    print(f"\n{'─'*100}")
    print(f"  new-labeled-sessions  ({len(session_dirs)} sessions)")
    print(f"{'─'*100}")
    print(f"  {'Session':<50} {'Label file':<28} {'Kept':>5} {'Disc':>5}")

    for name in session_dirs:
        session_dir = os.path.join(NEW_LABELED, name)
        label_path, label_fname = pick_label_file(session_dir)

        if label_path is None:
            print(f"  {name:<50} {'— no labels —':<28}")
            summary.append({"session": name, "status": "no_labels", "n_kept": 0})
            continue

        seg_rows, n_disc = parse_label_file(label_path, name)
        rows.extend(seg_rows)
        classes = sorted(set(r["canonical_label"] for r in seg_rows))
        print(f"  {name:<50} {label_fname:<28} {len(seg_rows):>5} {n_disc:>5}")
        print(f"    classes: {', '.join(classes)}")
        summary.append({"session": name, "status": "labeled", "n_kept": len(seg_rows)})

    return rows, summary


# ── Entry point ───────────────────────────────────────────────

def main():
    all_rows = []

    print("\n" + "═"*100)
    print("  COMPREHENSIVE LABEL INDEX BUILDER")
    print("═"*100)

    # Source 1: unified-data
    rows1, sum1 = process_unified_data()
    all_rows.extend(rows1)

    # Source 2: new-labeled-sessions
    rows2, sum2 = process_new_labeled()
    all_rows.extend(rows2)

    # Source 3: app-data — skipped (music filenames, no movement labels)
    print(f"\n{'─'*100}")
    print("  app-data  — skipped (11 files named after music tracks, no movement labels)")
    print(f"{'─'*100}")

    # Build and save DataFrame
    cols = ["session", "start_s", "end_s", "canonical_label", "label_type", "source_folder"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False)

    # ── Final summary ──────────────────────────────────────────
    all_summary = sum1 + sum2
    n_labeled   = sum(1 for r in all_summary if r["status"] == "labeled")
    n_skipped   = sum(1 for r in all_summary if r["status"] != "labeled")
    total_segs  = len(df)

    print(f"\n{'═'*100}")
    print(f"  Total sessions labeled : {n_labeled}")
    print(f"  Total sessions skipped : {n_skipped}")
    print(f"  Total segments/entries : {total_segs}")
    print(f"\n  label_type breakdown:")
    for lt, cnt in df["label_type"].value_counts().items():
        print(f"    {lt:<12} {cnt} entries")
    print(f"\n  Per-class segment counts:")
    for cls, cnt in df["canonical_label"].value_counts().items():
        print(f"    {cls:<22} {cnt}")
    print(f"\n  Saved → {OUTPUT_CSV}")
    print()


if __name__ == "__main__":
    main()
