import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bisect import bisect_right
from scipy.signal import savgol_filter, find_peaks

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
 )
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# -- Paths ----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED = os.path.join(ROOT, 'data', 'processed')
LABELED_RAW_BASE = os.path.join(ROOT, 'data', 'raw', 'new-labeled-sessions')

# -- Config ---------------------------------------------------------------
DIRECT_CFG = {
    'FS':                 50.0,
    'WINDOW':             32,
    'PEAK_PROM_DEGS':     50.0,
    'PEAK_MIN_DEGS':      50.0,
    'PEAK_SAVGOL_WINDOW': 15,
    'PEAK_SAVGOL_POLY':   3,
    'PEAK_MIN_PERIOD_S':  0.2,
    'MERGE_GAP_S':        0.20,
    'PCA_VAR':            0.99,
    # Put your exact 12 supervised classes here; all other labels map to UNKNOWN_CLASS.
    'SUPERVISED_CLASSES': [
        'dragon_roll',
        'underhand_right',
        'underhand_left',
        'overhand_left',
        'overhand_right',
        'sneak_underhand_left',
        'sneak_underhand_right',
        'sneak_overhand_left',
        'sneak_overhand_right',
        'clockwise',
        'counter_clockwise',
        'idle'
    ],
    #'UNKNOWN_CLASS':      'unknown',
}
if len(DIRECT_CFG['SUPERVISED_CLASSES']) != 12:
    raise ValueError('DIRECT_CFG[SUPERVISED_CLASSES] must contain exactly 12 known classes.')

# -- Results directory (new folder per run) --------------------------------
import datetime
RUN_NAME    = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
RESULTS_DIR = os.path.join('..', '..', 'results', 'Full_pipeline', RUN_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'Results will be saved to: {RESULTS_DIR}')


# -- Session discovery ---------------------------------------------------
def discover_direct_sessions(processed_dir):
    entries = []
    for d0 in sorted(glob.glob(os.path.join(processed_dir, '*_device0_processed.csv'))):
        d1 = d0.replace('_device0_', '_device1_')
        if not os.path.isfile(d1):
            continue
        stem = os.path.basename(d0).replace('_device0_processed.csv', '')
        entries.append((d0, d1, stem))
    return entries

# -- Label canonicalization ----------------------------------------------
LABEL_ALIAS_MAP = {
    'bf': 'dragon_roll', 'bf2': 'dragon_roll', 'fb': 'dragon_roll', 'fb2': 'dragon_roll',
    'dragon roll': 'dragon_roll', 'dragon_roll': 'dragon_roll',
    'ur': 'underhand_right', 'ur0': 'underhand_right',
    'underhand right': 'underhand_right', 'underhand': 'underhand_right',
    'ul': 'underhand_left', 'ul0': 'underhand_left',
    'underhand left': 'underhand_left',
    'ol': 'overhand_left', 'ol0': 'overhand_left', 'ol2': 'overhand_left',
    'overhand left': 'overhand_left', 'overhand': 'overhand_left',
    'or': 'overhand_right', 'or2': 'overhand_right', 'or3': 'overhand_right',
    'overhand right': 'overhand_right',
    'usl': 'sneak_underhand_left', 'sneak underhand left': 'sneak_underhand_left', 'sneak underhand': 'sneak_underhand_left',
    'usr': 'sneak_underhand_right', 'sneak underhand right': 'sneak_underhand_right',
    'osl': 'sneak_overhand_left',  'sneak overhand left': 'sneak_overhand_left', 'sneak overhand': 'sneak_overhand_left',
    'osr': 'sneak_overhand_right', 'sneak overhand right': 'sneak_overhand_right',
    'cw': 'clockwise', 'clockwise': 'clockwise',
    'ccw': 'counter_clockwise', 'counter clockwise': 'counter_clockwise',
    'idle': 'idle', 'idle3': 'idle', 'no movement': 'idle',
    'vq5': 'vq', 'vq15': 'vq', 'vq16': 'vq', 'vq': 'vq',
    'excluded': 'excluded', 'ur cw': 'excluded', 'or osl': 'excluded', 'idle or ol?': 'excluded', 'vq': 'excluded',
}

def _normalize_label_key(label):
    s = str(label).strip().lower().replace('_', ' ').replace('-', ' ')
    return ' '.join(s.split())

def canonicalize_label(label):
    key = _normalize_label_key(label)
    if key in LABEL_ALIAS_MAP:
        return LABEL_ALIAS_MAP[key]
    for sep in ('/', '|'):
        if sep in key:
            parts = [p.strip() for p in key.split(sep) if p.strip()]
            if parts:
                first = _normalize_label_key(parts[0])
                return LABEL_ALIAS_MAP.get(first, first)
    return key

def _safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

# -- Cycle-extraction helpers ---------------------------------------------
def _smooth_mag_deg(omega_rad, cfg):
    mag_deg = np.linalg.norm(omega_rad, axis=1) * (180.0 / np.pi)
    n = len(mag_deg)
    if n < 7:
        return mag_deg
    win = int(cfg['PEAK_SAVGOL_WINDOW'])
    if win % 2 == 0:
        win += 1
    max_odd = n if n % 2 == 1 else n - 1
    win = max(5, min(win, max_odd))
    poly = max(1, min(int(cfg['PEAK_SAVGOL_POLY']), win - 2))
    return savgol_filter(mag_deg, window_length=win, polyorder=poly, mode='interp')

def detect_cycle_peaks(omega_rad, fs, cfg):
    mag_smooth = _smooth_mag_deg(omega_rad, cfg)
    peaks, _ = find_peaks(
        mag_smooth,
        distance=max(1, int(cfg['PEAK_MIN_PERIOD_S'] * fs)),
        prominence=cfg['PEAK_PROM_DEGS'],
    )
    peaks = np.array([int(p) for p in peaks if mag_smooth[p] >= cfg['PEAK_MIN_DEGS']], dtype=int)
    return peaks, mag_smooth

def merge_device_peaks_pairs(peaks0, peaks1, fs_or_gap=100.0, gap_s=None):
    if gap_s is None:
        fs = float(DIRECT_CFG.get('FS', 100.0))
        gap_s = float(fs_or_gap)
    else:
        fs = float(fs_or_gap)
        gap_s = float(gap_s)

    tagged = [(p / fs, p, 'D0') for p in peaks0] + [(p / fs, p, 'D1') for p in peaks1]
    if not tagged:
        return []
    tagged.sort(key=lambda x: x[0])
    all_idx_t = [x[0] for x in tagged]

    accepted = [0]
    for i in range(1, len(all_idx_t)):
        if all_idx_t[i] - all_idx_t[accepted[-1]] > gap_s:
            accepted.append(i)

    group_peaks = [{} for _ in accepted]
    a_idx = 0
    for i, (_, peak_idx, src) in enumerate(tagged):
        if a_idx + 1 < len(accepted) and i >= accepted[a_idx + 1]:
            a_idx += 1
        group_peaks[a_idx].setdefault(src, peak_idx)

    # Bug 3 fix: return ALL unique peaks, not just paired ones; p0/p1 may be None
    return [(g.get('D0'), g.get('D1')) for g in group_peaks]

def extract_fixed_window(ch6, center_idx, window=64):
    half = int(window // 2)
    start = int(center_idx) - half
    end = start + int(window)
    out = np.zeros((6, int(window)), dtype=np.float32)
    src_lo = max(0, start)
    src_hi = min(ch6.shape[0], end)
    if src_hi <= src_lo:
        return out
    dst_lo = src_lo - start
    out[:, dst_lo:dst_lo + (src_hi - src_lo)] = ch6[src_lo:src_hi].T
    return out

def load_session(path_d0, path_d1):
    d0 = pd.read_csv(path_d0)
    d1 = pd.read_csv(path_d1)
    return d0, d1

def extract_signals(df):
    t = df['timestamp_ms'].values / 1000.0
    A = df[['ax_w', 'ay_w', 'az_w']].values
    omega = df[['gx', 'gy', 'gz']].values * (np.pi / 180.0)
    return t, A, omega

# -- Annotation loading & time labeling -----------------------------------
def _load_time_labels_for_session(session_id):
    session_dir = os.path.join(LABELED_RAW_BASE, session_id)
    if not os.path.isdir(session_dir):
        return None
    for name in ('labels_corrected.json', 'labels.json', 'labels_vad.json'):
        path = os.path.join(session_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        segments = data.get('segments', []) or []
        label_events = data.get('label_events', []) or []

        clean_segs = []
        for s in segments:
            if not isinstance(s, dict):
                continue
            if not {'start', 'end', 'label'}.issubset(s):
                continue
            t0 = _safe_float(s.get('start'))
            t1 = _safe_float(s.get('end'))
            if t0 is None or t1 is None:
                continue
            if not np.isfinite(t0) or not np.isfinite(t1):
                continue
            if t1 <= t0:
                continue
            clean_segs.append((t0, t1, canonicalize_label(s.get('label'))))

        clean_evs = []
        for ev in label_events:
            if not isinstance(ev, dict):
                continue
            if 'time' not in ev or 'label' not in ev:
                continue
            te = _safe_float(ev.get('time'))
            if te is None or (not np.isfinite(te)):
                continue
            clean_evs.append((te, canonicalize_label(ev.get('label'))))
        clean_evs.sort(key=lambda x: x[0])

        if not clean_segs and not clean_evs:
            continue

        return {
            'json_path': path,
            'source_name': name,
            'segments': clean_segs,
            'events': clean_evs,
        }
    return None

def _apply_time_offset_to_ann(ann, csv_t_min, csv_t_max, sid=''):
    # Bug 2 fix: correct JSON timestamps that are in a different reference frame than
    # the processed CSV (which starts from 0 after normalization in preprocessing).
    if not ann['segments'] and not ann['events']:
        return ann

    all_times = [t for t0, t1, _ in ann['segments'] for t in (t0, t1)]
    all_times += [te for te, _ in ann['events']]
    if not all_times:
        return ann

    json_t_min = min(all_times)
    json_t_max = max(all_times)
    print(f'  [{sid}] CSV range: {csv_t_min:.3f}–{csv_t_max:.3f} s | '
          f'JSON range: {json_t_min:.3f}–{json_t_max:.3f} s')

    # Only correct when JSON times are entirely outside CSV range
    if json_t_min > csv_t_max or json_t_max < csv_t_min:
        offset = json_t_min - csv_t_min
        print(f'  [{sid}] -> offset correction: -{offset:.3f} s')
        corrected = dict(ann)
        corrected['segments'] = [(t0 - offset, t1 - offset, lab)
                                  for t0, t1, lab in ann['segments']]
        corrected['events']   = [(te - offset, lab) for te, lab in ann['events']]
        return corrected
    return ann

def _label_at_time(t_s, ann):
    # Bug 4 fix: no nearest-segment fallback — gaps return None (training poison)
    for t0, t1, lab in ann['segments']:
        if t0 <= t_s < t1:
            return lab
    if ann['segments']:
        return None
    if ann['events']:
        times = [e[0] for e in ann['events']]
        return ann['events'][max(0, bisect_right(times, t_s) - 1)][1]
    return None

def map_to_supervised_class(raw_label, cfg):
    if raw_label is None:
        return None
    lab = canonicalize_label(raw_label)
    if lab == 'excluded':
        return None
    known = set(cfg['SUPERVISED_CLASSES'])
    if lab in known:
        return lab
    return cfg.get('UNKNOWN_CLASS', 'unknown')

_DIAG_SID = '20260406_212408_experimental_jo div'

def extract_labeled_cycles_from_entry(entry, cfg, ann):
    d0_path, d1_path, sid = entry
    d0, d1 = load_session(d0_path, d1_path)
    t0, A0, om0 = extract_signals(d0)
    t1, A1, om1 = extract_signals(d1)

    # Bug 2: correct JSON timestamp reference frame to match processed CSV (starts at 0)
    ann = _apply_time_offset_to_ann(ann, float(t0.min()), float(t0.max()), sid)

    ch0 = np.column_stack([A0, om0 * (180.0 / np.pi)])
    ch1 = np.column_stack([A1, om1 * (180.0 / np.pi)])

    mats, labels, mids = [], [], []
    n_excl = 0

    if ann['segments']:
        # Labeled session: each JSON segment IS one cycle — use its midpoint directly.
        # No peak detection; no peak-to-label matching; exactly 1 sample per segment.
        for seg_start, seg_end, seg_label in ann['segments']:
            y_lab = map_to_supervised_class(seg_label, cfg)
            if y_lab is None:
                n_excl += 1
                continue
            t_mid = 0.5 * (seg_start + seg_end)
            center_d0 = int(np.argmin(np.abs(t0 - t_mid)))
            center_d1 = int(np.argmin(np.abs(t1 - t_mid)))
            w0 = extract_fixed_window(ch0, center_d0, cfg['WINDOW'])
            w1 = extract_fixed_window(ch1, center_d1, cfg['WINDOW'])
            mats.append(np.vstack([w0, w1]).astype(np.float32))
            labels.append(y_lab)
            mids.append(float(t_mid))

        if sid == _DIAG_SID:
            print(f'\n[DIAG] {sid}')
            print(f'  Segments in JSON: {len(ann["segments"])} | '
                  f'Cycles extracted: {len(mats)} | excluded/unknown-None: {n_excl}')
    else:
        # Unlabeled session (label_events fallback): use peak detection for cycle centers.
        peaks0, _ = detect_cycle_peaks(om0, cfg['FS'], cfg)
        peaks1, _ = detect_cycle_peaks(om1, cfg['FS'], cfg)
        pairs = merge_device_peaks_pairs(peaks0, peaks1, cfg['FS'], cfg.get('MERGE_GAP_S', 0.15))
        n_gap = 0
        for p0, p1 in pairs:
            if p0 is not None and p1 is not None:
                t_ref = 0.5 * (t0[int(p0)] + t1[int(p1)])
            elif p0 is not None:
                t_ref = float(t0[int(p0)])
            else:
                t_ref = float(t1[int(p1)])
            center_d0 = int(np.argmin(np.abs(t0 - t_ref)))
            center_d1 = int(np.argmin(np.abs(t1 - t_ref)))
            y_raw = _label_at_time(float(t_ref), ann)
            if y_raw is None:
                n_gap += 1
                continue
            y_lab = map_to_supervised_class(y_raw, cfg)
            if y_lab is None:
                n_excl += 1
                continue
            w0 = extract_fixed_window(ch0, center_d0, cfg['WINDOW'])
            w1 = extract_fixed_window(ch1, center_d1, cfg['WINDOW'])
            mats.append(np.vstack([w0, w1]).astype(np.float32))
            labels.append(y_lab)
            mids.append(float(t_ref))

    return mats, labels, mids

def build_labeled_cycle_dataset(entries, cfg):
    X_list, y_list, g_list, sid_list, tm_list = [], [], [], [], []
    ann_cache = {}
    n_sessions_with_labels = 0

    # DIAGNOSTIC: print timestamp alignment for first 3 sessions with labels
    n_diag = 0
    for entry in entries:
        sid = entry[2]
        ann = _load_time_labels_for_session(sid)
        if ann is None or not ann['segments']:
            continue
        d0 = pd.read_csv(entry[0])
        csv_start = d0['timestamp_ms'].iloc[0] / 1000.0
        csv_end = d0['timestamp_ms'].iloc[-1] / 1000.0
        json_starts = [s[0] for s in ann['segments']]
        json_ends = [s[1] for s in ann['segments']]
        print(f'\n  TIMESTAMP CHECK: {sid}')
        print(f'    CSV range:  {csv_start:.3f} — {csv_end:.3f} s')
        print(f'    JSON range: {min(json_starts):.3f} — {max(json_ends):.3f} s')
        print(f'    Offset:     {min(json_starts) - csv_start:.3f} s')
        print(f'    First 3 segments: {ann["segments"][:3]}')
        n_diag += 1
        if n_diag >= 3:
            break
        
    for entry in entries:
        sid = entry[2]
        if sid not in ann_cache:
            ann_cache[sid] = _load_time_labels_for_session(sid)
        ann = ann_cache[sid]
        if ann is None:
            continue
        mats, labs, mids = extract_labeled_cycles_from_entry(entry, cfg, ann)
        if not mats:
            continue
        n_sessions_with_labels += 1
        for m, y, tm in zip(mats, labs, mids):
            X_list.append(m)
            y_list.append(y)
            g_list.append(sid)
            sid_list.append(sid)
            tm_list.append(tm)

    if not X_list:
        raise RuntimeError('No labeled cycles extracted.')

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=object)
    groups = np.array(g_list, dtype=object)
    session_ids = np.array(sid_list, dtype=object)
    t_mid_s = np.array(tm_list, dtype=np.float32)

    print(f'Sessions found: {len(entries)}')
    print(f'Sessions with labels and cycles: {n_sessions_with_labels}')
    print(f'Labeled dataset: X={X.shape} | classes={len(np.unique(y))}')
    return X, y, groups, session_ids, t_mid_s

def build_feature_matrix(X_cycles):
    eps = 1e-8

    def biomech_features_from_cycle(cycle_12xt):
        w0 = cycle_12xt[3:6, :].T
        w1 = cycle_12xt[9:12, :].T
        n = min(len(w0), len(w1))
        if n < 8:
            return np.zeros(24, dtype=np.float32)

        w0 = w0[:n]
        w1 = w1[:n]
        m0 = np.linalg.norm(w0, axis=1)
        m1 = np.linalg.norm(w1, axis=1)

        def dom_stats(w):
            e = np.mean(w ** 2, axis=0)
            k = int(np.argmax(e))
            share = float(e[k] / (np.sum(e) + eps))
            signed_mean = float(np.mean(w[:, k]) / (np.sqrt(e[k]) + eps))
            signflip = float(np.mean(np.sign(w[:-1, k]) != np.sign(w[1:, k])))
            return float(k), share, signed_mean, signflip

        k0, sh0, sgn0, sf0 = dom_stats(w0)
        k1, sh1, sgn1, sf1 = dom_stats(w1)

        mid = n // 2
        hw = max(3, int(0.15 * n))
        seg0 = w0[max(0, mid - hw):min(n, mid + hw), :2]
        seg1 = w1[max(0, mid - hw):min(n, mid + hw), :2]
        mid_evt0 = float(np.max(np.linalg.norm(seg0, axis=1)) / (np.sqrt(np.mean(w0[:, :2] ** 2)) + eps))
        mid_evt1 = float(np.max(np.linalg.norm(seg1, axis=1)) / (np.sqrt(np.mean(w1[:, :2] ** 2)) + eps))
        mid_asym = float(np.abs(mid_evt0 - mid_evt1))

        a = m0 - np.mean(m0)
        b = m1 - np.mean(m1)
        cc = np.correlate(a, b, mode='full')
        lag = int(np.argmax(cc) - (n - 1))
        phase_deg = float((lag / max(1, n)) * 360.0)
        inphase = float(np.cos(np.deg2rad(phase_deg)))
        antiphase = float(np.cos(np.deg2rad(phase_deg - 180.0)))

        win = max(8, int(0.25 * n))
        step = max(2, win // 2)
        lags = []
        for st in range(0, n - win + 1, step):
            aa = m0[st:st + win] - np.mean(m0[st:st + win])
            bb = m1[st:st + win] - np.mean(m1[st:st + win])
            c = np.correlate(aa, bb, mode='full')
            lags.append(np.argmax(c) - (win - 1))
        lag_var = float(np.var(lags)) if len(lags) > 1 else 0.0

        imp0 = float(np.max(m0) / (np.mean(m0) + eps))
        imp1 = float(np.max(m1) / (np.mean(m1) + eps))
        imp_asym = float(max(imp0 / (imp1 + eps), imp1 / (imp0 + eps)))

        q0 = np.percentile(m0, 20)
        q1 = np.percentile(m1, 20)
        h0 = np.percentile(m0, 80)
        h1 = np.percentile(m1, 80)
        quiet0 = float(np.mean(m0 <= q0))
        quiet1 = float(np.mean(m1 <= q1))
        quiet_active_overlap = float(
            max(np.mean((m0 <= q0) & (m1 >= h1)), np.mean((m1 <= q1) & (m0 >= h0)))
        )

        e0 = m0 ** 2
        e1 = m1 ** 2
        asym_t = np.abs(e0 - e1) / (e0 + e1 + eps)
        asym_mean = float(np.mean(asym_t))
        asym_std = float(np.std(asym_t))
        asym_peak = float(np.max(asym_t))

        return np.array([
            k0, sh0, sgn0, sf0,
            k1, sh1, sgn1, sf1,
            mid_evt0, mid_evt1, mid_asym,
            phase_deg, inphase, antiphase, lag_var,
            imp0, imp1, imp_asym,
            quiet0, quiet1, quiet_active_overlap,
            asym_mean, asym_std, asym_peak,
        ], dtype=np.float32)

    F_flat = X_cycles.reshape(len(X_cycles), -1).astype(np.float32)
    F_biomech = np.vstack([biomech_features_from_cycle(c) for c in X_cycles]).astype(np.float32)
    F = np.hstack([F_flat, F_biomech]).astype(np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'Feature matrix: {F.shape} (flattened + 24 biomech)')
    return F


def build_class_list(cfg, y_labels=None):
    classes = list(cfg['SUPERVISED_CLASSES'])
    unk = str(cfg.get('UNKNOWN_CLASS', 'unknown'))
    has_unknown = y_labels is not None and unk in set(y_labels)
    if has_unknown and unk not in classes:
        classes.append(unk)
    return classes

def encode_labels(y_str, class_names):
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    unknown_name = 'unknown'
    y_idx_list = []
    unseen = []
    for y in y_str:
        if y in cls_to_idx:
            y_idx_list.append(cls_to_idx[y])
        elif unknown_name in cls_to_idx:
            y_idx_list.append(cls_to_idx[unknown_name])
        else:
            unseen.append(y)
    if unseen:
        raise ValueError(f'Found labels not in class_names and no unknown class present: {sorted(set(unseen))}')
    y_idx = np.array(y_idx_list, dtype=np.int32)
    return y_idx, cls_to_idx

def _save_supervised_eval(y_true, y_pred, class_names, tag, results_dir):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    per_class_rows = []
    for cname in class_names:
        stats = report.get(cname, {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0})
        per_class_rows.append({
            'class': cname,
            'precision': float(stats['precision']),
            'recall': float(stats['recall']),
            'f1': float(stats['f1-score']),
            'support': int(stats['support']),
        })

    df_per_class = pd.DataFrame(per_class_rows)
    out_pc = os.path.join(results_dir, f'{tag}_per_class_metrics.csv')
    df_per_class.to_csv(out_pc, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False
    )
    ax.set_title(f'{tag} | acc={acc:.3f}, macro-F1={macro_f1:.3f}')
    fig.tight_layout()
    out_cm = os.path.join(results_dir, f'fig_confusion_{tag}.png')
    fig.savefig(out_cm, bbox_inches='tight')
    plt.close(fig)

    print(f'[{tag}] accuracy={acc:.4f} | macro-F1={macro_f1:.4f}')
    print(df_per_class.to_string(index=False))

    return {
        'approach': tag,
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class_csv': out_pc,
        'confusion_png': out_cm,
    }


def make_shared_stratified_split(X_feat, X_cycles, y, test_size=0.2, random_state=42):
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=random_state)
    return {
        'Xf_train': X_feat[tr_idx],
        'Xf_test': X_feat[te_idx],
        'Xc_train': X_cycles[tr_idx],
        'Xc_test': X_cycles[te_idx],
        'y_train': y[tr_idx],
        'y_test': y[te_idx],
    }


def run_stratified_rf(X_train_s, X_test_s, y_train, y_test, class_names, results_dir):
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=24,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    print(f'[RF] Accuracy: {acc:.4f}')

    res = _save_supervised_eval(y_test, y_pred, class_names, 'stratified_rf', results_dir)
    return res


def run_stratified_pca_gbm(X_train_s, X_test_s, y_train, y_test, class_names, results_dir):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=DIRECT_CFG.get('PCA_VAR', 0.99), svd_solver='full')
    X_tr_p = pca.fit_transform(X_train_s)
    X_te_p = pca.transform(X_test_s)

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.08,
        random_state=42,
    )
    clf.fit(X_tr_p, y_train)
    y_pred = clf.predict(X_te_p)

    acc = accuracy_score(y_test, y_pred)
    print(f'[PCA+GBM] Accuracy: {acc:.4f} | PCA dims: {X_tr_p.shape[1]}')

    res = _save_supervised_eval(y_test, y_pred, class_names, 'stratified_pca_gbm', results_dir)
    return res


def run_stratified_cnn(X_train, X_test, y_train, y_test, class_names, results_dir):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    n_classes = len(class_names)

    class CycleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(12, 64, 5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

        def forward(self, x):
            return self.head(self.conv(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CycleNet().to(device)

    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = torch.tensor(1.0 / (counts + 1.0), dtype=torch.float32).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    train_ds = TensorDataset(
        torch.tensor(X_train_n, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(80):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if (epoch + 1) % 20 == 0:
            print(f'  [CNN] epoch {epoch + 1}/80')

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test_n, dtype=torch.float32).to(device)
        y_pred = model(X_te_t).argmax(dim=1).cpu().numpy().astype(np.int32)

    acc = accuracy_score(y_test, y_pred)
    print(f'[CNN] Accuracy: {acc:.4f}')

    res = _save_supervised_eval(y_test, y_pred, class_names, 'stratified_cnn', results_dir)
    return res


def run_stratified_bilstm(X_train, X_test, y_train, y_test, class_names, results_dir):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    X_train_n = np.transpose(X_train_n, (0, 2, 1))
    X_test_n = np.transpose(X_test_n, (0, 2, 1))

    n_classes = len(class_names)

    class BiLSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=12,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3,
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMNet().to(device)

    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = torch.tensor(1.0 / (counts + 1.0), dtype=torch.float32).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    train_ds = TensorDataset(
        torch.tensor(X_train_n, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(80):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if (epoch + 1) % 20 == 0:
            print(f'  [BiLSTM] epoch {epoch + 1}/80')

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test_n, dtype=torch.float32).to(device)
        y_pred = model(X_te_t).argmax(dim=1).cpu().numpy().astype(np.int32)

    acc = accuracy_score(y_test, y_pred)
    print(f'[BiLSTM] Accuracy: {acc:.4f}')

    res = _save_supervised_eval(y_test, y_pred, class_names, 'stratified_bilstm', results_dir)
    return res


def _resample_sequence(seq, target_len=16):
    src_len = seq.shape[0]
    if src_len == target_len:
        return seq.astype(np.float32, copy=False)
    x_src = np.linspace(0.0, 1.0, src_len)
    x_dst = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for ch in range(seq.shape[1]):
        out[:, ch] = np.interp(x_dst, x_src, seq[:, ch])
    return out


def _dtw_distance_banded(a, b, radius=2):
    n, m = a.shape[0], b.shape[0]
    radius = max(int(radius), abs(n - m))
    inf = np.inf
    dp = np.full((n + 1, m + 1), inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - radius)
        j1 = min(m, i + radius)
        ai = a[i - 1]
        for j in range(j0, j1 + 1):
            cost = float(np.linalg.norm(ai - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def run_stratified_dtw_knn(X_train, X_test, y_train, y_test, class_names, results_dir, k=3):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    X_train_seq = np.stack([
        _resample_sequence(x.transpose(1, 0), target_len=16) for x in X_train_n
    ], axis=0)
    X_test_seq = np.stack([
        _resample_sequence(x.transpose(1, 0), target_len=16) for x in X_test_n
    ], axis=0)

    y_pred = np.empty(len(X_test_seq), dtype=np.int32)
    for i, xq in enumerate(X_test_seq):
        dists = np.array([_dtw_distance_banded(xq, xt, radius=2) for xt in X_train_seq], dtype=np.float32)
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx]
        votes = np.bincount(nn_labels, minlength=len(class_names))
        y_pred[i] = int(np.argmax(votes))
        if (i + 1) % 20 == 0 or i + 1 == len(X_test_seq):
            print(f'  [DTW+kNN] {i + 1}/{len(X_test_seq)}')

    acc = accuracy_score(y_test, y_pred)
    print(f'[DTW+kNN] Accuracy: {acc:.4f} | k={k}')

    res = _save_supervised_eval(y_test, y_pred, class_names, 'stratified_dtw_knn', results_dir)
    return res

def save_dataset_diagnostics(y_str, groups, results_dir):
    df = pd.DataFrame({'label': y_str, 'session_id': groups})

    label_counts = df['label'].value_counts().rename_axis('label').reset_index(name='count')
    label_counts.to_csv(os.path.join(results_dir, 'label_distribution.csv'), index=False)

    session_counts = df['session_id'].value_counts().rename_axis('session_id').reset_index(name='count')
    session_counts.to_csv(os.path.join(results_dir, 'session_cycle_counts.csv'), index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(label_counts['label'], label_counts['count'], color='#2f6f8f')
    ax.set_title('Cycle counts per class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'fig_label_distribution.png'), bbox_inches='tight')
    plt.close(fig)

    print('Saved dataset diagnostics:')
    print(f"  {os.path.join(results_dir, 'label_distribution.csv')}")
    print(f"  {os.path.join(results_dir, 'session_cycle_counts.csv')}")
    print(f"  {os.path.join(results_dir, 'fig_label_distribution.png')}")

    # ════════════════════════════════════════════════════════════════════════════
# MAIN (SUPERVISED DATASET BUILD)
# ════════════════════════════════════════════════════════════════════════════
SESSIONS = discover_direct_sessions(DATA_PROCESSED)
print(f'Looking in: {os.path.abspath(DATA_PROCESSED)}')
print(f'Directory exists: {os.path.isdir(DATA_PROCESSED)}')
if os.path.isdir(DATA_PROCESSED):
    csvs = glob.glob(os.path.join(DATA_PROCESSED, '*_device0_processed.csv'))
    print(f'Device0 CSVs found: {len(csvs)}')
    if csvs:
        print(f'  Example: {csvs[0]}')
        
print(f'Sessions found: {len(SESSIONS)}')

X_cycles, y_labels, groups, session_ids, t_mid_s = build_labeled_cycle_dataset(SESSIONS, DIRECT_CFG)

X_feat = build_feature_matrix(X_cycles)

CLASS_NAMES = build_class_list(DIRECT_CFG, y_labels)
if 'unknown' not in set(y_labels):
    CLASS_NAMES = [c for c in CLASS_NAMES if c != 'unknown']
y_idx, CLASS_TO_IDX = encode_labels(y_labels, CLASS_NAMES)

print(f'Using {len(CLASS_NAMES)} classes:')
print(CLASS_NAMES)

save_dataset_diagnostics(y_labels, groups, RESULTS_DIR)

supervised_dataset = pd.DataFrame({
    'session_id': session_ids,
    't_mid_s': t_mid_s,
    'label': y_labels,
    'label_idx': y_idx,
})
out_ds = os.path.join(RESULTS_DIR, 'supervised_cycle_index.csv')
supervised_dataset.to_csv(out_ds, index=False)
print(f'Supervised cycle index saved: {out_ds}')

split = make_shared_stratified_split(X_feat, X_cycles, y_idx, test_size=0.2, random_state=42)
Xf_train, Xf_test = split['Xf_train'], split['Xf_test']
Xc_train, Xc_test = split['Xc_train'], split['Xc_test']
y_train, y_test = split['y_train'], split['y_test']

sc = StandardScaler()
Xf_train_s = sc.fit_transform(Xf_train)
Xf_test_s = sc.transform(Xf_test)

print(f'\nTrain: {len(y_train)} | Test: {len(y_test)} (shared split, random_state=42)')

print('\n-- Stratified 80/20: RF --')
res_rf = run_stratified_rf(Xf_train_s, Xf_test_s, y_train, y_test, CLASS_NAMES, RESULTS_DIR)

print('\n-- Stratified 80/20: PCA+GBM --')
res_pca_gbm = run_stratified_pca_gbm(Xf_train_s, Xf_test_s, y_train, y_test, CLASS_NAMES, RESULTS_DIR)

print('\n-- Stratified 80/20: 1D-CNN --')
res_cnn = run_stratified_cnn(Xc_train, Xc_test, y_train, y_test, CLASS_NAMES, RESULTS_DIR)

print('\n-- Stratified 80/20: BiLSTM --')
res_bilstm = run_stratified_bilstm(Xc_train, Xc_test, y_train, y_test, CLASS_NAMES, RESULTS_DIR)

print('\n-- Stratified 80/20: DTW+k-NN --')
res_dtw_knn = run_stratified_dtw_knn(Xc_train, Xc_test, y_train, y_test, CLASS_NAMES, RESULTS_DIR)

summary = pd.DataFrame([
    {
        'approach': 'RF',
        'accuracy': res_rf['accuracy'],
        'macro_f1': res_rf['macro_f1'],
    },
    {
        'approach': 'PCA+GBM',
        'accuracy': res_pca_gbm['accuracy'],
        'macro_f1': res_pca_gbm['macro_f1'],
    },
    {
        'approach': '1D-CNN',
        'accuracy': res_cnn['accuracy'],
        'macro_f1': res_cnn['macro_f1'],
    },
    {
        'approach': 'BiLSTM',
        'accuracy': res_bilstm['accuracy'],
        'macro_f1': res_bilstm['macro_f1'],
    },
    {
        'approach': 'DTW+k-NN',
        'accuracy': res_dtw_knn['accuracy'],
        'macro_f1': res_dtw_knn['macro_f1'],
    },
]).sort_values('macro_f1', ascending=False).reset_index(drop=True)

print('\n' + '=' * 56)
print('Stratified supervised performance (cycle-level)')
print('=' * 56)
print(summary.to_string(index=False))
print('=' * 56)

out_summary = os.path.join(RESULTS_DIR, 'stratified_supervised_summary.csv')
summary.to_csv(out_summary, index=False)
print(f'Summary saved: {out_summary}')

print('Results saved to', RESULTS_DIR)