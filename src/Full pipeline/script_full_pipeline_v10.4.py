import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bisect import bisect_right
from scipy.signal import savgol_filter, find_peaks

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    'PCA_VAR':            0.95,
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
    'UNKNOWN_CLASS':      'unknown',
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
    return cfg['UNKNOWN_CLASS']

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

def build_feature_matrix(X_cycles, cfg):
    fs = cfg['FS']
    win = cfg['WINDOW']
    n, n_ch = len(X_cycles), X_cycles.shape[1]
    F = np.zeros((n, n_ch * win), dtype=np.float32)
    for i in range(n):
        cycle = X_cycles[i].copy()          # (12, 32)
        for ch in range(n_ch):
            mu  = cycle[ch].mean()
            std = cycle[ch].std()
            cycle[ch] = (cycle[ch] - mu) / std if std > 1e-6 else cycle[ch] - mu
        F[i] = cycle.ravel()
    print(f'Window: {win} samples = {win/fs:.3f}s at {fs} Hz')
    print(f'Feature matrix: {F.shape} (12 x {win} per-cycle normalized + flattened)')
    return F

def build_class_list(cfg):
    classes = list(cfg['SUPERVISED_CLASSES'])
    unk = str(cfg.get('UNKNOWN_CLASS', 'unknown'))
    if unk not in classes:
        classes.append(unk)
    return classes

def encode_labels(y_str, class_names):
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([cls_to_idx.get(y, cls_to_idx[class_names[-1]]) for y in y_str], dtype=np.int32)
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

def run_loso_pca_gbm(X_feat, y, groups, class_names, cfg, results_dir):
    uniq = np.unique(groups)
    y_true_all, y_pred_all = [], []

    for fi, g in enumerate(uniq, 1):
        tr, te = groups != g, groups == g
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]
        if len(yte) == 0:
            continue

        if len(np.unique(ytr)) < 2:
            pred = np.full_like(yte, fill_value=int(ytr[0]))
            y_true_all.extend(yte.tolist())
            y_pred_all.extend(pred.tolist())
            print(f'  [PCA+GBM] fold {fi}/{len(uniq)} (single-class fallback)')
            continue

        sc = StandardScaler()
        Xts = sc.fit_transform(Xtr)
        Xes = sc.transform(Xte)

        pca = PCA(n_components=float(cfg.get('PCA_VAR', 0.95)), svd_solver='full')
        Xtp = pca.fit_transform(Xts)
        Xep = pca.transform(Xes)

        clf = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=42,
        )
        clf.fit(Xtp, ytr)
        pred = clf.predict(Xep)

        y_true_all.extend(yte.tolist())
        y_pred_all.extend(pred.tolist())
        print(f'  [PCA+GBM] fold {fi}/{len(uniq)}')

    y_true = np.array(y_true_all, dtype=np.int32)
    y_pred = np.array(y_pred_all, dtype=np.int32)
    return _save_supervised_eval(y_true, y_pred, class_names, 'pca_gbm_loso', results_dir)

def run_loso_rf(X_feat, y, groups, class_names, results_dir):
    uniq = np.unique(groups)
    y_true_all, y_pred_all = [], []

    for fi, g in enumerate(uniq, 1):
        tr, te = groups != g, groups == g
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]
        if len(yte) == 0:
            continue

        if len(np.unique(ytr)) < 2:
            pred = np.full_like(yte, fill_value=int(ytr[0]))
            y_true_all.extend(yte.tolist())
            y_pred_all.extend(pred.tolist())
            print(f'  [RF] fold {fi}/{len(uniq)} (single-class fallback)')
            continue

        sc = StandardScaler()
        Xts = sc.fit_transform(Xtr)
        Xes = sc.transform(Xte)

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=24,
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(Xts, ytr)
        pred = clf.predict(Xes)

        y_true_all.extend(yte.tolist())
        y_pred_all.extend(pred.tolist())
        print(f'  [RF] fold {fi}/{len(uniq)}')

    y_true = np.array(y_true_all, dtype=np.int32)
    y_pred = np.array(y_pred_all, dtype=np.int32)
    return _save_supervised_eval(y_true, y_pred, class_names, 'rf_loso', results_dir)

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
# DIAGNOSTIC: what raw labels became "unknown"?
from collections import Counter
ann_cache = {}
unknown_raw = []
for entry in SESSIONS:
    sid = entry[2]
    if sid not in ann_cache:
        ann_cache[sid] = _load_time_labels_for_session(sid)
    ann = ann_cache[sid]
    if ann is None:
        continue
    for seg in ann['segments']:
        raw = seg[2]  # already canonicalized
        mapped = map_to_supervised_class(raw, DIRECT_CFG)
        if mapped == 'unknown':
            unknown_raw.append(raw)
print(f'\nRaw labels mapped to "unknown": {Counter(unknown_raw)}')

X_feat = build_feature_matrix(X_cycles, DIRECT_CFG)

CLASS_NAMES = build_class_list(DIRECT_CFG)
y_idx, CLASS_TO_IDX = encode_labels(y_labels, CLASS_NAMES)

print(f'Using {len(CLASS_NAMES)} classes (including unknown):')
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

print('\n-- LOSO: PCA + GBM --')
res_pca = run_loso_pca_gbm(X_feat, y_idx, groups, CLASS_NAMES, DIRECT_CFG, RESULTS_DIR)

print('\n-- LOSO: Random Forest --')
res_rf = run_loso_rf(X_feat, y_idx, groups, CLASS_NAMES, RESULTS_DIR)

summary = pd.DataFrame([
    {'approach': 'PCA + GBM', 'accuracy': res_pca['accuracy'], 'macro_f1': res_pca['macro_f1']},
    {'approach': 'Random Forest', 'accuracy': res_rf['accuracy'], 'macro_f1': res_rf['macro_f1']},
]).sort_values('macro_f1', ascending=False).reset_index(drop=True)
summary.index += 1

print('\n' + '=' * 56)
print('LOSO supervised performance (cycle-level)')
print('=' * 56)
print(summary.to_string())
print('=' * 56)

out_summary = os.path.join(RESULTS_DIR, 'loso_supervised_summary.csv')
summary.to_csv(out_summary, index=False)
print(f'Summary saved: {out_summary}')

fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(summary['approach'][::-1], summary['macro_f1'][::-1], color=['#4c956c', '#2c6e91'])
ax.set_xlabel('Macro-F1')
ax.set_xlim(0, 1.0)
ax.set_title('LOSO macro-F1 comparison (supervised)')
for i, row in summary[::-1].reset_index(drop=True).iterrows():
    ax.text(row['macro_f1'] + 0.01, i, f"{row['macro_f1']:.3f}", va='center', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'fig_loso_supervised_comparison.png'), bbox_inches='tight')
plt.close(fig)

print('Results saved to', RESULTS_DIR)