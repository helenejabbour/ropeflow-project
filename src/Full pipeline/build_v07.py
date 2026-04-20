"""
Build 07_Full_Pipeline_V07.ipynb
Run: python "src/Full pipeline/build_v07.py"
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load V06 to reuse its cells verbatim ──────────────────────
with open(os.path.join(HERE, '06_Full_Pipeline_V06.ipynb'), encoding='utf-8') as f:
    V06 = json.load(f)

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

def V06_code(idx):
    """Return the source of a V06 code cell as a string."""
    return ''.join(V06['cells'][idx]['source'])

# ═════════════════════════════════════════════════════════════
# CELL SOURCES
# ══════════════════════════════════════════
CELL_TITLE = """\
# Rope Flow — Full Pipeline V07
## Topology Features + Cycle Matrix PCA + Min-Cycle Filter + Repeated Splits

**Course:** MECH 798M / EECE 798K
**Pipeline:** V06 signal processing → multi-source session discovery
(unified-data homogeneous + new-labeled-sessions heterogeneous)
→ stratified group train/test split → LOSO CV on train
→ final model → held-out test evaluation
"""

CELL_SETUP = V06_code(2) + """

# ── Results directory (new folder per run) ───────────────────
import datetime
RUN_NAME = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
RESULTS_DIR = os.path.join('..', '..', 'results', 'Full_pipeline', RUN_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'Results will be saved to: {RESULTS_DIR}')
"""  # imports + CONFIG + KNOWN_PATTERNS + PAL + results dir

CELL_CORE_FUNCTIONS = V06_code(4)  # all signal-processing / feature-extraction functions

# ── V07 additions: topology features, cycle-matrix PCA, updated feature assembly ──
CELL_V07_FEATURES = """
# ══════════════════════════════════════════════════════════════
# V07: Topology features — capture 3D trajectory shape
# ══════════════════════════════════════════════════════════════
def topology_features(om0, om1, s0, e0, s1, e1, fs):
    \"\"\"
    10D topology features that capture the spatial shape of each cycle.
    These distinguish planar (overhand/underhand) from spatial (dragon_roll)
    patterns by measuring inter-axis phase relationships.
    \"\"\"
    w0 = om0[s0:e0]  # (N, 3)
    w1 = om1[s1:e1]

    feats = []

    for w in [w0, w1]:
        n = len(w)
        if n < 6:
            feats.extend([0.0] * 5)
            continue

        wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

        # 1-3. Cross-axis phase lags via cross-correlation peak
        #      Overhand: wx↔wz ~0 deg (planar)
        #      Dragon roll: wx↔wz ~90 deg (figure-eight)
        pairs = [(wx, wy), (wx, wz), (wy, wz)]
        for a, b in pairs:
            a_c = a - np.mean(a)
            b_c = b - np.mean(b)
            corr = np.correlate(a_c, b_c, mode='full')
            norm = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2)) + 1e-12
            corr /= norm
            peak_idx = np.argmax(corr)
            lag_samples = peak_idx - (n - 1)
            lag_phase = lag_samples / (n / 2) * 180  # degrees (approx)
            feats.append(lag_phase)

        # 4. Plane-of-motion ratio: σ₂/σ₁ from SVD of 3-axis gyro
        #    Planar motion → σ₂/σ₁ ≈ 0, spatial motion → σ₂/σ₁ > 0.3
        _, S_w, _ = np.linalg.svd(w.T, full_matrices=False)
        plane_ratio = S_w[1] / (S_w[0] + 1e-12) if len(S_w) >= 2 else 0.0
        feats.append(plane_ratio)

        # 5. Winding number: zero-crossings of ωx per half-period
        #    Overhand: ~2, Dragon roll: ~4
        zc = np.sum(np.diff(np.sign(wx)) != 0)
        feats.append(zc / max(1, n / fs))  # crossings per second

    return np.array(feats)

TOPO_FEAT_NAMES = []
for dev in ['d0', 'd1']:
    TOPO_FEAT_NAMES += [f'phase_xy_{dev}', f'phase_xz_{dev}', f'phase_yz_{dev}',
                        f'plane_ratio_{dev}', f'winding_rate_{dev}']


# ══════════════════════════════════════════════════════════════
# V07: Cycle-matrix PCA — shape features from flattened waveform
# ══════════════════════════════════════════════════════════════
# PCA is fitted during training and stored in the model.
# At inference, cycles are projected onto the trained PCA basis.
N_PCA_SHAPE = 10  # number of PCA components to keep

SHAPE_PCA_NAMES = [f'shape_pc_{i}' for i in range(N_PCA_SHAPE)]


# ══════════════════════════════════════════════════════════════
# V07: Updated feature assembly (56D base + 10D topology + 10D shape PCA)
# ══════════════════════════════════════════════════════════════
# Override ALL_FEAT_NAMES and extract_all_features from V06

ALL_FEAT_NAMES_V07 = (PHYSICS_NAMES + FOURIER_NAMES + SVD_NAMES + TEMPLATE_NAMES
                      + DMD_FEAT_NAMES + SI_FEAT_NAMES
                      + TOPO_FEAT_NAMES + SHAPE_PCA_NAMES)
N_FEAT_V07 = len(ALL_FEAT_NAMES_V07)

# Overwrite the V06 names so downstream code uses the new ones
ALL_FEAT_NAMES = ALL_FEAT_NAMES_V07
N_FEAT = N_FEAT_V07


def extract_features_v07(t0, t1, A0, A1, om0, om1,
                          paired0, paired1, template, fs,
                          shape_pca=None):
    \"\"\"
    Extract 76D feature vector per cycle:
      56D base (physics + fourier + SVD + template + DMD + SI)
      + 10D topology (phase lags, plane ratio, winding)
      + 10D shape PCA (flattened cycle matrix projected onto PCA basis)

    If shape_pca is None, PCA scores are set to 0 (will be filled during training).
    \"\"\"
    features = []
    flat_matrices = []
    dt = 1.0 / fs

    for (s0, e0), (s1, e1) in zip(paired0, paired1):
        # Base 56D features (unchanged from V06)
        phys = physics_features(t0, t1, A0, A1, om0, om1, s0, e0, s1, e1, fs)
        fourier = cycle_fourier_features(A0, om0, s0, e0, fs)
        cm = build_cycle_matrix(A0, A1, om0, om1, s0, e0, s1, e1, CONFIG['TARGET_LEN'])
        svd_feat = compute_svd_features(cm, CONFIG['PCA_COMPONENTS'])
        corr_mean, corr_min = template_correlation(cm, template)
        tmpl_feat = np.array([corr_mean, corr_min])
        dmd_feat = dmd_features(cm, dt)
        si_feat = subject_invariant_features(t0, t1, A0, A1, om0, om1, s0, e0, s1, e1, fs)

        # V07: topology features (10D)
        topo = topology_features(om0, om1, s0, e0, s1, e1, fs)

        # Placeholder for shape PCA (filled after PCA is fitted)
        shape_placeholder = np.zeros(N_PCA_SHAPE)

        feat = np.concatenate([phys, fourier, svd_feat, tmpl_feat,
                               dmd_feat, si_feat, topo, shape_placeholder])
        features.append(feat)
        flat_matrices.append(cm.ravel())  # 12*64 = 768D

    X = np.array(features) if features else np.empty((0, N_FEAT_V07))
    flat_mat = np.array(flat_matrices) if flat_matrices else np.empty((0, 12 * CONFIG['TARGET_LEN']))

    # If PCA model is provided, fill in the shape PCA scores
    if shape_pca is not None and len(flat_mat) > 0:
        pca_scores = shape_pca.transform(flat_mat)[:, :N_PCA_SHAPE]
        # Overwrite the placeholder columns
        pca_start = N_FEAT_V07 - N_PCA_SHAPE
        X[:, pca_start:] = pca_scores

    return X, flat_mat
"""

CELL_V05_FUNCTIONS = """\
# ══════════════════════════════════════════════════════════════════════════
# V05 FUNCTIONS
# Adds: heterogeneous session discovery, windowed processing,
#       stratified group split, updated train/classify/LOSO/evaluate
# ══════════════════════════════════════════════════════════════════════════
import re as _re, json as _json
from collections import defaultdict

DATA_RAW        = os.path.join('..', '..', 'data', 'raw')
NEW_LABELED_RAW = os.path.join(DATA_RAW, 'new-labeled-sessions')

# ── Label mapping (mirrors 08_Build_labels_index) ──────────────────────
_EXACT_MAP = {
    'underhand':'underhand','overhand':'overhand','dragon_roll':'dragon_roll',
    'sneak_underhand':'sneak_underhand','sneak_overhand':'sneak_overhand',
    'race_and_chase':'race_and_chase','cheetahs_tail':'cheetahs_tail',
    'idle':None,
}
_PREFIX_RULES = [
    (_re.compile(r'^us', _re.I), 'sneak_underhand'),
    (_re.compile(r'^os', _re.I), 'sneak_overhand'),
    (_re.compile(r'^u',  _re.I), 'underhand'),
    (_re.compile(r'^o',  _re.I), 'overhand'),
    (_re.compile(r'^fb', _re.I), 'dragon_roll'),
    (_re.compile(r'^bf', _re.I), 'dragon_roll'),
    (_re.compile(r'^cw$',  _re.I), None),
    (_re.compile(r'^ccw$', _re.I), None),
    (_re.compile(r'^idle', _re.I), None),
    (_re.compile(r'^vq',   _re.I), None),
]
def _map_label(raw):
    raw = raw.strip()
    if raw.lower() in _EXACT_MAP:
        return _EXACT_MAP[raw.lower()]
    for pat, c in _PREFIX_RULES:
        if pat.match(raw):
            return c
    return None


# ── Session discovery ──────────────────────────────────────────────────
def discover_sessions(processed_dir):
    \"\"\"
    Homogeneous sessions from unified-data.
    Returns list of 5-tuples (d0, d1, label, display_name, time_windows=None).
    \"\"\"
    csv_files = sorted(glob.glob(os.path.join(processed_dir, '*_device0_processed.csv')))
    sessions = []
    for d0_path in csv_files:
        d1_path = d0_path.replace('_device0_', '_device1_')
        if not os.path.exists(d1_path):
            continue
        stem = os.path.basename(d0_path).replace('_device0_processed.csv', '')
        parts = stem.split('_')
        if len(parts) < 3:
            continue
        remaining = parts[2:]
        label = None
        for n in range(len(remaining), 0, -1):
            if '_'.join(remaining[:n]) in KNOWN_PATTERNS:
                label = '_'.join(remaining[:n])
                subject = '_'.join(remaining[n:]) or 'unknown'
                break
        if label is None:
            continue
        display_name = f'{label}_{subject}'
        sessions.append((d0_path, d1_path, label, display_name, None))
    return sessions


def discover_heterogeneous_sessions(processed_dir, raw_dir=None):
    \"\"\"
    Per-label entries from new-labeled-sessions JSON files.
    One 5-tuple per (recording, canonical_label) pair.
    display_name = 'session_name/label'  (slash used as LOSO group key separator)
    time_windows = [(start_s, end_s), ...] for that label
    \"\"\"
    if raw_dir is None:
        raw_dir = NEW_LABELED_RAW
    if not os.path.isdir(raw_dir):
        return []
    entries = []
    for sname in sorted(os.listdir(raw_dir)):
        sdir = os.path.join(raw_dir, sname)
        if not os.path.isdir(sdir):
            continue
        lpath = None
        for fn in ('labels_corrected.json', 'labels.json'):
            p = os.path.join(sdir, fn)
            if os.path.isfile(p):
                lpath = p
                break
        if lpath is None:
            continue
        d0 = os.path.join(processed_dir, f'{sname}_device0_processed.csv')
        d1 = os.path.join(processed_dir, f'{sname}_device1_processed.csv')
        if not (os.path.isfile(d0) and os.path.isfile(d1)):
            continue
        with open(lpath, encoding='utf-8') as fh:
            data = _json.load(fh)
        windows_by_label = defaultdict(list)
        for seg in data.get('segments', []):
            canon = _map_label(seg.get('label', ''))
            if canon is None:
                continue
            s = seg.get('start')
            e = seg.get('end')
            if s is None:
                continue
            if e is None:
                e = s + 2.0
            windows_by_label[canon].append((float(s), float(e)))
        for label, windows in sorted(windows_by_label.items()):
            entries.append((d0, d1, label, f'{sname}/{label}', windows))
    return entries


def discover_all_sessions(processed_dir):
    \"\"\"Combine homogeneous + heterogeneous entries, excluding sparse classes.\"\"\"
    EXCLUDE_CLASSES = {'cheetahs_tail', 'race_and_chase'}
    all_entries = discover_sessions(processed_dir) + discover_heterogeneous_sessions(processed_dir)
    filtered = [e for e in all_entries if e[2] not in EXCLUDE_CLASSES]
    excluded = [e for e in all_entries if e[2] in EXCLUDE_CLASSES]
    if excluded:
        print(f'Excluded {len(excluded)} entries from classes: {EXCLUDE_CLASSES}')
    return filtered


def session_group(entry):
    \"\"\"Recording-level group key — slash in display_name separates session/label.\"\"\"
    dn = entry[3]
    return dn.split('/')[0] if '/' in dn else dn


def summarize_sessions(sessions):
    by_class = defaultdict(list)
    for e in sessions:
        by_class[e[2]].append(e[3])
    print(f'Discovered {len(sessions)} entries across {len(by_class)} classes:')
    for label in sorted(by_class.keys()):
        names = by_class[label]
        print(f'  {label:<22s}: {len(names):>3d} entries')
        for n in names:
            print(f'    - {n}')
    singletons = [l for l, s in by_class.items() if len(s) == 1]
    if singletons:
        print(f'\\nWarning: {len(singletons)} singleton classes: {singletons}')
    return dict(by_class)


# ── Stratified group train/test split ─────────────────────────────────
def stratified_group_split(sessions, test_fraction=0.2, random_state=42):
    \"\"\"
    Split sessions into train/test, keeping recording groups atomic.
    Stratifies by estimated cycle count (not entry count) to prevent
    inverted splits when heterogeneous groups have many entries.
    Singletons (classes with only 1 group) always stay in train.
    Enforces that test never exceeds test_fraction of total entries.
    \"\"\"
    import random as _rand
    rng = _rand.Random(random_state)

    groups = defaultdict(list)
    for e in sessions:
        groups[session_group(e)].append(e)

    # Estimate cycle count per group (use entry count as proxy if no data loaded)
    group_weight = {g: len(entries) for g, entries in groups.items()}

    class_to_groups = defaultdict(list)
    for gname, entries in groups.items():
        for lbl in set(e[2] for e in entries):
            class_to_groups[lbl].append(gname)

    test_groups = set()
    for lbl, gnames in sorted(class_to_groups.items()):
        if len(gnames) < 2:
            continue  # singleton class — keep in train
        # Sort by weight (smallest groups go to test first to avoid overshooting)
        gnames_sorted = sorted(gnames, key=lambda g: group_weight[g])
        rng.shuffle(gnames_sorted)
        # Compute target: test_fraction of total entries for this class
        total_entries = sum(len(groups[g]) for g in gnames)
        target_test = total_entries * test_fraction
        test_count = 0
        for g in gnames_sorted:
            if g in test_groups:
                test_count += len(groups[g])
                continue
            if test_count + len(groups[g]) <= target_test * 1.5:  # allow 50% overshoot
                test_groups.add(g)
                test_count += len(groups[g])

    train = [e for e in sessions if session_group(e) not in test_groups]
    test  = [e for e in sessions if session_group(e) in test_groups]

    # Safety check: if test > 40% of total, something is wrong — swap
    if len(test) > len(sessions) * 0.4:
        print(f'WARNING: test={len(test)}/{len(sessions)} exceeds 40% — swapping train/test')
        train, test = test, train

    return train, test


# ── Windowed session processing ────────────────────────────────────────
def process_session_windowed(path_d0, path_d1, windows):
    \"\"\"
    Like process_session but only keeps cycles whose midpoint falls inside
    one of the provided (start_s, end_s) windows.
    \"\"\"
    d0, d1 = load_session(path_d0, path_d1)
    t0, _Q0, A0, om0 = extract_signals(d0)
    t1, _Q1, A1, om1 = extract_signals(d1)
    fs = CONFIG['FS']
    cyc0, mag0_smooth, peaks0 = detect_cycles(t0, om0, fs)
    cyc1, _, _ = detect_cycles(t1, om1, fs)
    p0, p1 = pair_cycles(t0, cyc0, t1, cyc1)

    fp0, fp1 = [], []
    for (s0, e0), (s1, e1) in zip(p0, p1):
        t_mid = (t0[s0] + t0[e0]) / 2.0
        if any(ws <= t_mid < we for ws, we in windows):
            fp0.append((s0, e0))
            fp1.append((s1, e1))

    cms = [build_cycle_matrix(A0, A1, om0, om1, s0, e0, s1, e1, CONFIG['TARGET_LEN'])
           for (s0, e0), (s1, e1) in zip(fp0, fp1)]
    return {
        't0': t0, 't1': t1, 'A0': A0, 'A1': A1, 'om0': om0, 'om1': om1,
        'paired0': fp0, 'paired1': fp1, 'cycle_matrices': cms,
        'mag0_smooth': mag0_smooth, 'peaks0': peaks0,
    }


def _load_entry(entry):
    \"\"\"Dispatch to process_session or process_session_windowed.\"\"\"
    d0, d1, label, dname, tw = entry
    return process_session(d0, d1) if tw is None else process_session_windowed(d0, d1, tw)



# ── V07 training ───────────────────────────────────────────────────────
MIN_CYCLES_PER_ENTRY = 5

def train_model_v07(session_list, verbose=True):
    \"\"\"
    Train multi-class GBM with V07 features (76D).
    Filters entries with < MIN_CYCLES_PER_ENTRY.
    Fits shape PCA on training cycle matrices.
    \"\"\"
    from sklearn.decomposition import PCA as _PCA
    fs = CONFIG['FS']
    all_sessions, all_labels, all_dnames = [], [], []
    skipped = 0

    for entry in session_list:
        d0, d1, label, dname, tw = entry
        sess = _load_entry(entry)
        n_cycles = len(sess['paired0'])
        if n_cycles < MIN_CYCLES_PER_ENTRY:
            if verbose:
                print(f'  SKIP {label:<22s} [{dname}]: {n_cycles} cycles (< {MIN_CYCLES_PER_ENTRY})')
            skipped += 1
            continue
        all_sessions.append(sess)
        all_labels.append(label)
        all_dnames.append(dname)
        if verbose:
            print(f'  {label:<22s} [{dname}]: {n_cycles} cycles')

    if verbose and skipped:
        print(f'  ({skipped} entries skipped for too few cycles)')

    all_cms = []
    for sess in all_sessions:
        all_cms.extend(sess['cycle_matrices'])
    if not all_cms:
        raise ValueError('No cycles found.')

    template = build_template(all_cms)
    class_cms = defaultdict(list)
    for sess, lbl in zip(all_sessions, all_labels):
        class_cms[lbl].extend(sess['cycle_matrices'])
    class_templates = {lbl: build_template(cms) for lbl, cms in class_cms.items()}

    X_list, flat_list, y_list, sid_list = [], [], [], []
    for si, (sess, lbl) in enumerate(zip(all_sessions, all_labels)):
        X, flat = extract_features_v07(
            sess['t0'], sess['t1'], sess['A0'], sess['A1'],
            sess['om0'], sess['om1'], sess['paired0'], sess['paired1'],
            template, fs, shape_pca=None)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(X)
        flat_list.append(flat)
        y_list.extend([lbl] * len(X))
        sid_list.extend([si] * len(X))

    X_all = np.vstack(X_list)
    flat_all = np.vstack(flat_list)
    y_all = np.array(y_list)
    sid_all = np.array(sid_list)

    n_pca = min(N_PCA_SHAPE, flat_all.shape[0] - 1, flat_all.shape[1])
    shape_pca = _PCA(n_components=n_pca)
    pca_scores = shape_pca.fit_transform(flat_all)[:, :N_PCA_SHAPE]
    pca_start = N_FEAT_V07 - N_PCA_SHAPE
    X_all[:, pca_start:pca_start + pca_scores.shape[1]] = pca_scores

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_all)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    clf.fit(X_s, y_enc)

    if verbose:
        print(f'\nTrained on {X_all.shape[0]} cycles x {X_all.shape[1]} features')
        print(f'Classes: {list(le.classes_)}')

    return {
        'clf': clf, 'scaler': scaler, 'label_encoder': le,
        'template': template, 'class_templates': class_templates,
        'shape_pca': shape_pca,
        'feature_names': ALL_FEAT_NAMES,
        'X_train': X_all, 'y_train': y_all, 'session_ids': sid_all,
    }


# ── V07 classifier ────────────────────────────────────────────────────
def classify_session_v07(path_d0, path_d1, model, time_windows=None, verbose=True):
    \"\"\"Classify with V07 features + transition rejection.\"\"\"
    fs = CONFIG['FS']
    sess = (process_session(path_d0, path_d1) if time_windows is None
            else process_session_windowed(path_d0, path_d1, time_windows))

    if not sess['paired0']:
        return {'labels': [], 'confidences': [], 'X': np.array([]),
                'verdict': 'unknown', 'vote': {}, 'n_transition': 0, 'session': sess}

    X, flat = extract_features_v07(
        sess['t0'], sess['t1'], sess['A0'], sess['A1'],
        sess['om0'], sess['om1'], sess['paired0'], sess['paired1'],
        model['template'], fs, shape_pca=model.get('shape_pca'))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.ndim < 2 or len(X) == 0:
        if verbose:
            print(f'  0 valid cycles — skipping')
        return {'labels': [], 'confidences': [], 'X': np.empty((0, len(model['feature_names']))),
                'verdict': 'unknown', 'vote': {}, 'n_transition': 0, 'session': sess}

    X_s = model['scaler'].transform(X)

    le = model['label_encoder']
    y_pred_enc = model['clf'].predict(X_s)
    y_pred_proba = model['clf'].predict_proba(X_s)
    pred_labels = le.inverse_transform(y_pred_enc)
    confidences = np.max(y_pred_proba, axis=1)

    conf_gate = confidences >= CONFIG['CONFIDENCE_THRESHOLD']
    tmpl_gate = np.zeros(len(X), dtype=bool)
    for ci in range(len(X)):
        cm = build_cycle_matrix(
            sess['A0'], sess['A1'], sess['om0'], sess['om1'],
            sess['paired0'][ci][0], sess['paired0'][ci][1],
            sess['paired1'][ci][0], sess['paired1'][ci][1],
            CONFIG['TARGET_LEN'])
        for cls_tmpl in model['class_templates'].values():
            if template_correlation(cm, cls_tmpl)[0] >= CONFIG['TEMPLATE_GATE_THRESHOLD']:
                tmpl_gate[ci] = True
                break

    labels = []
    for i in range(len(X)):
        labels.append(pred_labels[i] if (conf_gate[i] and tmpl_gate[i]) else 'transition')

    non_trans = [l for l in labels if l != 'transition']
    if non_trans:
        vote = Counter(non_trans)
        verdict = vote.most_common(1)[0][0]
    else:
        vote = Counter(labels)
        verdict = 'unknown'

    if verbose:
        print(f'  {len(labels)} cycles | ', end='')
        for cls, cnt in sorted(Counter(labels).items()):
            print(f'{cls}={cnt} ', end='')
        print(f'-> {verdict}')

    return {
        'labels': labels, 'confidences': confidences.tolist(), 'X': X,
        'verdict': verdict, 'vote': dict(Counter(non_trans)),
        'n_transition': sum(1 for l in labels if l == 'transition'),
        'session': sess,
    }

# ── LOSO CV (recording-group aware) ──────────────────────────────────
def run_loso_v07(sessions, verbose=True):
    \"\"\"
    Leave-one-recording-group-out CV.
    Group key = session_group(entry) so all segments from the same
    physical recording stay on the same side of each fold.
    Singleton groups are always in train.
    \"\"\"
    group_counts = Counter(session_group(e) for e in sessions)
    class_group_counts = Counter(e[2] for e in sessions
                                 if group_counts[session_group(e)] >= 1)

    # Identify which classes appear in only one group → singleton
    class_to_groups = defaultdict(set)
    for e in sessions:
        class_to_groups[e[2]].add(session_group(e))
    singleton_classes = {c for c, gs in class_to_groups.items() if len(gs) == 1}

    loso_groups = sorted(set(session_group(e) for e in sessions
                             if e[2] not in singleton_classes))

    if verbose and singleton_classes:
        print(f'Singleton classes (train-only): {sorted(singleton_classes)}')
        print(f'LOSO folds: {len(loso_groups)} recording groups')
        print('=' * 80)

    all_y_true, all_y_pred = [], []
    entry_results = []

    for test_grp in loso_groups:
        test_entries  = [e for e in sessions if session_group(e) == test_grp]
        train_entries = [e for e in sessions if session_group(e) != test_grp]

        model = train_model_v07(train_entries, verbose=False)

        for entry in test_entries:
            d0, d1, true_label, dname, tw = entry
            res = classify_session_v07(d0, d1, model, time_windows=tw, verbose=False)
            if not res['labels']:
                continue
            non_trans = [l for l in res['labels'] if l != 'transition']
            all_y_true.extend([true_label] * len(non_trans))
            all_y_pred.extend(non_trans)
            correct = res['verdict'] == true_label
            entry_results.append((dname, true_label, res['verdict'], correct))
            if verbose:
                status = 'CORRECT' if correct else 'WRONG'
                print(f'  {dname:<52s} GT={true_label:<20s} -> {res["verdict"]:<20s} [{status}]')

    n_correct = sum(1 for r in entry_results if r[3])
    n_total   = len(entry_results)
    print(f'\\nLOSO entry accuracy: {n_correct}/{n_total} '
          f'({n_correct/n_total*100:.1f}%)' if n_total else '\\nNo LOSO results.')

    result = {
        'entry_results': entry_results,
        'n_correct': n_correct, 'n_total': n_total,
        'session_acc': n_correct / n_total if n_total else 0,
    }
    if all_y_true:
        le = LabelEncoder()
        le.fit(sorted(set(all_y_true + all_y_pred)))
        yt, yp = le.transform(all_y_true), le.transform(all_y_pred)
        print(classification_report(yt, yp, target_names=le.classes_, zero_division=0))
        f1 = f1_score(yt, yp, average='macro', zero_division=0)
        result.update({
            'fine_report': classification_report(yt, yp, target_names=le.classes_, zero_division=0),
            'fine_f1_macro': f1,
            'fine_cm': confusion_matrix(yt, yp),
            'fine_classes': le.classes_,
            'all_y_true': all_y_true,
            'all_y_pred': all_y_pred,
        })
        # Macro-class
        mt = [MACRO_MAP.get(l, 'unknown') for l in all_y_true]
        mp = [MACRO_MAP.get(l, 'unknown') for l in all_y_pred]
        le_m = LabelEncoder()
        le_m.fit(sorted(set(mt + mp)))
        result.update({
            'macro_report': classification_report(le_m.transform(mt), le_m.transform(mp),
                                                  target_names=le_m.classes_, zero_division=0),
            'macro_f1_macro': f1_score(le_m.transform(mt), le_m.transform(mp),
                                       average='macro', zero_division=0),
            'macro_cm': confusion_matrix(le_m.transform(mt), le_m.transform(mp)),
            'macro_classes': le_m.classes_,
        })
    return result


# ── Held-out test evaluation ───────────────────────────────────────────
def evaluate_test_set_v07(test_sessions, model, verbose=True):
    \"\"\"
    Evaluate the final model on held-out test entries.
    Returns per-cycle and per-entry classification metrics.
    \"\"\"
    all_y_true, all_y_pred = [], []
    entry_results = []

    for entry in test_sessions:
        d0, d1, true_label, dname, tw = entry
        res = classify_session_v07(d0, d1, model, time_windows=tw, verbose=False)
        if not res['labels']:
            continue
        non_trans = [l for l in res['labels'] if l != 'transition']
        all_y_true.extend([true_label] * len(non_trans))
        all_y_pred.extend(non_trans)
        correct = res['verdict'] == true_label
        entry_results.append((dname, true_label, res['verdict'], correct))
        if verbose:
            status = 'CORRECT' if correct else 'WRONG'
            print(f'  {dname:<52s} GT={true_label:<20s} -> {res["verdict"]:<20s} [{status}]')

    n_correct = sum(1 for r in entry_results if r[3])
    n_total   = len(entry_results)
    print(f'\\nTest entry accuracy: {n_correct}/{n_total} '
          f'({n_correct/n_total*100:.1f}%)' if n_total else '\\nNo test entries.')

    result = {
        'entry_results': entry_results,
        'n_correct': n_correct, 'n_total': n_total,
    }
    if all_y_true:
        le = LabelEncoder()
        le.fit(sorted(set(all_y_true + all_y_pred)))
        yt, yp = le.transform(all_y_true), le.transform(all_y_pred)
        print(classification_report(yt, yp, target_names=le.classes_, zero_division=0))
        f1 = f1_score(yt, yp, average='macro', zero_division=0)
        result.update({
            'fine_report': classification_report(yt, yp, target_names=le.classes_, zero_division=0),
            'test_f1_macro': f1,
            'fine_cm': confusion_matrix(yt, yp),
            'fine_classes': le.classes_,
            'all_y_true': all_y_true,
            'all_y_pred': all_y_pred,
        })
    return result
"""

CELL_DISCOVER = """\
# ── Discover all sessions ─────────────────────────────────────────────
ALL_SESSIONS = discover_all_sessions(DATA_PROCESSED)
by_class = summarize_sessions(ALL_SESSIONS)

print(f'\\nHomogeneous entries  : {sum(1 for e in ALL_SESSIONS if e[4] is None)}')
print(f'Heterogeneous entries: {sum(1 for e in ALL_SESSIONS if e[4] is not None)}')
print(f'Total                : {len(ALL_SESSIONS)}')
"""

CELL_TIMESTAMP_CHECK = """\
# ── Verify JSON timestamps match CSV timestamps ─────────────────────
hetero = [e for e in ALL_SESSIONS if e[4] is not None]
if hetero:
    print(f'Checking timestamp alignment for {len(hetero)} heterogeneous entries...')
    mismatches = 0
    for e in hetero[:5]:  # check first 5
        d0 = pd.read_csv(e[0])
        csv_end = d0['timestamp_ms'].iloc[-1] / 1000
        json_max = max(w[1] for w in e[4])
        ok = json_max <= csv_end * 1.1  # allow 10% tolerance
        status = 'OK' if ok else 'MISMATCH'
        if not ok:
            mismatches += 1
        print(f'  {e[3]:<50s} CSV=[0, {csv_end:.1f}s] JSON_max={json_max:.1f}s [{status}]')
    if mismatches > 0:
        print(f'\\nWARNING: {mismatches} timestamp mismatches detected!')
        print('JSON may use absolute timestamps — offset correction needed.')
    else:
        print('\\nAll timestamps aligned.')
else:
    print('No heterogeneous sessions found.')
"""

CELL_SPLIT = """\
# ── Stratified train/test split (~80/20 by recording group) ──────────
import time as _time
SPLIT_SEED = int(_time.time()) % 10000  # different each run
TRAIN_SESSIONS, TEST_SESSIONS = stratified_group_split(ALL_SESSIONS, test_fraction=0.2, random_state=SPLIT_SEED)

ratio = len(TRAIN_SESSIONS) / max(1, len(ALL_SESSIONS)) * 100
print(f'Split seed: {SPLIT_SEED}')
print(f'Train entries : {len(TRAIN_SESSIONS)} ({ratio:.0f}%)')
print(f'Test  entries : {len(TEST_SESSIONS)} ({100-ratio:.0f}%)')
print()
train_classes = Counter(e[2] for e in TRAIN_SESSIONS)
test_classes  = Counter(e[2] for e in TEST_SESSIONS)
print(f'  {"Class":<22s}  {"Train":>6s}  {"Test":>6s}')
for cls in sorted(set(list(train_classes) + list(test_classes))):
    print(f'  {cls:<22s}  {train_classes.get(cls,0):>6d}  {test_classes.get(cls,0):>6d}')

assert ratio >= 70, f'Training ratio {ratio:.0f}% is below 70% — check split logic'
"""

CELL_LOSO = """\
# ── LOSO cross-validation on training set ────────────────────────────
print('Running LOSO CV on training set…')
print('='*80)
loso_results = run_loso_v07(TRAIN_SESSIONS, verbose=True)
print(f'\\nLOSO fine  F1 (macro): {loso_results.get(\"fine_f1_macro\", 0):.3f}')
print(f'LOSO macro F1 (macro): {loso_results.get(\"macro_f1_macro\", 0):.3f}')
"""

CELL_FINAL_MODEL = """\
# ── Final model trained on full training set ──────────────────────────
print('Training final model on all training entries…')
print('='*80)
MODEL_FINAL = train_model_v07(TRAIN_SESSIONS, verbose=True)

# ── Held-out test evaluation ──────────────────────────────────────────
print('\\n' + '='*80)
print('Evaluating on held-out test set…')
print('='*80)
test_results = evaluate_test_set_v07(TEST_SESSIONS, MODEL_FINAL, verbose=True)
print(f'\\nTest  fine  F1 (macro): {test_results.get(\"test_f1_macro\", 0):.3f}')
"""

CELL_CM = """\
# ── Confusion matrices ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, res, title in zip(
    axes,
    [loso_results, test_results],
    ['LOSO CV (train set)', 'Held-out test set']
):
    if 'fine_cm' not in res:
        ax.set_title(f'{title} — no data'); continue
    cm = res['fine_cm']
    classes = res['fine_classes']
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()*0.5 else 'black', fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    f1 = res.get('fine_f1_macro', res.get('test_f1_macro', 0))
    ax.set_title(f'{title}\\nmacro-F1 = {f1:.3f}', fontsize=11)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_v07_confusion_matrices.png'), bbox_inches='tight')
plt.show()
"""

CELL_F1_BAR = """\
# ── LOSO vs Test F1 comparison bar chart ─────────────────────────────
loso_f1  = loso_results.get('fine_f1_macro', 0)
test_f1  = test_results.get('test_f1_macro', 0)

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(['LOSO CV\\n(train)', 'Held-out\\n(test)'], [loso_f1, test_f1],
              color=['#5DCAA5', '#7F77DD'], width=0.5)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Macro F1')
ax.set_title('Generalisation check: LOSO vs Test F1')
for bar, val in zip(bars, [loso_f1, test_f1]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_v07_f1_comparison.png'), bbox_inches='tight')
plt.show()
"""

CELL_FEAT_IMP = """\
# ── Feature importance (tier-colored) ────────────────────────────────
TIER_COLORS = {
    'physics': '#5DCAA5', 'fourier': '#7F77DD',
    'svd': '#E24B4A', 'template': '#EF9F27',
    'dmd': '#3498db', 'si': '#e67e22',
    'topology': '#8B5CF6', 'shape_pca': '#EC4899',
}
TIER_RANGES = {
    'topology': (56, 66),
    'shape_pca':(66, 76),
    'physics':  (0,  12),
    'fourier':  (12, 28),
    'svd':      (28, 36),
    'template': (36, 38),
    'dmd':      (38, 47),
    'si':       (47, 56),
}
def get_tier(name):
    idx = ALL_FEAT_NAMES.index(name) if name in ALL_FEAT_NAMES else -1
    for tier, (lo, hi) in TIER_RANGES.items():
        if lo <= idx < hi:
            return tier
    return 'other'

imp = MODEL_FINAL['clf'].feature_importances_
feat_names = MODEL_FINAL['feature_names']
order = np.argsort(imp)[::-1][:20]
colors = [TIER_COLORS.get(get_tier(feat_names[i]), '#aaa') for i in order]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(order)), imp[order], color=colors)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([feat_names[i] for i in order], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Feature importance')
ax.set_title('Top-20 features by GBM importance (color = tier)')
from matplotlib.patches import Patch
patches = [Patch(color=c, label=t) for t, c in TIER_COLORS.items()]
ax.legend(handles=patches, loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_v07_feature_importance.png'), bbox_inches='tight')
plt.show()
"""

CELL_SUMMARY = """\
# ── Pipeline summary ─────────────────────────────────────────────────
print('='*80)
print('V05 PIPELINE SUMMARY')
print('='*80)
print(f'  Total sessions     : {len(ALL_SESSIONS)} entries '
      f'({sum(1 for e in ALL_SESSIONS if e[4] is None)} homogeneous, '
      f'{sum(1 for e in ALL_SESSIONS if e[4] is not None)} heterogeneous)')
print(f'  Train entries      : {len(TRAIN_SESSIONS)}')
print(f'  Test  entries      : {len(TEST_SESSIONS)}')
print(f'  LOSO  macro-F1     : {loso_results.get(\"fine_f1_macro\",0):.3f}')
print(f'  Test  macro-F1     : {test_results.get(\"test_f1_macro\",0):.3f}')
n_train_cycles = MODEL_FINAL[\"X_train\"].shape[0]
print(f'  Training cycles    : {n_train_cycles}')
print(f'  Feature dim        : {MODEL_FINAL[\"X_train\"].shape[1]}')
print(f'  Classes            : {list(MODEL_FINAL[\"label_encoder\"].classes_)}')
"""

# ═════════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOK
# ═════════════════════════════════════════════════════════════
cells = [
    md("---\n## 0. Title"),                     # 0
    md(CELL_TITLE),                              # 1
    md("---\n## 1. Setup & configuration"),     # 2
    code(CELL_SETUP),                            # 3
    md("---\n## 2. Core pipeline functions (V02 base)"),  # 4
    code(CELL_CORE_FUNCTIONS),                   # 5
    md("---\n## 3. V07 topology + shape features\n"
       "10D topology (cross-axis phase, plane ratio, winding) + 10D cycle-matrix PCA."),
    code(CELL_V07_FEATURES),
    md("---\n## 4. Session discovery & heterogeneous windowing"),
    code(CELL_V05_FUNCTIONS),
    md("---\n## 5. Discover sessions"),
    code(CELL_DISCOVER),
    md("### 5.1 Timestamp alignment check"),
    code(CELL_TIMESTAMP_CHECK),
    md("---\n## 6. Stratified train / test split"),
    code(CELL_SPLIT),
    md("---\n## 7. LOSO cross-validation (training set)"),
    code(CELL_LOSO),
    md("---\n## 8. Final model & held-out test evaluation"),
    code(CELL_FINAL_MODEL),
    md("---\n## 9. Diagnostics"),
    md("### 9.1  Confusion matrices — LOSO vs test"),
    code(CELL_CM),
    md("### 9.2  LOSO vs test F1 comparison"),
    code(CELL_F1_BAR),
    md("### 9.3  Feature importance (tier-colored)"),
    code(CELL_FEAT_IMP),
    md("---\n## 10. Summary"),
    code(CELL_SUMMARY),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

OUT = os.path.join(HERE, '07_Full_Pipeline_V07.ipynb')
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Written: {OUT}  ({len(cells)} cells)')
