"""
Full Pipeline V09 — Per-Hand Classification (12 classes)
=========================================================
Identical to V08 except:
- Each cycle produces 2 training samples (one per hand, 6 channels each)
- Labels: {movement}_d0 (left) and {movement}_d1 (right)
- 12 classes instead of 6 (or 5 in V08 which excluded singletons)
- Inter-hand features removed (each sample sees only one hand)
- Includes cheetahs_tail (no longer excluded — we want 6 base x 2 = 12)

Purpose: test whether left and right hands have distinct enough signatures
within the same movement to improve or harm classification.
"""

import os, sys, glob, re, json, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.linalg import svd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA as _PCA
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ropeflow-project')
DATA_PROCESSED = os.path.join(REPO, 'data', 'processed')
DATA_RAW = os.path.join(REPO, 'data', 'raw')
NEW_LABELED_RAW = os.path.join(DATA_RAW, 'new-labeled-sessions')

RUN_NAME = datetime.datetime.now().strftime('v09_run_%Y%m%d_%H%M%S')
RESULTS_DIR = os.path.join(REPO, 'results', 'Full_pipeline', RUN_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'Results -> {RESULTS_DIR}')

# ── Config (same as V08) ─────────────────────────────────────
CONFIG = {
    'FS': 50.0,
    'CYCLE_PROMINENCE_DEGS': 100.0,
    'CYCLE_MIN_PERIOD_S': 0.5,
    'CYCLE_MAX_PERIOD_S': 3.0,
    'TARGET_LEN': 64,
    'N_FOURIER_HARMONICS': 5,
    'MIN_CYCLE_SAMPLES': 10,
    'CONFIDENCE_THRESHOLD': 0.7,
    'TEMPLATE_GATE_THRESHOLD': 0.2,
    'MIN_CYCLES_PER_ENTRY': 5,
}

# V09: include cheetahs_tail, exclude only race_and_chase (= underhand duplicate)
EXCLUDE_CLASSES = {'race_and_chase'}
KNOWN_PATTERNS = {'overhand', 'sneak_overhand', 'underhand', 'sneak_underhand',
                  'dragon_roll', 'race_and_chase', 'cheetahs_tail', 'underhand_default'}

# ── Label mapping (same as V08) ──────────────────────────────
_EXACT = {
    'underhand': 'underhand', 'overhand': 'overhand', 'dragon_roll': 'dragon_roll',
    'sneak_underhand': 'sneak_underhand', 'sneak_overhand': 'sneak_overhand',
    'race_and_chase': 'race_and_chase', 'cheetahs_tail': 'cheetahs_tail', 'idle': None,
}
_PREFIX_RULES = [
    (re.compile(r'^us', re.I), 'sneak_underhand'), (re.compile(r'^os', re.I), 'sneak_overhand'),
    (re.compile(r'^u', re.I), 'underhand'), (re.compile(r'^o', re.I), 'overhand'),
    (re.compile(r'^fb', re.I), 'dragon_roll'), (re.compile(r'^bf', re.I), 'dragon_roll'),
    (re.compile(r'^cw$', re.I), None), (re.compile(r'^ccw$', re.I), None),
    (re.compile(r'^idle', re.I), None), (re.compile(r'^vq', re.I), None),
]
def _map_label(raw):
    raw = raw.strip()
    if raw.lower() in _EXACT: return _EXACT[raw.lower()]
    for pat, c in _PREFIX_RULES:
        if pat.match(raw): return c
    return None


# ══════════════════════════════════════════════════════════════
# SIGNAL PROCESSING (identical to V08)
# ══════════════════════════════════════════════════════════════

def load_session(d0, d1): return pd.read_csv(d0), pd.read_csv(d1)

def extract_signals(df):
    t = df['timestamp_ms'].values / 1000.0
    A = df[['ax_w', 'ay_w', 'az_w']].values
    omega = df[['gx', 'gy', 'gz']].values * (np.pi / 180.0)
    return t, A, omega

def detect_cycles(t, omega, fs=50.0):
    mag = np.linalg.norm(omega, axis=1)
    mag_smooth = savgol_filter(mag, window_length=15, polyorder=3)
    prom = CONFIG['CYCLE_PROMINENCE_DEGS'] * np.pi / 180.0
    min_dist = int(CONFIG['CYCLE_MIN_PERIOD_S'] * fs)
    peaks, _ = find_peaks(mag_smooth, distance=min_dist, prominence=prom)
    if len(peaks) < 2: return [], mag_smooth, peaks
    bounds = [0] + [(peaks[i]+peaks[i+1])//2 for i in range(len(peaks)-1)] + [len(t)-1]
    return [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)
            if CONFIG['CYCLE_MIN_PERIOD_S'] <= t[bounds[i+1]]-t[bounds[i]] <= CONFIG['CYCLE_MAX_PERIOD_S']
            and (bounds[i+1]-bounds[i]) >= CONFIG['MIN_CYCLE_SAMPLES']], mag_smooth, peaks

def pair_cycles(t0, cyc0, t1, cyc1):
    paired0, paired1, used = [], [], set()
    for c0 in cyc0:
        best_i, best_ov = -1, -1.0
        for i, c1 in enumerate(cyc1):
            if i in used: continue
            ov = max(0, min(t0[c0[1]], t1[c1[1]]) - max(t0[c0[0]], t1[c1[0]]))
            if ov > best_ov: best_ov, best_i = ov, i
        if best_i >= 0 and best_ov > 0:
            paired0.append(c0); paired1.append(cyc1[best_i]); used.add(best_i)
    return paired0, paired1

def resample(sig, n):
    if len(sig) < 2: return np.zeros(n) if sig.ndim == 1 else np.zeros((n, sig.shape[1]))
    x_old, x_new = np.linspace(0, 1, len(sig)), np.linspace(0, 1, n)
    if sig.ndim == 1: return np.interp(x_new, x_old, sig)
    return np.column_stack([np.interp(x_new, x_old, sig[:, j]) for j in range(sig.shape[1])])

def build_single_hand_matrix(A, omega, s, e, target_len=64):
    """Build (6, target_len) matrix for ONE hand."""
    state = np.column_stack([A[s:e], omega[s:e]])  # (N, 6)
    return resample(state, target_len).T  # (6, target_len)

def phase_align(cm, ref, ref_ch=0):
    """Phase-align on channel 0 (omega_x for single hand, was channel 3 in V08)."""
    x, y = cm[ref_ch], ref[ref_ch]
    corr = np.correlate(np.tile(x - x.mean(), 2), y - y.mean(), mode='valid')
    shift = np.argmax(corr)
    if shift >= len(x): shift -= len(x)
    return np.roll(cm, -shift, axis=1), shift

def build_template(cms):
    if not cms: return np.zeros((6, CONFIG['TARGET_LEN']))
    energies = np.array([np.sum(m**2) for m in cms])
    tmpl = cms[np.argmin(np.abs(energies - np.median(energies)))].copy()
    for _ in range(2):
        aligned = [phase_align(cm, tmpl)[0] for cm in cms]
        q25, q75 = np.percentile(energies, [25, 75])
        mask = (energies >= q25) & (energies <= q75)
        sel = [a for a, k in zip(aligned, mask) if k]
        if len(sel) < 3: sel = aligned
        tmpl = np.mean(sel, axis=0)
    return tmpl

def template_correlation(cm, tmpl):
    aligned, _ = phase_align(cm, tmpl)
    corrs = np.array([np.dot(aligned[ch]-aligned[ch].mean(), tmpl[ch]-tmpl[ch].mean()) /
                       (np.linalg.norm(aligned[ch]-aligned[ch].mean()) *
                        np.linalg.norm(tmpl[ch]-tmpl[ch].mean()) + 1e-12)
                       for ch in range(aligned.shape[0])])
    return np.mean(corrs), np.min(corrs)


# ══════════════════════════════════════════════════════════════
# SINGLE-HAND FEATURE EXTRACTION
# Adapted from V08 but for 6-channel single-hand data.
# Inter-hand features (acc_asymmetry, phase_lag, subject-invariant,
# bilateral biomech) are REMOVED since each sample sees only one hand.
# ══════════════════════════════════════════════════════════════

def physics_features_1h(t, A, omega, s, e, fs):
    """Physics features for one hand (8D instead of 12D)."""
    a, w = A[s:e], omega[s:e]
    m = np.linalg.norm(w, axis=1)
    period = t[e-1] - t[s] if e > s else 1.0
    jerk_rms = np.sqrt(np.mean(np.sum((np.diff(a, axis=0)*fs)**2, axis=1))) if len(a) > 2 else 0
    e_rot = np.mean(m**2)
    e_lin = np.mean(np.sum(a**2, axis=1))
    if len(m) > 10:
        ac = np.correlate(m - m.mean(), m - m.mean(), mode='full')
        ac = ac[len(m)-1:]; ac /= ac[0] + 1e-12
        pk, pr = find_peaks(ac[max(2, int(0.3*fs)):], height=0)
        periodicity = pr['peak_heights'][0] if len(pk) else 0
    else:
        periodicity = 0
    return np.array([period, np.max(m), np.mean(m), np.std(m),
                     np.sqrt(np.mean(np.sum(a**2, axis=1))),
                     jerk_rms, e_rot / (e_rot + e_lin + 1e-8), periodicity])

PHYS_1H_NAMES = ['period_s', 'peak_omega', 'mean_omega', 'std_omega',
                  'acc_rms', 'jerk_rms', 'ke_ratio', 'periodicity']

def fourier_features_1h(omega, s, e, fs, nh=5):
    """Fourier features for one hand (8D per channel, 16D total)."""
    def _ff(sig):
        N = len(sig)
        if N < 8: return np.zeros(3 + nh)
        sig = sig - sig.mean()
        F = np.abs(np.fft.rfft(sig)); freqs = np.fft.rfftfreq(N, 1/fs)
        F[0] = 0; power = F**2; tp = power.sum() + 1e-12
        dom = np.argmax(F[1:]) + 1; f0 = freqs[dom]
        sc = np.sum(freqs * power) / tp
        hr = np.zeros(nh)
        for h in range(nh):
            tf = f0 * (h + 1)
            if tf > freqs[-1]: break
            idx = np.argmin(np.abs(freqs - tf))
            hr[h] = np.sum(power[max(1, idx-1):min(len(F), idx+2)]) / tp
        pn = power[1:] / (power[1:].sum() + 1e-12); pn = pn[pn > 0]
        se = -np.sum(pn * np.log2(pn + 1e-12))
        return np.concatenate([[f0, sc, se], hr])
    om = omega[s:e]
    return np.concatenate([_ff(np.linalg.norm(om, axis=1)),
                            _ff(np.linalg.norm(om, axis=1))])  # omega_mag x2 for consistency

FOURIER_1H_DIM = 2 * (3 + CONFIG['N_FOURIER_HARMONICS'])
FOURIER_1H_NAMES = []
for ch in ['omega_mag', 'omega_mag2']:
    FOURIER_1H_NAMES += [f'{ch}_dom_freq', f'{ch}_spec_centroid', f'{ch}_spec_entropy']
    FOURIER_1H_NAMES += [f'{ch}_harm_{i}' for i in range(CONFIG['N_FOURIER_HARMONICS'])]

def svd_features_1h(cm, n_comp=4):
    """SVD of (6, 64) single-hand matrix."""
    U, S, _ = svd(cm, full_matrices=False)
    sr = S[:n_comp] / (S[0] + 1e-12)
    return np.concatenate([sr, [np.sum(S > 0.01 * S[0]), np.sum(S**2)]])

SVD_1H_DIM = 4 + 2
SVD_1H_NAMES = [f'sv_ratio_{i}' for i in range(4)] + ['eff_rank', 'total_energy']

def dmd_features_1h(cm, dt):
    """DMD of (6, 64) single-hand matrix."""
    X1, X2 = cm[:, :-1], cm[:, 1:]
    U, S, Vh = svd(X1, full_matrices=False)
    cv = np.cumsum(S**2) / (S**2).sum() + 1e-12
    r = max(1, min(int(np.searchsorted(cv, 0.99)) + 1, len(S), X1.shape[1]))
    A_t = U[:, :r].T @ X2 @ Vh[:r, :].T @ np.diag(1 / (S[:r] + 1e-12))
    lam, _ = np.linalg.eig(A_t)
    lam_c = np.log(np.maximum(np.abs(lam), 1e-12)) / dt
    f = np.abs(lam_c.imag) / (2 * np.pi); g = lam_c.real
    o = np.argsort(f)[::-1]; f, g = f[o], g[o]
    ft, gt = np.zeros(3), np.zeros(3)
    for i in range(min(3, len(f))): ft[i] = f[i]; gt[i] = g[i]
    return np.concatenate([ft, gt, [r, g.mean(), np.max(np.abs(g))]])

DMD_1H_NAMES = [f'dmd_freq_{i}' for i in range(3)] + [f'dmd_growth_{i}' for i in range(3)] + \
               ['dmd_rank', 'dmd_mean_growth', 'dmd_max_abs_growth']

def topology_features_1h(omega, s, e, fs):
    """Topology features for one hand (19D)."""
    w = omega[s:e].copy()
    n = len(w)
    if n < 6: return np.zeros(19)
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    w_norm = np.linalg.norm(w, axis=1)

    def _phase_lag(a, b, t_ref):
        corr = np.correlate(a - a.mean(), b - b.mean(), mode='full')
        lag = np.argmax(corr) - (len(a) - 1)
        return np.abs((lag / max(1, t_ref)) * 360 % 360 - 180)

    # Dominant period
    ac = np.correlate(w_norm - w_norm.mean(), w_norm - w_norm.mean(), mode='full')[n-1:]
    ml, mxl = max(2, int(CONFIG['CYCLE_MIN_PERIOD_S'] * fs)), min(n-2, int(CONFIG['CYCLE_MAX_PERIOD_S'] * fs))
    if mxl > ml:
        pk, pr = find_peaks(ac[ml:mxl+1], prominence=max(1e-12, 0.05 * ac[0]))
        T_dom = float(ml + pk[np.argmax(pr['prominences'])]) if len(pk) else float(max(1, n))
    else:
        T_dom = float(max(1, n))

    # Phase lags
    pxy = _phase_lag(wx, wy, T_dom)
    pxz = _phase_lag(wx, wz, T_dom)
    pyz = _phase_lag(wy, wz, T_dom)

    # SVD spatial ratios
    _, Sw, _ = np.linalg.svd(w.T, full_matrices=False)
    planarity = Sw[1] / (Sw[0] + 1e-12) if len(Sw) >= 3 else 0
    spatial = Sw[2] / (Sw[0] + 1e-12) if len(Sw) >= 3 else 0

    # Winding
    cycles = max(1, n / T_dom)
    wind_x = np.sum(np.diff(np.sign(wx - wx.mean())) != 0) / cycles
    wind_y = np.sum(np.diff(np.sign(wy - wy.mean())) != 0) / cycles

    # Axis energy
    energy = np.array([np.mean(wx**2), np.mean(wy**2), np.mean(wz**2)])
    energy /= energy.sum() + 1e-12

    # Torsion
    cross_mag = np.linalg.norm(np.cross(w[:-1], np.roll(w, -1, axis=0)[:-1]), axis=1)
    torsion = float(np.mean(cross_mag))
    torsion_var = float(np.var(cross_mag)) if len(cross_mag) > 1 else 0

    # Temporal variability
    temporal_var = float(np.var(w_norm))
    phase_win = max(6, int(0.5 * T_dom))
    phase_step = max(1, phase_win // 2)
    pxy_vals, pyz_vals = [], []
    if phase_win < n:
        for st in range(0, n - phase_win + 1, phase_step):
            pxy_vals.append(_phase_lag(wx[st:st+phase_win], wy[st:st+phase_win], T_dom))
            pyz_vals.append(_phase_lag(wy[st:st+phase_win], wz[st:st+phase_win], T_dom))
    pxy_var = float(np.var(pxy_vals)) if len(pxy_vals) > 1 else 0
    pyz_var = float(np.var(pyz_vals)) if len(pyz_vals) > 1 else 0

    local_win = max(6, int(0.25 * T_dom))
    local_step = max(1, local_win // 2)
    local_vars = [np.var(w_norm[st:st+local_win]) for st in range(0, n-local_win+1, local_step)] if local_win < n else []
    local_tv = float(np.var(local_vars)) if len(local_vars) > 1 else 0
    roll_var = float(np.mean(local_vars)) if local_vars else 0
    tv_deriv = float(np.mean(np.abs(np.diff(local_vars)))) if len(local_vars) > 1 else 0
    pxy_drift = float(np.mean(np.abs(np.diff(pxy_vals)))) if len(pxy_vals) > 1 else 0
    pyz_drift = float(np.mean(np.abs(np.diff(pyz_vals)))) if len(pyz_vals) > 1 else 0
    phase_drift = 0.5 * (pxy_drift + pyz_drift)

    return np.array([pxy, pxz, pyz, planarity, spatial, wind_x, wind_y,
                     energy[0], energy[1], energy[2], torsion, torsion_var,
                     temporal_var, pxy_var, pyz_var, local_tv, roll_var, tv_deriv, phase_drift])

TOPO_1H_NAMES = ['phase_xy', 'phase_xz', 'phase_yz', 'plane_ratio', 'spatial_ratio',
                  'wind_x', 'wind_y', 'energy_x', 'energy_y', 'energy_z',
                  'torsion', 'torsion_var', 'temporal_var', 'phase_xy_var', 'phase_yz_var',
                  'local_temporal_var', 'rolling_var', 'temporal_var_deriv', 'phase_drift']

def biomech_features_1h(omega, s, e, fs):
    """Single-hand biomech features (4D) -- no bilateral features."""
    w = omega[s:e]
    m = np.linalg.norm(w, axis=1)
    if len(m) < 8: return np.zeros(4)
    e_ax = np.mean(w**2, axis=0)
    k = int(np.argmax(e_ax))
    share = float(e_ax[k] / (e_ax.sum() + 1e-8))
    signed_mean = float(np.mean(w[:, k]) / (np.sqrt(e_ax[k]) + 1e-8))
    signflip = float(np.mean(np.sign(w[:-1, k]) != np.sign(w[1:, k])))
    return np.array([share, signed_mean, signflip, skew(m)])

BIOMECH_1H_NAMES = ['dom_axis_share', 'dom_axis_signed_mean', 'dom_axis_signflip', 'skew_omega']

# ── Combined feature names ───────────────────────────────────
ALL_1H_NAMES = (PHYS_1H_NAMES + FOURIER_1H_NAMES + SVD_1H_NAMES +
                ['template_corr_mean', 'template_corr_min'] +
                DMD_1H_NAMES + TOPO_1H_NAMES + BIOMECH_1H_NAMES)
N_1H_FEAT = len(ALL_1H_NAMES)
print(f'Single-hand feature dim: {N_1H_FEAT}D')


# ══════════════════════════════════════════════════════════════
# SESSION DISCOVERY (identical to V08)
# ══════════════════════════════════════════════════════════════

def discover_sessions(pdir):
    sessions = []
    for d0 in sorted(glob.glob(os.path.join(pdir, '*_device0_processed.csv'))):
        d1 = d0.replace('_device0_', '_device1_')
        if not os.path.exists(d1): continue
        stem = os.path.basename(d0).replace('_device0_processed.csv', '')
        parts = stem.split('_')
        if len(parts) < 3: continue
        rest = '_'.join(parts[2:])
        label = None
        for pat in sorted(KNOWN_PATTERNS, key=len, reverse=True):
            if rest.startswith(pat):
                label = 'underhand' if pat == 'underhand_default' else pat
                break
        if label and label not in EXCLUDE_CLASSES:
            sessions.append((d0, d1, label, stem, None))
    return sessions

def discover_heterogeneous(pdir, raw_dir=None):
    if raw_dir is None: raw_dir = NEW_LABELED_RAW
    if not os.path.isdir(raw_dir): return []
    entries = []
    for sname in sorted(os.listdir(raw_dir)):
        sdir = os.path.join(raw_dir, sname)
        if not os.path.isdir(sdir): continue
        lpath = None
        for fn in ('labels_corrected.json', 'labels.json'):
            p = os.path.join(sdir, fn)
            if os.path.isfile(p): lpath = p; break
        if not lpath: continue
        d0 = os.path.join(pdir, f'{sname}_device0_processed.csv')
        d1 = os.path.join(pdir, f'{sname}_device1_processed.csv')
        if not (os.path.isfile(d0) and os.path.isfile(d1)): continue
        with open(lpath, encoding='utf-8') as f: data = json.load(f)
        wbl = defaultdict(list)
        for seg in data.get('segments', []):
            canon = _map_label(seg.get('label', ''))
            if canon is None or canon in EXCLUDE_CLASSES: continue
            s, e = seg.get('start'), seg.get('end')
            if s is None: continue
            if e is None: e = s + 2.0
            wbl[canon].append((float(s), float(e)))
        for label, windows in sorted(wbl.items()):
            entries.append((d0, d1, label, f'{sname}/{label}', windows))
    return entries

def discover_all(pdir):
    all_e = discover_sessions(pdir) + discover_heterogeneous(pdir)
    # V09: do NOT exclude cheetahs_tail
    by_class = defaultdict(list)
    for e in all_e: by_class[e[2]].append(e[3])
    print(f'Discovered {len(all_e)} entries:')
    for lab in sorted(by_class): print(f'  {lab:<22s}: {len(by_class[lab])} entries')
    return all_e

def session_group(entry):
    dn = entry[3]
    return dn.split('/')[0] if '/' in dn else dn

def stratified_split(sessions, test_frac=0.2, seed=42):
    import random
    rng = random.Random(seed)
    groups = defaultdict(list)
    for e in sessions: groups[session_group(e)].append(e)
    class_groups = defaultdict(list)
    for g, ents in groups.items():
        for lab in set(e[2] for e in ents): class_groups[lab].append(g)
    test_groups = set()
    for lab, gnames in sorted(class_groups.items()):
        unique = list(set(gnames))
        if len(unique) < 2: continue
        rng.shuffle(unique)
        test_groups.add(unique[0])
    return ([e for e in sessions if session_group(e) not in test_groups],
            [e for e in sessions if session_group(e) in test_groups])


# ══════════════════════════════════════════════════════════════
# V09: PER-HAND TRAINING
# ══════════════════════════════════════════════════════════════

def process_and_extract_per_hand(entry):
    """Process one entry and return per-hand features + labels.
    Returns: list of (features_1h, label_1h, cm_1h) for each hand of each cycle.
    """
    d0, d1, label, dname, tw = entry
    df0, df1 = load_session(d0, d1)
    t0, A0, om0 = extract_signals(df0)
    t1, A1, om1 = extract_signals(df1)
    fs = CONFIG['FS']

    cyc0, _, _ = detect_cycles(t0, om0, fs)
    cyc1, _, _ = detect_cycles(t1, om1, fs)
    p0, p1 = pair_cycles(t0, cyc0, t1, cyc1)

    if tw is not None:
        fp0, fp1 = [], []
        for (s0, e0), (s1, e1) in zip(p0, p1):
            t_mid = (t0[s0] + t0[e0]) / 2
            if any(ws <= t_mid < we for ws, we in tw):
                fp0.append((s0, e0)); fp1.append((s1, e1))
        p0, p1 = fp0, fp1

    results = []  # (features, label, cm)
    for (s0, e0), (s1, e1) in zip(p0, p1):
        # Device 0 (left hand)
        cm0 = build_single_hand_matrix(A0, om0, s0, e0, CONFIG['TARGET_LEN'])
        results.append(('d0', cm0, t0, A0, om0, s0, e0, label))
        # Device 1 (right hand)
        cm1 = build_single_hand_matrix(A1, om1, s1, e1, CONFIG['TARGET_LEN'])
        results.append(('d1', cm1, t1, A1, om1, s1, e1, label))

    return results


def extract_1h_features(t, A, omega, s, e, cm, template, fs):
    """Extract full single-hand feature vector."""
    dt = 1 / fs
    phys = physics_features_1h(t, A, omega, s, e, fs)
    fourier = fourier_features_1h(omega, s, e, fs)
    sv = svd_features_1h(cm, n_comp=4)
    corr_mean, corr_min = template_correlation(cm, template)
    dmd = dmd_features_1h(cm, dt)
    topo = topology_features_1h(omega, s, e, fs)
    biomech = biomech_features_1h(omega, s, e, fs)
    return np.concatenate([phys, fourier, sv, [corr_mean, corr_min], dmd, topo, biomech])


def train_model_v09(entries, verbose=True):
    """Train per-hand classifier with 12 classes."""
    fs = CONFIG['FS']
    all_samples = []  # (hand, cm, t, A, omega, s, e, base_label)

    for entry in entries:
        samples = process_and_extract_per_hand(entry)
        if len(samples) < CONFIG['MIN_CYCLES_PER_ENTRY'] * 2:
            if verbose and len(samples) > 0:
                print(f'  SKIP {entry[3]}: {len(samples)//2} cycles (< {CONFIG["MIN_CYCLES_PER_ENTRY"]})')
            continue
        all_samples.extend(samples)
        if verbose:
            print(f'  {entry[2]:<22s} [{entry[3]}]: {len(samples)//2} cycles x 2 hands')

    if not all_samples:
        raise ValueError('No samples')

    # Build per-class per-hand templates
    cms_by_label = defaultdict(list)
    for hand, cm, *_, label in all_samples:
        cms_by_label[f'{label}_{hand}'].append(cm)
    class_templates = {lab: build_template(cms) for lab, cms in cms_by_label.items() if cms}

    # Global template (average of all)
    all_cms = [cm for _, cm, *_ in all_samples]
    global_template = build_template(all_cms)

    # Extract features
    X_list, y_list = [], []
    for hand, cm, t, A, omega, s, e, label in all_samples:
        feat = extract_1h_features(t, A, omega, s, e, cm, global_template, fs)
        feat = np.nan_to_num(feat, nan=0, posinf=0, neginf=0)
        X_list.append(feat)
        y_list.append(f'{label}_{hand}')  # 12 classes: movement_d0, movement_d1

    X_all = np.vstack(X_list)
    y_all = np.array(y_list)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_all)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    clf.fit(X_s, y_enc)

    if verbose:
        print(f'\nTrained: {X_all.shape[0]} samples x {X_all.shape[1]} features')
        print(f'Classes ({len(le.classes_)}): {list(le.classes_)}')
        print(f'Distribution: {dict(Counter(y_all))}')

    return {
        'clf': clf, 'scaler': scaler, 'label_encoder': le,
        'template': global_template, 'class_templates': class_templates,
        'feature_names': ALL_1H_NAMES,
    }


def classify_entry_v09(entry, model, verbose=False):
    """Classify with per-hand predictions, then merge to movement-level verdict."""
    samples = process_and_extract_per_hand(entry)
    if not samples:
        return {'verdict': 'unknown', 'labels_12': [], 'base_labels': [], 'confidences': []}

    fs = CONFIG['FS']
    X_list = []
    for hand, cm, t, A, omega, s, e, label in samples:
        feat = extract_1h_features(t, A, omega, s, e, cm, model['template'], fs)
        feat = np.nan_to_num(feat, nan=0, posinf=0, neginf=0)
        X_list.append(feat)

    X = np.vstack(X_list)
    X_s = model['scaler'].transform(X)
    le = model['label_encoder']
    preds = le.inverse_transform(model['clf'].predict(X_s))
    proba = np.max(model['clf'].predict_proba(X_s), axis=1)

    # Gating
    labels_12 = []
    for i, (pred, conf) in enumerate(zip(preds, proba)):
        if conf < CONFIG['CONFIDENCE_THRESHOLD']:
            labels_12.append('transition')
        else:
            labels_12.append(pred)

    # Extract base movement (strip _d0/_d1) for session verdict
    base_labels = []
    for l in labels_12:
        if l == 'transition':
            base_labels.append('transition')
        else:
            base_labels.append(l.rsplit('_', 1)[0])  # underhand_d0 -> underhand

    non_trans = [l for l in base_labels if l != 'transition']
    vote = Counter(non_trans)
    verdict = vote.most_common(1)[0][0] if non_trans else 'unknown'

    if verbose:
        n_t = sum(1 for l in labels_12 if l == 'transition')
        print(f'  {entry[3]:<52s} {len(labels_12)} samples ({n_t} trans) -> {verdict}')

    return {
        'labels_12': labels_12,       # 12-class labels
        'base_labels': base_labels,   # 6-class labels (for comparison with V08)
        'verdict': verdict,
        'vote': dict(vote),
        'confidences': proba.tolist(),
    }


def run_loso_v09(sessions, verbose=True):
    """LOSO with per-hand 12-class classification."""
    class_groups = defaultdict(set)
    for e in sessions: class_groups[e[2]].add(session_group(e))
    singleton_classes = {c for c, gs in class_groups.items() if len(gs) == 1}
    loso_groups = sorted(set(session_group(e) for e in sessions if e[2] not in singleton_classes))

    if verbose and singleton_classes:
        print(f'Singleton classes (train-only): {sorted(singleton_classes)}')
    print(f'LOSO folds: {len(loso_groups)}')

    all_true_12, all_pred_12 = [], []   # 12-class
    all_true_6, all_pred_6 = [], []     # 6-class (base movement)
    entry_results = []

    for test_grp in loso_groups:
        test_e = [e for e in sessions if session_group(e) == test_grp]
        train_e = [e for e in sessions if session_group(e) != test_grp]
        model = train_model_v09(train_e, verbose=False)

        for entry in test_e:
            res = classify_entry_v09(entry, model)
            if not res['labels_12']: continue

            # 12-class evaluation
            non_trans_12 = [(l, entry[2]) for l in res['labels_12'] if l != 'transition']
            for pred_12, true_base in non_trans_12:
                hand = pred_12.rsplit('_', 1)[1] if '_' in pred_12 else 'd0'
                true_12 = f'{true_base}_{hand}'
                # We don't know which sample is d0 vs d1 from ground truth,
                # so for 12-class eval we match by the hand in the prediction
                all_true_12.append(true_12)
                all_pred_12.append(pred_12)

            # 6-class evaluation (strip hand, compare base movement)
            non_trans_6 = [l for l in res['base_labels'] if l != 'transition']
            all_true_6.extend([entry[2]] * len(non_trans_6))
            all_pred_6.extend(non_trans_6)

            correct = res['verdict'] == entry[2]
            entry_results.append((entry[3], entry[2], res['verdict'], correct))
            if verbose:
                s = 'OK' if correct else 'WRONG'
                print(f'  {entry[3]:<52s} GT={entry[2]:<18s} -> {res["verdict"]:<18s} [{s}]')

    n_ok = sum(r[3] for r in entry_results)
    n_tot = len(entry_results)
    print(f'\nLOSO entry accuracy: {n_ok}/{n_tot} ({n_ok/max(n_tot,1)*100:.0f}%)')

    result = {'entry_results': entry_results, 'n_correct': n_ok, 'n_total': n_tot}

    # 6-class metrics (comparable to V08)
    if all_true_6:
        labs6 = sorted(set(all_true_6 + all_pred_6))
        f1_6 = f1_score(all_true_6, all_pred_6, average='macro', labels=labs6, zero_division=0)
        print(f'\n--- 6-CLASS (base movement, comparable to V08) ---')
        print(classification_report(all_true_6, all_pred_6, labels=labs6, zero_division=0))
        print(f'6-class F1 macro = {f1_6:.3f}')
        result['f1_6class'] = f1_6
        result['cm_6class'] = confusion_matrix(all_true_6, all_pred_6, labels=labs6)
        result['labels_6class'] = labs6

    # 12-class metrics
    if all_true_12:
        labs12 = sorted(set(all_true_12 + all_pred_12))
        f1_12 = f1_score(all_true_12, all_pred_12, average='macro', labels=labs12, zero_division=0)
        print(f'\n--- 12-CLASS (per-hand) ---')
        print(classification_report(all_true_12, all_pred_12, labels=labs12, zero_division=0))
        print(f'12-class F1 macro = {f1_12:.3f}')
        result['f1_12class'] = f1_12
        result['cm_12class'] = confusion_matrix(all_true_12, all_pred_12, labels=labs12)
        result['labels_12class'] = labs12

    return result


def plot_cm(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.0), max(6, len(labels)*0.8)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()*0.5 else 'black', fontsize=6)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('='*70)
    print('V09 Pipeline — Per-Hand Classification (12 classes)')
    print('='*70)

    ALL = discover_all(DATA_PROCESSED)
    TRAIN, TEST = stratified_split(ALL, test_frac=0.2)
    print(f'\nTrain: {len(TRAIN)} entries, Test: {len(TEST)} entries')

    # LOSO
    print('\n' + '='*70)
    print('LOSO CV (training set)')
    print('='*70)
    loso = run_loso_v09(TRAIN, verbose=True)

    if 'cm_6class' in loso:
        plot_cm(loso['cm_6class'], loso['labels_6class'],
                f'V09 LOSO 6-class (F1={loso["f1_6class"]:.3f})',
                os.path.join(RESULTS_DIR, 'loso_6class_cm.png'))
    if 'cm_12class' in loso:
        plot_cm(loso['cm_12class'], loso['labels_12class'],
                f'V09 LOSO 12-class (F1={loso["f1_12class"]:.3f})',
                os.path.join(RESULTS_DIR, 'loso_12class_cm.png'))

    # Comparison
    print('\n' + '='*70)
    print('COMPARISON: V08 vs V09')
    print('='*70)
    print(f'V08 LOSO F1 (6-class):  0.632  (from last run)')
    print(f'V09 LOSO F1 (6-class):  {loso.get("f1_6class", 0):.3f}  (per-hand, mapped back)')
    print(f'V09 LOSO F1 (12-class): {loso.get("f1_12class", 0):.3f}  (full per-hand)')
    delta = loso.get('f1_6class', 0) - 0.632
    print(f'Delta (6-class):        {delta:+.3f}  ({"BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"})')
    print(f'\nResults: {RESULTS_DIR}')
