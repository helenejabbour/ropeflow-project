"""Create 10_Full_Pipeline_V10.2.ipynb from V10.1 with a clean section 10."""
import json, copy

SRC = r'C:/Users/Admin/Projects/ropeflow-project/src/Full pipeline/10_Full_Pipeline_V10.1.ipynb'
DST = r'C:/Users/Admin/Projects/ropeflow-project/src/Full pipeline/10_Full_Pipeline_V10.2.ipynb'

NEW_CELL = r"""import datetime as _dt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import silhouette_score as _sil

# ── Config ────────────────────────────────────────────────────────────────
DIRECT_CFG = {
    'FS':                 CONFIG['FS'],
    'WINDOW':             64,
    'PEAK_PROM_DEGS':     CONFIG['CYCLE_PROMINENCE_DEGS'],
    'PEAK_MIN_DEGS':      50.0,
    'PEAK_SAVGOL_WINDOW': CONFIG['CYCLE_SAVGOL_WINDOW'],
    'PEAK_SAVGOL_POLY':   CONFIG['CYCLE_SAVGOL_POLYORDER'],
    'PEAK_MIN_PERIOD_S':  CONFIG['CYCLE_MIN_PERIOD_S'],
    'PEAK_PAIR_MAX_DT_S': 0.25,
    'N_CLUSTERS':         12,
}

RUN_NAME    = _dt.datetime.now().strftime('run_%Y%m%d_%H%M%S')
RESULTS_DIR = os.path.join('..', '..', 'results', 'Full_pipeline', RUN_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'V10.2 results -> {RESULTS_DIR}')

# ── Session discovery ─────────────────────────────────────────────────────
def discover_direct_sessions(processed_dir):
    entries = []
    for d0 in sorted(glob.glob(os.path.join(processed_dir, '*_device0_processed.csv'))):
        d1 = d0.replace('_device0_', '_device1_')
        if not os.path.isfile(d1):
            continue
        stem = os.path.basename(d0).replace('_device0_processed.csv', '')
        entries.append((d0, d1, 'unlabeled', stem, None))
    return entries

# ── Cycle-extraction helpers ──────────────────────────────────────────────
def _smooth_mag_deg(omega_rad, cfg):
    mag_deg = np.linalg.norm(omega_rad, axis=1) * (180.0 / np.pi)
    n = len(mag_deg)
    if n < 7: return mag_deg
    win  = int(cfg['PEAK_SAVGOL_WINDOW'])
    if win % 2 == 0: win += 1
    max_odd = n if n % 2 == 1 else n - 1
    win  = max(5, min(win, max_odd))
    poly = max(1, min(int(cfg['PEAK_SAVGOL_POLY']), win - 2))
    y = savgol_filter(mag_deg, window_length=win, polyorder=poly, mode='interp')
    y = savgol_filter(y,       window_length=win, polyorder=poly, mode='interp')
    return y

def detect_cycle_peaks(omega_rad, fs, cfg):
    mag_smooth = _smooth_mag_deg(omega_rad, cfg)
    peaks, _   = find_peaks(
        mag_smooth,
        distance=max(1, int(cfg['PEAK_MIN_PERIOD_S'] * fs)),
        prominence=cfg['PEAK_PROM_DEGS'],
    )
    peaks = np.array([int(p) for p in peaks if mag_smooth[p] >= cfg['PEAK_MIN_DEGS']], dtype=int)
    return peaks, mag_smooth

def pair_peaks_same_swing(t0, peaks0, t1, peaks1, max_dt_s):
    if len(peaks0) == 0 or len(peaks1) == 0: return []
    used, pairs = set(), []
    t1_peaks    = t1[peaks1]
    for p0 in peaks0:
        d = np.abs(t1_peaks - t0[p0])
        for idx in np.argsort(d):
            p1 = int(peaks1[idx])
            if p1 in used: continue
            if d[idx] <= max_dt_s:
                used.add(p1); pairs.append((int(p0), p1)); break
    return pairs

def extract_fixed_window(ch6, center_idx, window=64):
    half  = window // 2
    start = int(center_idx) - half
    end   = start + window
    out   = np.zeros((6, window), dtype=np.float32)
    src_lo, src_hi = max(0, start), min(ch6.shape[0], end)
    if src_hi <= src_lo: return out
    dst_lo = src_lo - start
    out[:, dst_lo:dst_lo + (src_hi - src_lo)] = ch6[src_lo:src_hi].T
    return out

def extract_cycles_from_entry(entry, cfg):
    d0_path, d1_path, _, dname, windows = entry
    d0, d1 = load_session(d0_path, d1_path)
    t0, _q0, A0, om0 = extract_signals(d0)
    t1, _q1, A1, om1 = extract_signals(d1)
    peaks0, mag0 = detect_cycle_peaks(om0, cfg['FS'], cfg)
    peaks1, mag1 = detect_cycle_peaks(om1, cfg['FS'], cfg)
    pairs = pair_peaks_same_swing(t0, peaks0, t1, peaks1, cfg['PEAK_PAIR_MAX_DT_S'])
    ch0   = np.column_stack([A0, om0 * (180.0 / np.pi)])
    ch1   = np.column_stack([A1, om1 * (180.0 / np.pi)])
    mats  = []
    for p0, p1 in pairs:
        t_mid = 0.5 * (t0[p0] + t1[p1])
        if windows is not None and not any(ws <= t_mid < we for ws, we in windows):
            continue
        w0 = extract_fixed_window(ch0, p0, cfg['WINDOW'])
        w1 = extract_fixed_window(ch1, p1, cfg['WINDOW'])
        mats.append(np.vstack([w0, w1]).astype(np.float32))
    return mats

def build_cycle_dataset(entries, cfg):
    X_list, g_list, sid_list = [], [], []
    n_with = 0
    for e in entries:
        mats = extract_cycles_from_entry(e, cfg)
        if mats: n_with += 1
        grp = session_group(e)
        for m in mats:
            X_list.append(m)
            g_list.append(grp)
            sid_list.append(e[3])
    if not X_list:
        raise RuntimeError('No cycles extracted.')
    X   = np.stack(X_list).astype(np.float32)
    g   = np.array(g_list)
    sid = np.array(sid_list)
    print(f'Sessions: {len(entries)} | with cycles: {n_with}')
    print(f'Dataset:  X={X.shape} | groups={len(np.unique(g))}')
    return X, g, sid

# ── Feature matrix: flattened 12x64 = 768-D ──────────────────────────────
def build_feature_matrix(X_cycles):
    F = X_cycles.reshape(len(X_cycles), -1).astype(np.float32)
    print(f'Feature matrix: {F.shape}')
    return F

# ── K-Means clustering ────────────────────────────────────────────────────
def fit_cluster_pipeline(X_feat, n_clusters=12):
    sc     = StandardScaler()
    X_s    = sc.fit_transform(X_feat)
    n_comp = min(50, X_s.shape[1], X_s.shape[0] - 1)
    pca    = PCA(n_components=n_comp, svd_solver='full')
    X_p    = pca.fit_transform(X_s)
    km     = KMeans(n_clusters=n_clusters, init='k-means++',
                    n_init=20, max_iter=500, random_state=42)
    labels = km.fit_predict(X_p)
    sil    = _sil(X_p, labels, sample_size=min(2000, len(labels)))
    print(f'K-Means(k={n_clusters}) silhouette={sil:.3f}')
    return labels, km, sc, pca

# ── Cluster visualisation & analysis ─────────────────────────────────────
def save_cluster_composition(cluster_labels, session_ids, n_clusters, results_dir):
    df    = pd.DataFrame({'cluster': cluster_labels, 'session': session_ids})
    pivot = df.groupby(['cluster', 'session']).size().unstack(fill_value=0)
    pivot['dominant'] = pivot.idxmax(axis=1)
    pivot.to_csv(os.path.join(results_dir, 'cluster_composition.csv'))
    print('\nCluster composition saved to cluster_composition.csv')
    print(pivot[['dominant']].to_string())
    return pivot

def plot_cluster_prototypes(X_cycles, cluster_labels, n_clusters, results_dir):
    fig, axes = plt.subplots(n_clusters, 2, figsize=(12, n_clusters * 1.8), squeeze=False)
    for c in range(n_clusters):
        mask = cluster_labels == c
        if not mask.any():
            for ax in axes[c]: ax.axis('off')
            continue
        W   = X_cycles[mask]
        gm0 = np.linalg.norm(W[:, 3:6, :], axis=1)
        gm1 = np.linalg.norm(W[:, 9:12, :], axis=1)
        axes[c][0].plot(gm0.T,       color='steelblue', alpha=0.12, lw=0.5)
        axes[c][0].plot(gm0.mean(0), color='navy',      lw=2)
        axes[c][0].set_title(f'C{c:02d} dev0 |omega| (n={mask.sum()})', fontsize=8)
        axes[c][1].plot(gm1.T,       color='tomato',    alpha=0.12, lw=0.5)
        axes[c][1].plot(gm1.mean(0), color='darkred',   lw=2)
        axes[c][1].set_title(f'C{c:02d} dev1 |omega|', fontsize=8)
        for ax in axes[c]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Cluster prototypes -- gyro magnitude per device', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'fig_cluster_prototypes.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

def plot_pca_var(X_feat, results_dir):
    sc     = StandardScaler()
    X_s    = sc.fit_transform(X_feat)
    nmax   = min(X_s.shape[0], X_s.shape[1])
    pv     = PCA(n_components=nmax, svd_solver='full')
    pv.fit(X_s)
    cumvar = np.cumsum(pv.explained_variance_ratio_)
    n95    = int(np.searchsorted(cumvar, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, lw=2)
    ax.axhline(0.95, color='r', ls='--', lw=1)
    ax.axvline(n95,  color='r', ls='--', lw=1, label=f'n95={n95}')
    ax.set_xlabel('Components'); ax.set_ylabel('Cumulative explained variance')
    ax.set_title('PCA explained variance (768-D flat)'); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'fig_pca_var.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'PCA: {n95} components for 95% variance.')

def visualize_clusters_umap(X_feat, cluster_labels, session_ids, results_dir):
    try:
        import umap.umap_ as umap
        sc  = StandardScaler()
        emb = umap.UMAP(n_components=2, random_state=42).fit_transform(sc.fit_transform(X_feat))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc0 = axes[0].scatter(emb[:, 0], emb[:, 1], c=cluster_labels,
                               cmap='tab20', s=8, alpha=0.7)
        plt.colorbar(sc0, ax=axes[0])
        axes[0].set_title('UMAP -- cluster IDs'); axes[0].axis('off')
        uniq  = sorted(np.unique(session_ids))
        cmap2 = plt.cm.get_cmap('tab20', len(uniq))
        li    = {l: i for i, l in enumerate(uniq)}
        axes[1].scatter(emb[:, 0], emb[:, 1],
                        c=[cmap2(li[s]) for s in session_ids], s=8, alpha=0.7)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap2(i), markersize=5, label=l)
                   for i, l in enumerate(uniq)]
        axes[1].legend(handles=handles, fontsize=7, ncol=2)
        axes[1].set_title('UMAP -- recording IDs'); axes[1].axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, 'fig_umap_clusters.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('UMAP scatter saved.')
    except ImportError:
        print('umap-learn not installed -- run: pip install umap-learn')

# ── LOSO evaluation helper ────────────────────────────────────────────────
def _save_cluster_eval(y_true, y_pred, n_clusters, tag, results_dir):
    cnames   = [f'C{i:02d}' for i in range(n_clusters)]
    cm       = confusion_matrix(y_true, y_pred, labels=np.arange(n_clusters))
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_f1   = f1_score(y_true, y_pred, average=None,
                        labels=np.arange(n_clusters), zero_division=0)
    pd.DataFrame({'cluster': cnames, 'f1': per_f1,
                  'support': cm.sum(axis=1)}).to_csv(
        os.path.join(results_dir, f'{tag}_cluster_f1.csv'), index=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=cnames).plot(
        ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
    ax.set_title(f'{tag} | macro-F1={macro_f1:.3f}')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f'fig_confusion_{tag}.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'[{tag}] macro-F1={macro_f1:.3f}  acc={np.mean(y_true == y_pred):.3f}')
    return macro_f1

# ── LOSO: PCA + GBM ──────────────────────────────────────────────────────
def run_loso_pca_gbm(X_feat, y_clusters, groups, n_clusters, results_dir):
    uniq = np.unique(groups)
    y_true_all, y_pred_all = [], []
    for fi, g in enumerate(uniq, 1):
        tr, te   = groups != g, groups == g
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y_clusters[tr], y_clusters[te]
        if not len(yte): continue
        sc  = StandardScaler()
        Xts = sc.fit_transform(Xtr)
        Xes = sc.transform(Xte)
        pca = PCA(n_components=0.95, svd_solver='full')
        Xtp = pca.fit_transform(Xts)
        Xep = pca.transform(Xes)
        if len(np.unique(ytr)) < 2:
            y_true_all.extend(yte.tolist())
            y_pred_all.extend([int(ytr[0])] * len(yte))
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                              learning_rate=0.08, random_state=42)
        clf.fit(Xtp, ytr)
        pred = clf.predict(Xep)
        y_true_all.extend(yte.tolist())
        y_pred_all.extend(pred.tolist())
        print(f'  [PCA+GBM] fold {fi}/{len(uniq)} | {g} | pca_k={pca.n_components_} | acc={np.mean(pred == yte):.3f}')
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    return _save_cluster_eval(y_true, y_pred, n_clusters, 'pca_gbm', results_dir)

# ── LOSO: Random Forest ───────────────────────────────────────────────────
def run_loso_rf(X_feat, y_clusters, groups, n_clusters, results_dir):
    uniq = np.unique(groups)
    y_true_all, y_pred_all = [], []
    for fi, g in enumerate(uniq, 1):
        tr, te   = groups != g, groups == g
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y_clusters[tr], y_clusters[te]
        if not len(yte): continue
        sc       = StandardScaler()
        Xts      = sc.fit_transform(Xtr)
        Xes      = sc.transform(Xte)
        present  = np.unique(ytr)
        cw       = compute_class_weight('balanced', classes=present, y=ytr)
        cw_dict  = {int(c): float(w) for c, w in zip(present, cw)}
        clf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                     min_samples_leaf=1, class_weight=cw_dict,
                                     n_jobs=-1, random_state=42)
        clf.fit(Xts, ytr)
        pred = clf.predict(Xes)
        y_true_all.extend(yte.tolist())
        y_pred_all.extend(pred.tolist())
        print(f'  [RF] fold {fi}/{len(uniq)} | {g} | acc={np.mean(pred == yte):.3f}')
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    return _save_cluster_eval(y_true, y_pred, n_clusters, 'rf', results_dir)

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
SESSIONS = discover_direct_sessions(DATA_PROCESSED)
print(f'Sessions found: {len(SESSIONS)}')

X_cycles, groups, session_ids = build_cycle_dataset(SESSIONS, DIRECT_CFG)

X_feat = build_feature_matrix(X_cycles)
plot_pca_var(X_feat, RESULTS_DIR)

N_CLUSTERS = DIRECT_CFG['N_CLUSTERS']
print(f'\nClustering {len(X_feat)} cycles into {N_CLUSTERS} clusters...')
global_clusters, km_global, sc_global, pca_global = fit_cluster_pipeline(X_feat, N_CLUSTERS)

np.save(os.path.join(RESULTS_DIR, 'global_cluster_labels.npy'), global_clusters)
np.save(os.path.join(RESULTS_DIR, 'session_ids.npy'),           session_ids)

save_cluster_composition(global_clusters, session_ids, N_CLUSTERS, RESULTS_DIR)
plot_cluster_prototypes(X_cycles, global_clusters, N_CLUSTERS, RESULTS_DIR)
visualize_clusters_umap(X_feat, global_clusters, session_ids, RESULTS_DIR)

print('\n-- LOSO: PCA + GBM --')
f1_pca = run_loso_pca_gbm(X_feat, global_clusters, groups, N_CLUSTERS, RESULTS_DIR)

print('\n-- LOSO: Random Forest --')
f1_rf  = run_loso_rf(X_feat, global_clusters, groups, N_CLUSTERS, RESULTS_DIR)

# ── Summary ───────────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {'approach': 'PCA + GBM',              'macro_f1': f1_pca},
    {'approach': 'RF (768-D, balanced)',   'macro_f1': f1_rf},
]).sort_values('macro_f1', ascending=False).reset_index(drop=True)
summary.index += 1

print('\n' + '=' * 45)
print('LOSO macro-F1  --  cluster consistency')
print('=' * 45)
print(summary.to_string())
print('=' * 45)
summary.to_csv(os.path.join(RESULTS_DIR, 'loso_summary.csv'), index=False)

fig, ax = plt.subplots(figsize=(6, 2.5))
colors = ['#5DCAA5' if f == summary['macro_f1'].max() else '#7F77DD'
          for f in summary['macro_f1']]
ax.barh(summary['approach'][::-1], summary['macro_f1'][::-1],
        color=colors[::-1], edgecolor='white')
ax.set_xlabel('LOSO macro-F1'); ax.set_xlim(0, 1.0)
ax.set_title('PCA+GBM vs RF -- cluster-prediction consistency')
for i, row in summary[::-1].reset_index(drop=True).iterrows():
    ax.text(row['macro_f1'] + 0.01, i, f"{row['macro_f1']:.3f}", va='center', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'fig_method_comparison.png'), bbox_inches='tight')
plt.close(fig)
print('Results saved to', RESULTS_DIR)
print('Next: open cluster_composition.csv and assign pattern names per cluster.')
print('Tip:  pip install umap-learn   for the UMAP scatter.')
"""

nb = json.load(open(SRC, encoding='utf-8'))
nb2 = copy.deepcopy(nb)
nb2['cells'][34]['source'] = NEW_CELL
nb2['cells'][34]['outputs'] = []
nb2['cells'][34]['execution_count'] = None

with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, ensure_ascii=False, indent=1)
print(f'Created {DST}')
print(f'Cell 34 chars: {len(NEW_CELL)}')
