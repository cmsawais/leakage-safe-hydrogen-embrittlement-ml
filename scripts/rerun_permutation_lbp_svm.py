import os, warnings
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern

DEVICE = 'cpu'


def build_metadata(image_dir):
    rows = []
    for path in sorted(Path(image_dir).glob('*.png')):
        try:
            parts = path.stem.split('_')
            if len(parts) != 4:
                raise ValueError
            material, condition, region, _ = parts
            rows.append({
                'path': str(path), 'filename': path.name,
                'material': material, 'condition': condition,
                'region': region,
                'group_id': f'{material}_{condition}_{region}',
            })
        except ValueError:
            pass
    return pd.DataFrame(rows)


def prepare_binary_task(df, material, conditions):
    sub = df[(df['material'] == material) & (df['condition'].isin(conditions))].copy()
    sub['label_name'] = sub['condition']
    lmap = {n: i for i, n in enumerate(sorted(sub['label_name'].unique()))}
    sub['label'] = sub['label_name'].map(lmap)
    return sub.reset_index(drop=True)


def _load_arr(path, size=96):
    return np.array(Image.open(path).convert('L').resize((size, size)), dtype=np.uint8)


def feat_lbp(arr, n_points=24, radius=3):
    lbp = local_binary_pattern(arr, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                            range=(0, n_points + 2), density=True)
    return hist.astype(np.float32)


def extract_lbp_features(paths, size=96):
    return np.array([feat_lbp(_load_arr(p, size)) for p in paths])


def permutation_test_loro(X, y, groups, n_permutations=500, seed=42, model_label='LBP+SVM'):
    logo = LeaveOneGroupOut()
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)

    g2lbl = {}
    for g in unique_groups:
        lbls = np.unique(y[groups == g])
        g2lbl[g] = int(lbls[0])
    orig_glabels = np.array([g2lbl[g] for g in unique_groups])

    def _score(y_use):
        yt_all, yp_all = [], []
        for tri, tei in logo.split(X, y_use, groups):
            if len(np.unique(y_use[tri])) < 2:
                continue
            sc = StandardScaler()
            clf = SVC(kernel='rbf', class_weight='balanced', random_state=seed)
            clf.fit(sc.fit_transform(X[tri]), y_use[tri])
            yt_all.extend(y_use[tei].tolist())
            yp_all.extend(clf.predict(sc.transform(X[tei])).tolist())
        if not yt_all or len(np.unique(yt_all)) < 2:
            return 0.5
        return balanced_accuracy_score(yt_all, yp_all)

    observed = _score(y)
    print(f'Observed balanced accuracy ({model_label}): {observed:.4f}')
    print(f'Running {n_permutations} permutations...')

    null_scores = []
    for i in range(n_permutations):
        perm_glabels = rng.permutation(orig_glabels)
        perm_map = {g: int(l) for g, l in zip(unique_groups, perm_glabels)}
        y_perm = np.array([perm_map[g] for g in groups])
        null_scores.append(_score(y_perm))
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{n_permutations} done...')

    null_scores = np.array(null_scores)
    p_value = (np.sum(null_scores >= observed) + 1) / (n_permutations + 1)
    return observed, null_scores, p_value


if __name__ == '__main__':
    DATA_DIR = Path(os.environ.get('SEM_DATA_DIR', Path(__file__).resolve().parent.parent / 'data' / 'Mixed'))
    df = build_metadata(DATA_DIR)
    task_316 = prepare_binary_task(df, material='316L', conditions=['AR', 'H2'])
    print(f'316L task: {len(task_316)} images, {task_316["group_id"].nunique()} regions')

    paths_316 = task_316['path'].tolist()
    X_lbp = extract_lbp_features(paths_316)
    y_316 = task_316['label'].to_numpy()
    g_316 = task_316['group_id'].to_numpy()

    observed_ba, null_dist, p_val = permutation_test_loro(
        X_lbp, y_316, g_316, n_permutations=500, seed=42, model_label='LBP+SVM'
    )

    print(f'\nObserved BA = {observed_ba:.4f}   p-value = {p_val:.4f}')
    if p_val < 0.05:
        print('SIGNIFICANT: LBP features predict H2 condition above chance (p < 0.05)')
    elif p_val < 0.10:
        print('MARGINAL: p < 0.10 — borderline significance, report honestly')
    else:
        print('NOT significant at p < 0.05 — results are consistent with chance')

    # Save the permutation distribution plot for LBP+SVM
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_dist, bins=30, color='steelblue', alpha=0.75, label='Null distribution')
    ax.axvline(observed_ba, color='red', lw=2.0, ls='--',
               label=f'Observed BA = {observed_ba:.3f}')
    ax.axvline(0.5, color='gray', lw=1.2, ls=':', label='Chance = 0.5')
    pct = np.percentile(null_dist, 95)
    ax.axvline(pct, color='orange', lw=1.5, ls='-.', label=f'95th pct = {pct:.3f}')
    ax.set_xlabel('Balanced Accuracy (LBP+SVM)')
    ax.set_ylabel('Count')
    ax.set_title(f'Group-Level Permutation Test (LBP+SVM, n={len(null_dist)})   p = {p_val:.4f}',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / 'figures' / 'permutation_test_lbp_svm.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved figure to {out_path}')

    null_path = Path(__file__).resolve().parent.parent / 'results' / 'permutation_null_lbp_svm.csv'
    pd.DataFrame({'balanced_accuracy': null_dist}).to_csv(null_path, index=False)
    print(f'Saved null distribution to {null_path}')
