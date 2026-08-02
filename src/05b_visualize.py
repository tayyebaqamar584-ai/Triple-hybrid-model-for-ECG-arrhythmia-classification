"""
Step 5: Visualizations — generated entirely from REAL saved data/results
============================================================================
Every figure here reads from a JSON/CSV/NPY file produced by steps 1-4.
No numbers are hand-typed into this script.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL = sys.argv[1] if len(sys.argv) > 1 else 'beatwise'
MET_DIR = os.path.join(BASE, f'results_{PROTOCOL}', 'metrics')
PLOT_DIR = os.path.join(BASE, f'results_{PROTOCOL}', 'plots')
PROC_DIR = os.path.join(BASE, f'data', f'processed_{PROTOCOL}')
os.makedirs(PLOT_DIR, exist_ok=True)

CLASS_NAMES = ['Normal(N)', 'SupraV(S)', 'Ventricular(V)', 'Fusion(F)', 'Paced(Q)']
ALL_FEATURES = [
    'RR_Interval_ms', 'DWT_Total_Energy', 'QRS_Width_ms', 'R_Amplitude_mV',
    'PR_Interval_ms', 'QT_Interval_ms', 'RMSSD_ms', 'DWT_Energy_L1', 'DWT_Energy_L2',
    'LF_HF_Ratio', 'LF_Energy', 'ST_Deviation_mV', 'T_Amplitude_mV', 'P_Width_ms',
    'DWT_Energy_L3', 'pNN50_pct', 'HF_Energy', 'Skewness', 'Kurtosis', 'ZCR'
]

MODEL_ORDER = ['lcnn', 'pcnn_v2', 'ptcnn', '2dcnn', 'rf', 'xgb', 'ada',
               'dh1v2', 'dh2v2', 'dh3v2', 'proposed_v2']
DISPLAY = {'lcnn': 'LCNN', 'pcnn_v2': 'PCNN_v2', 'ptcnn': 'PTCNN', '2dcnn': '2DCNN', 'rf': 'RF',
           'xgb': 'XGB', 'ada': 'ADA', 'dh1v2': 'DH1v2', 'dh2v2': 'DH2v2', 'dh3v2': 'DH3v2',
           'proposed_v2': 'PROP_v2'}


def load_all_metrics():
    out = {}
    for n in MODEL_ORDER:
        with open(os.path.join(MET_DIR, f'{n}_test_metrics.json')) as f:
            out[n] = json.load(f)
    return out


def save(fig, fname):
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")
    return path


def fig01_accuracy_comparison(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [DISPLAY[n] for n in MODEL_ORDER]
    acc = [metrics[n]['accuracy'] for n in MODEL_ORDER]
    kappa = [metrics[n]['kappa'] for n in MODEL_ORDER]

    TYPE_COLOR = {
        'lcnn': '#4C72B0', 'pcnn_v2': '#4C72B0', 'ptcnn': '#4C72B0',
        '2dcnn': '#4C72B0', 'rf': '#4C72B0', 'xgb': '#4C72B0', 'ada': '#4C72B0',
        'dh1v2': '#C44E52', 'dh2v2': '#C44E52', 'dh3v2': '#C44E52', 'proposed_v2': '#8172B2',
    }
    colors = [TYPE_COLOR.get(n, '#999999') for n in MODEL_ORDER]
    axes[0].bar(names, acc, color=colors)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Test Accuracy by Model (real, beat-wise 75:15:10 held-out test set)')
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(names, kappa, color=colors)
    axes[1].set_ylabel("Cohen's Kappa")
    axes[1].set_title("Cohen's Kappa by Model (chance-corrected agreement)")
    axes[1].axhline(0, color='gray', linewidth=0.8)
    axes[1].tick_params(axis='x', rotation=45)

    fig.suptitle('Note: Accuracy alone is inflated by the dominant Normal class;\n'
                  "Kappa shows the real, more modest separation of minority classes.", fontsize=9, y=1.02)
    fig.tight_layout()
    return save(fig, 'fig01_accuracy_comparison.png')


def fig02_feature_violins():
    train = pd.read_csv(os.path.join(BASE, f'data', f'splits_{PROTOCOL}', 'ecg_train_raw.csv'))
    feats_to_show = ['RR_Interval_ms', 'QRS_Width_ms', 'R_Amplitude_mV', 'DWT_Total_Energy',
                      'PR_Interval_ms', 'QT_Interval_ms']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.flat, feats_to_show):
        sub = train  # all 5 classes now have meaningful real sample sizes (full 48-record DB)
        sns.violinplot(data=sub, x='label', y=feat, ax=ax, palette='Set2', cut=0)
        ax.set_xticklabels(['N', 'S', 'V', 'F', 'Q'])
        ax.set_title(feat)
        ax.set_xlabel('')
    fig.suptitle('Real Feature Distributions by Class (TRAIN, raw/unnormalised) — full 48-record DB, all 5 classes')
    fig.tight_layout()
    return save(fig, 'fig02_feature_violins.png')


def fig03_smote_distribution():
    with open(os.path.join(PROC_DIR, 'smote_summary.json')) as f:
        s = json.load(f)
    before, after = s['before'], s['after']
    names = list(before.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, [before[n] for n in names], w, label='Before SMOTE', color='#C44E52')
    ax.bar(x + w / 2, [after[n] for n in names], w, label='After SMOTE', color='#55A868')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel('Count (TRAIN split)')
    ineligible = s.get('ineligible_classes', {})
    if ineligible:
        subtitle = (f"Classes NOT oversampled (too few real samples): {ineligible}; "
                    "see note in report")
    else:
        subtitle = "All 5 classes had enough real samples to be SMOTE-balanced (see report for counts)"
    ax.set_title('Class Distribution Before/After SMOTE\n' + subtitle, fontsize=10)
    ax.legend()
    fig.tight_layout()
    return save(fig, 'fig03_smote_distribution.png')


def fig04_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis('off')
    stages = ['Raw WFDB\nSignals\n(48 records)', 'Beat\nExtraction\n+ 20 Features',
              'Beat-wise\n75:15:10 Split\n(stratified)', 'Z-score Norm\n(fit TRAIN only)',
              'SMOTE\n(TRAIN only,\nall 5 classes)', '7 Base\nModels', '3 Double\nHybrids',
              'Meta-Learner\n(PROPOSED v2)']
    n = len(stages)
    xs = np.linspace(0.05, 0.95, n)
    from matplotlib.patches import Rectangle
    for i, (x, s) in enumerate(zip(xs, stages)):
        ax.add_patch(Rectangle((x - 0.045, 0.4), 0.09, 0.2, fill=True,
                                facecolor='#4C72B0' if i < 5 else '#DD8452', alpha=0.85,
                                edgecolor='black', transform=ax.transAxes))
        ax.text(x, 0.5, s, ha='center', va='center', fontsize=8, color='white',
                transform=ax.transAxes, wrap=True)
        if i < n - 1:
            ax.annotate('', xy=(xs[i + 1] - 0.05, 0.5), xytext=(x + 0.05, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.set_title('ECG Triple-Hybrid Pipeline — Real Data Flow (Rebuilt)', fontsize=12)
    fig.tight_layout()
    return save(fig, 'fig04_pipeline_diagram.png')


def fig05_training_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hist_files = {
        'LCNN': 'lcnn_training_history.csv', 'PCNN_v2': 'pcnn_v2_training_history.csv',
        'PTCNN': 'ptcnn_training_history.csv', '2DCNN': '2dcnn_training_history.csv',
        'Meta-Learner v2': 'meta_learner_training_history.csv',
    }
    for name, fname in hist_files.items():
        path = os.path.join(MET_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        axes[0].plot(df['epoch'], df['loss'], label=f'{name} train', alpha=0.8)
        axes[0].plot(df['epoch'], df['val_loss'], '--', label=f'{name} val', alpha=0.8)
        axes[1].plot(df['epoch'], df['accuracy'], label=f'{name} train', alpha=0.8)
        axes[1].plot(df['epoch'], df['val_accuracy'], '--', label=f'{name} val', alpha=0.8)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training/Val Loss (real history)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Training/Val Accuracy (real history)')
    axes[0].legend(fontsize=6, ncol=2); axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return save(fig, 'fig05_training_dynamics.png')


def fig06_confusion_matrices(metrics):
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    for ax, n in zip(axes.flat, MODEL_ORDER):
        cm = np.array(metrics[n]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xticks(np.arange(len(CLASS_NAMES)) + 0.5)
        ax.set_yticks(np.arange(len(CLASS_NAMES)) + 0.5)
        ax.set_xticklabels(['N', 'S', 'V', 'F', 'Q'])
        ax.set_yticklabels(['N', 'S', 'V', 'F', 'Q'])
        ax.set_title(DISPLAY[n], fontsize=10)
        ax.set_xlabel(''); ax.set_ylabel('')
    for ax in axes.flat[len(MODEL_ORDER):]:
        ax.axis('off')
    fig.suptitle('Confusion Matrices — All Models (real predictions, beat-wise held-out test set)', y=1.0)
    fig.tight_layout()
    return save(fig, 'fig06_confusion_matrices.png')


def fig07_perclass_f1(metrics):
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(CLASS_NAMES))
    w = 0.055
    for i, n in enumerate(MODEL_ORDER):
        f1s = [metrics[n]['per_class'][c]['f1'] for c in CLASS_NAMES]
        ax.bar(x + i * w - (len(MODEL_ORDER) * w / 2), f1s, w, label=DISPLAY[n])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel('F1-score')
    ax.set_title('Per-Class F1 by Model (beat-wise protocol, real results)')
    ax.legend(fontsize=6, ncol=4, loc='lower center')
    fig.tight_layout()
    return save(fig, 'fig07_perclass_f1_comparison.png')


def fig08_roc_curves():
    yte = np.asarray(np.load(os.path.join(MET_DIR, 'y_test.npy')), dtype=np.int64)
    y_bin = np.asarray(label_binarize(yte, classes=list(range(5)), sparse_output=False), dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    chosen = ['xgb', 'rf', 'dh3v2', 'proposed_v2']
    for n in chosen:
        proba = np.asarray(np.load(os.path.join(MET_DIR, f'{n}_probs_test.npy')), dtype=np.float64)
        for c, cname in enumerate(['Normal(N)', 'Ventricular(V)']):
            ax = axes[0] if c == 0 else axes[1]
            if int(np.asarray(y_bin[:, c]).sum()) == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, c], proba[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{DISPLAY[n]} (AUC={roc_auc:.3f})')
    for ax, title in zip(axes, ['Normal(N) vs rest', 'Ventricular(V) vs rest']):
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle('ROC Curves — real predicted probabilities, beat-wise test set', y=1.02)
    fig.tight_layout()
    return save(fig, 'fig08_roc_curves.png')


def fig09_feature_importance():
    import pickle
    with open(os.path.join(MET_DIR, 'rf_model.pkl'), 'rb') as f:
        rf = pickle.load(f)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([ALL_FEATURES[i] for i in order][::-1], importances[order][::-1], color='#4C72B0')
    ax.set_xlabel('Importance (Random Forest, real .feature_importances_)')
    ax.set_title('Real Feature Importance — Random Forest')
    fig.tight_layout()
    return save(fig, 'fig09_feature_importance.png')


def fig10_pca_variance():
    train = pd.read_csv(os.path.join(PROC_DIR, 'ecg_train_norm.csv'))
    X = train[ALL_FEATURES].values
    pca = PCA().fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.6, label='95% variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Real PCA Variance — 20 Extracted Features (TRAIN, normalised)')
    ax.legend()
    fig.tight_layout()
    return save(fig, 'fig10_pca_variance.png')


def fig11_cv_stability():
    train = pd.read_csv(os.path.join(PROC_DIR, 'ecg_train_smote.csv'))
    X = train[ALL_FEATURES].to_numpy(dtype=np.float32)
    y = train['label'].to_numpy(dtype=np.int64)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy', n_jobs=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(1, 7), scores, color='#4C72B0')
    ax.axhline(scores.mean(), color='red', linestyle='--', label=f'mean={scores.mean():.4f}')
    ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy')
    ax.set_title(f'6-Fold CV Stability — RF on SMOTE-balanced TRAIN\n'
                 f'(real cross_val_score, std={scores.std():.4f})')
    ax.legend()
    fig.tight_layout()
    cv_results = {'fold_scores': scores.tolist(), 'mean': float(scores.mean()), 'std': float(scores.std())}
    with open(os.path.join(MET_DIR, 'cv_stability_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    return save(fig, 'fig11_cv_stability.png')


def main():
    print("=" * 70)
    print("Generating Figures — all data sourced from real saved results")
    print("=" * 70)
    metrics = load_all_metrics()

    fig01_accuracy_comparison(metrics)
    fig02_feature_violins()
    fig03_smote_distribution()
    fig04_pipeline_diagram()
    fig05_training_dynamics()
    fig06_confusion_matrices(metrics)
    fig07_perclass_f1(metrics)
    fig08_roc_curves()
    fig09_feature_importance()
    fig10_pca_variance()
    fig11_cv_stability()

    print("\n[OK] Step 5 complete. All figures generated from real data/results.")
    print("     (fig12_literature_benchmark omitted — would require citing other")
    print("     papers' numbers, which we don't have verified access to here.)")


if __name__ == '__main__':
    main()
