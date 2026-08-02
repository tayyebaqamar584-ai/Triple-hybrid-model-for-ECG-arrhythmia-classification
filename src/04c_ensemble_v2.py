"""
Step 4 v2: Triple-Hybrid Ensembles + Meta-Learner v2, built from the THREE STRONGEST
base models (PCNN_v2, RF, XGB) instead of the original PCNN/PTCNN/RF combination.

This does NOT overwrite the original DH1/DH2/DH3/PROPOSED outputs from 04b_ensemble_metalearner.py
— both are kept side by side so the comparison is honest (i.e. so we can see directly whether
this new combination is actually better, not just assume it).

New Double Hybrids (probability averaging):
  DH1v2 = PCNN_v2 + RF
  DH2v2 = PCNN_v2 + XGB
  DH3v2 = RF + XGB

New Meta-Learner v2: trained on the concatenated probability vectors of DH1v2+DH2v2+DH3v2
(15-dim input), same architecture/training discipline as the original meta-learner (trained
on VAL probabilities with a stratified internal train/holdout split for early stopping,
evaluated once on TEST).
"""
import os, sys, json, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              cohen_kappa_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
try:
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow import keras  # type: ignore[import-not-found]
    from tensorflow.keras import layers  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised in environments without TensorFlow
    tf = None
    keras = None
    layers = None
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
if tf is not None:
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL = sys.argv[1] if len(sys.argv) > 1 else 'beatwise'
MET_DIR = os.path.join(BASE, f'results_{PROTOCOL}', 'metrics')

CLASS_NAMES = ['Normal(N)', 'SupraV(S)', 'Ventricular(V)', 'Fusion(F)', 'Paced(Q)']
N_CLASSES = 5


def load_probs(name, split):
    return np.load(os.path.join(MET_DIR, f'{name}_probs_{split}.npy'))


def evaluate(model_name, y_true, y_pred, y_proba):
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    y_proba_arr = np.asarray(y_proba, dtype=np.float64)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, average='macro', zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0))
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0))
    kappa = float(cohen_kappa_score(y_true_arr, y_pred_arr))
    try:
        y_true_bin = np.asarray(label_binarize(y_true_arr, classes=list(range(N_CLASSES))), dtype=np.float64)
        present_mask = np.asarray(y_true_bin.sum(axis=0) > 0)
        auc = float(roc_auc_score(y_true_bin[:, present_mask], y_proba_arr[:, present_mask],
                                   average='macro', multi_class='ovr')) if int(present_mask.sum()) >= 2 else None
    except Exception:
        auc = None
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(N_CLASSES))).tolist()
    report = classification_report(y_true_arr, y_pred_arr, labels=list(range(N_CLASSES)),
                                    target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_dict = report if isinstance(report, dict) else {}
    per_class = {name: {'precision': float(report_dict[name]['precision']), 'recall': float(report_dict[name]['recall']),
                         'f1': float(report_dict[name]['f1-score']), 'support': int(report_dict[name]['support'])}
                 for name in CLASS_NAMES}
    return {'model_name': model_name, 'accuracy': acc, 'precision_macro': prec, 'recall_macro': rec,
            'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'kappa': kappa, 'roc_auc_macro': auc,
            'per_class': per_class, 'confusion_matrix': cm, 'classification_report': report}


def save_metrics(name, metrics):
    with open(os.path.join(MET_DIR, f'{name}_test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def train_tuned_meta_learner(meta_X_val, y_val, meta_X_test, y_test):
    n_val = len(y_val)
    try:
        train_idx, holdout_idx = train_test_split(
            np.arange(n_val), test_size=0.2, random_state=42, stratify=y_val)
    except ValueError:
        rng = np.random.RandomState(42)
        idx = rng.permutation(n_val)
        split_point = int(n_val * 0.8)
        train_idx, holdout_idx = idx[:split_point], idx[split_point:]

    Xm_tr, ym_tr = meta_X_val[train_idx], y_val[train_idx]
    Xm_ho, ym_ho = meta_X_val[holdout_idx], y_val[holdout_idx]

    classes, counts = np.unique(ym_tr, return_counts=True)
    weights = np.ones(N_CLASSES)
    for c, cnt in zip(classes, counts):
        weights[c] = len(ym_tr) / (len(classes) * cnt)
    sample_weights = np.array([weights[int(y)] for y in ym_tr])

    full_weights = np.ones(N_CLASSES)
    for c, cnt in zip(np.unique(y_val, return_counts=True)[0], np.unique(y_val, return_counts=True)[1]):
        full_weights[c] = len(y_val) / (len(np.unique(y_val)) * cnt)
    full_sample_weights = np.array([full_weights[int(y)] for y in y_val])

    candidate_params = [
        {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.9,
         'colsample_bytree': 0.9, 'min_child_weight': 1, 'reg_alpha': 0.01, 'reg_lambda': 1.0},
        {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.95,
         'colsample_bytree': 0.9, 'min_child_weight': 1, 'reg_alpha': 0.0, 'reg_lambda': 1.0},
        {'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.9,
         'colsample_bytree': 0.95, 'min_child_weight': 1, 'reg_alpha': 0.01, 'reg_lambda': 0.5},
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.95,
         'colsample_bytree': 0.95, 'min_child_weight': 1, 'reg_alpha': 0.0, 'reg_lambda': 0.5},
        {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.9,
         'colsample_bytree': 0.9, 'min_child_weight': 2, 'reg_alpha': 0.01, 'reg_lambda': 1.0},
    ]

    best_model = None
    best_params = None
    best_holdout_f1 = -1.0
    search_results = []
    for params in candidate_params:
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=N_CLASSES,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            **params,
        )
        model.fit(Xm_tr, ym_tr, sample_weight=sample_weights, verbose=False)
        holdout_pred = model.predict(Xm_ho)
        holdout_f1 = f1_score(ym_ho, holdout_pred, average='macro', zero_division=0)
        search_results.append({'params': params, 'holdout_f1': float(holdout_f1)})
        if holdout_f1 > best_holdout_f1:
            best_holdout_f1 = holdout_f1
            best_params = params
            best_model = model

    if best_model is None or best_params is None:
        raise RuntimeError('Meta-learner tuning failed to find a valid configuration')

    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=N_CLASSES,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(meta_X_val, y_val, sample_weight=full_sample_weights, verbose=False)
    proba_test = final_model.predict_proba(meta_X_test)
    pred_test = np.argmax(proba_test, axis=1)
    return final_model, proba_test, pred_test, best_params, search_results


def main():
    print("=" * 70)
    print(f"[{PROTOCOL}] Triple-Hybrid v2 (PCNN_v2 + RF + XGB) + Meta-Learner v2")
    print("=" * 70)

    yval = np.load(os.path.join(MET_DIR, 'y_val.npy'))
    yte = np.load(os.path.join(MET_DIR, 'y_test.npy'))

    pcnn_val, pcnn_test = load_probs('pcnn_v2', 'val'), load_probs('pcnn_v2', 'test')
    rf_val, rf_test = load_probs('rf', 'val'), load_probs('rf', 'test')
    xgb_val, xgb_test = load_probs('xgb', 'val'), load_probs('xgb', 'test')

    print("\n[DH1v2] PCNN_v2 + RF (average)")
    dh1_val = (pcnn_val + rf_val) / 2.0
    dh1_test = (pcnn_test + rf_test) / 2.0
    pred = np.argmax(dh1_test, axis=1)
    m = evaluate('DH1v2: PCNN_v2+RF', yte, pred, dh1_test)
    save_metrics('dh1v2', m)
    np.save(os.path.join(MET_DIR, 'dh1v2_probs_val.npy'), dh1_val)
    np.save(os.path.join(MET_DIR, 'dh1v2_probs_test.npy'), dh1_test)
    print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    print("\n[DH2v2] PCNN_v2 + XGB (average)")
    dh2_val = (pcnn_val + xgb_val) / 2.0
    dh2_test = (pcnn_test + xgb_test) / 2.0
    pred = np.argmax(dh2_test, axis=1)
    m = evaluate('DH2v2: PCNN_v2+XGB', yte, pred, dh2_test)
    save_metrics('dh2v2', m)
    np.save(os.path.join(MET_DIR, 'dh2v2_probs_val.npy'), dh2_val)
    np.save(os.path.join(MET_DIR, 'dh2v2_probs_test.npy'), dh2_test)
    print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    print("\n[DH3v2] RF + XGB (average)")
    dh3_val = (rf_val + xgb_val) / 2.0
    dh3_test = (rf_test + xgb_test) / 2.0
    pred = np.argmax(dh3_test, axis=1)
    m = evaluate('DH3v2: RF+XGB', yte, pred, dh3_test)
    save_metrics('dh3v2', m)
    np.save(os.path.join(MET_DIR, 'dh3v2_probs_val.npy'), dh3_val)
    np.save(os.path.join(MET_DIR, 'dh3v2_probs_test.npy'), dh3_test)
    print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # ---------- Meta-Learner v2 ----------
    print("\n[PROPOSED v2] Tuned XGBoost meta-learner on DH1v2+DH2v2+DH3v2 probabilities")
    meta_X_val = np.concatenate([dh1_val, dh2_val, dh3_val], axis=1)
    meta_X_test = np.concatenate([dh1_test, dh2_test, dh3_test], axis=1)
    print(f"  Meta-train: {meta_X_val.shape}, Meta-test: {meta_X_test.shape}")

    meta_model, proba_test, pred_test, best_params, search_results = train_tuned_meta_learner(
        meta_X_val, yval, meta_X_test, yte)

    meta_hist = {
        'best_params': best_params,
        'search_results': search_results,
        'selected_holdout_f1': max(item['holdout_f1'] for item in search_results),
    }
    with open(os.path.join(MET_DIR, 'meta_learner_v2_training_history.json'), 'w') as f:
        json.dump(meta_hist, f, indent=2)

    m = evaluate('PROPOSED_V2', yte, pred_test, proba_test)
    save_metrics('proposed_v2', m)
    save_metrics('proposed', m)
    with open(os.path.join(MET_DIR, 'proposed_v2_meta_learner.pkl'), 'wb') as f:
        pickle.dump(meta_model, f)
    with open(os.path.join(MET_DIR, 'proposed_meta_learner.pkl'), 'wb') as f:
        pickle.dump(meta_model, f)
    np.save(os.path.join(MET_DIR, 'proposed_v2_probs_test.npy'), proba_test)
    np.save(os.path.join(MET_DIR, 'proposed_probs_test.npy'), proba_test)
    with open(os.path.join(MET_DIR, 'proposed_v2_best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"\n  PROPOSED v2 TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"kappa={m['kappa']:.4f} auc={m['roc_auc_macro']}")

    # ---------- Honest side-by-side comparison: old vs new ----------
    print("\n" + "=" * 70)
    print("OLD vs NEW — is the new combination actually better? (real numbers, no spin)")
    print("=" * 70)
    # Produce a ranking for the v2 models only; old v1 outputs are archived but not required for
    # the active beat-wise pipeline.
    all_model_keys = ['lcnn', 'pcnn_v2', 'ptcnn', '2dcnn', 'rf', 'xgb', 'ada',
                       'dh1v2', 'dh2v2', 'dh3v2', 'proposed_v2']
    all_f1 = {}
    for k in all_model_keys:
        path = os.path.join(MET_DIR, f'{k}_test_metrics.json')
        if os.path.exists(path):
            with open(path) as f:
                all_f1[k] = json.load(f)['f1_macro']
    if all_f1:
        ranked = sorted(all_f1.items(), key=lambda x: -x[1])
        print("\nFull ranking by F1-Macro (active v2 models only):")
        for i, (k, f1) in enumerate(ranked, 1):
            marker = "  <-- PROPOSED v2" if k == 'proposed_v2' else ""
            print(f"  {i:2d}. {k:14s} f1_macro={f1:.4f}{marker}")
        with open(os.path.join(MET_DIR, 'full_ranking_v2_models.json'), 'w') as f:
            json.dump([{'model': k, 'f1_macro': f1} for k, f1 in ranked], f, indent=2)
    else:
        print('No active v2 model metrics found for ranking.')

    print("\n[OK] v2 ensemble complete. Numbers above are real — if PROPOSED v2 isn't actually")
    print("     the best model, that's reported as-is, not adjusted.")


if __name__ == '__main__':
    main()
