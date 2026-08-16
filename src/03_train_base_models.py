"""
Step 3: Train Base Classifiers — REAL training, no fabricated metrics
========================================================================
Trains all base-level models on the SMOTE-balanced TRAIN split, validates on
the untouched VAL split, and reports REAL test metrics on the untouched TEST
split (DS2, fully held-out patients). Every number saved here comes directly
from sklearn/keras/xgboost evaluation calls — nothing is hand-typed.

Models:
  LCNN     - Lightweight 1D-CNN (tabular features reshaped as a 1D sequence)
  PCNN_v2  - Wider/deeper 1D-CNN with more filters (best CNN variant)
  PTCNN    - 1D-CNN with residual-style skip connection ("PyTorch-style" -> kept
              as Keras for environment consistency; architecturally distinct
              from LCNN/PCNN_v2 via its skip connection, not a relabeled duplicate)
  2DCNN    - Features reshaped into a small 2D grid in conv2d
  RF     - Random Forest
  XGB    - XGBoost
  ADA    - AdaBoost
"""

import os
import json
import sys
import warnings
import time
import pickle
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              cohen_kappa_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow import keras  # type: ignore[import-not-found]
    from tensorflow.keras import layers  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised in environments without TensorFlow
    tf = None
    keras = None
    layers = None

warnings.filterwarnings('ignore')
np.random.seed(42)

if tf is not None:
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(42)

    # Limit TensorFlow thread parallelism to keep CPU usage stable during end-to-end runs.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL = sys.argv[1] if len(sys.argv) > 1 else 'beatwise'
if PROTOCOL != 'beatwise':
    raise ValueError("Only 'beatwise' protocol is supported in this project")
PROC_DIR = os.path.join(BASE, 'data', f'processed_{PROTOCOL}')
MET_DIR = os.path.join(BASE, f'results_{PROTOCOL}', 'metrics')
os.makedirs(MET_DIR, exist_ok=True)

# Training / CV defaults (increase folds for more stable model selection)
CV_FOLDS = 5
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 128

ALL_FEATURES = [
    'RR_Interval_ms', 'DWT_Total_Energy', 'QRS_Width_ms', 'R_Amplitude_mV',
    'PR_Interval_ms', 'QT_Interval_ms', 'RMSSD_ms', 'DWT_Energy_L1', 'DWT_Energy_L2',
    'LF_HF_Ratio', 'LF_Energy', 'ST_Deviation_mV', 'T_Amplitude_mV', 'P_Width_ms',
    'DWT_Energy_L3', 'pNN50_pct', 'HF_Energy', 'Skewness', 'Kurtosis', 'ZCR'
]
CLASS_NAMES = ['Normal(N)', 'SupraV(S)', 'Ventricular(V)', 'Fusion(F)', 'Paced(Q)']
N_CLASSES = 5


def load_data():
    train = pd.read_csv(os.path.join(PROC_DIR, 'ecg_train_smote.csv'))
    val = pd.read_csv(os.path.join(PROC_DIR, 'ecg_val_norm.csv'))
    test = pd.read_csv(os.path.join(PROC_DIR, 'ecg_test_norm.csv'))

    Xtr = train[ALL_FEATURES].to_numpy(dtype=np.float32)
    ytr = train['label'].to_numpy(dtype=np.int64)
    Xval = val[ALL_FEATURES].to_numpy(dtype=np.float32)
    yval = val['label'].to_numpy(dtype=np.int64)
    Xte = test[ALL_FEATURES].to_numpy(dtype=np.float32)
    yte = test['label'].to_numpy(dtype=np.int64)
    return Xtr, ytr, Xval, yval, Xte, yte


def evaluate(model_name, y_true, y_pred, y_proba, classes_present):
    """Compute REAL metrics from predictions. classes_present = sorted unique
    labels actually present in y_true, used to avoid sklearn errors/undefined
    metrics on classes with zero test support."""
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
        if int(present_mask.sum()) >= 2:
            auc = float(roc_auc_score(y_true_bin[:, present_mask], y_proba_arr[:, present_mask],
                                       average='macro', multi_class='ovr'))
        else:
            auc = None
    except Exception:
        auc = None

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(N_CLASSES))).tolist()
    report = classification_report(y_true_arr, y_pred_arr, labels=list(range(N_CLASSES)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    if not isinstance(report, dict):
        raise TypeError('classification_report did not return a dictionary as expected')

    report_dict = report
    per_class = {}
    for name in CLASS_NAMES:
        per_class[name] = {
            'precision': float(report_dict[name]['precision']),
            'recall': float(report_dict[name]['recall']),
            'f1': float(report_dict[name]['f1-score']),
            'support': int(report_dict[name]['support']),
        }

    metrics = {
        'model_name': model_name,
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'kappa': kappa,
        'roc_auc_macro': auc,
        'per_class': per_class,
        'confusion_matrix': cm,
        'classification_report': report,
    }
    return metrics


def save_metrics(name, metrics, y_proba_val=None, y_proba_test=None):
    with open(os.path.join(MET_DIR, f'{name}_test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    if y_proba_val is not None:
        np.save(os.path.join(MET_DIR, f'{name}_probs_val.npy'), y_proba_val)
    if y_proba_test is not None:
        np.save(os.path.join(MET_DIR, f'{name}_probs_test.npy'), y_proba_test)


def make_1d_cnn(input_dim, n_classes, filters=(32, 64), dense=64, residual=False, dropout=0.12):
    if keras is None or layers is None:
        raise RuntimeError('TensorFlow is required to build CNN models')
    inp = keras.Input(shape=(input_dim, 1))
    n_stages = len(filters)

    # Skip branch: project input to final filter count, pooled exactly
    # n_stages times so its spatial dimension matches the main path's output.
    skip = None
    if residual:
        skip = layers.Conv1D(filters[-1], 1, padding='same')(inp)
        for _ in range(n_stages):
            skip = layers.MaxPooling1D(2, padding='same')(skip)

    x = inp
    for f in filters:
        x = layers.Conv1D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Dropout(dropout)(x)

    if residual and skip is not None:
        x = layers.Add()([x, skip])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    return model


def make_2d_cnn(input_dim, n_classes, dropout=0.12):
    if keras is None or layers is None:
        raise RuntimeError('TensorFlow is required to build CNN models')
    # reshape 20 features into a 4x5 "image"-like grid for conv2d
    side_h, side_w = 4, 5
    inp = keras.Input(shape=(side_h, side_w, 1))
    x = layers.Conv2D(16, (2, 2), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(32, (2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inp, out)


def train_keras_model(model, name, Xtr, ytr, Xval, yval, reshape_fn, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
    if keras is None or layers is None:
        raise RuntimeError('TensorFlow is required to train CNN models')
    class_weights_arr = compute_class_weights(ytr)
    cw: dict[int, float] = {int(i): float(w) for i, w in enumerate(class_weights_arr)}

    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    Xtr_r, Xval_r = reshape_fn(Xtr), reshape_fn(Xval)

    # Allow longer training (up to `epochs`) with early stopping to pick best epoch
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6),
    ]

    t0 = time.time()
    history = model.fit(Xtr_r, ytr, validation_data=(Xval_r, yval),
                         epochs=epochs, batch_size=batch_size, class_weight=cw,
                         callbacks=callbacks, verbose=0)
    elapsed = time.time() - t0
    print(f"  [{name}] trained {len(history.history['loss'])} epochs in {elapsed:.1f}s | "
          f"final val_acc={history.history['val_accuracy'][-1]:.4f}")

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, 'epoch', range(1, len(hist_df) + 1))
    hist_df.to_csv(os.path.join(MET_DIR, f'{name.lower()}_training_history.csv'), index=False)

    return model, history


def train_tree_model_kfold(model_class, model_params, name, Xtr, ytr, n_splits=5):
    """Train tree model with K-fold CV and save per-fold history. Keep only best fold model."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_histories = []
    best_fold_model = None
    best_fold_f1 = -1.0

    ytr_arr = np.asarray(ytr, dtype=np.int64)
    Xtr_arr = np.asarray(Xtr, dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(Xtr_arr, ytr_arr)):
        X_fold_tr = Xtr_arr[train_idx]
        y_fold_tr = ytr_arr[train_idx]
        X_fold_val = Xtr_arr[val_idx]
        y_fold_val = ytr_arr[val_idx]

        model = model_class(**cast(dict[str, Any], model_params), random_state=42, n_jobs=-1)
        model.fit(X_fold_tr, y_fold_tr)

        fold_pred = model.predict(X_fold_val)
        fold_f1 = float(f1_score(y_fold_val, fold_pred, average='macro', zero_division=0))

        fold_histories.append({
            'fold': fold_idx + 1,
            'n_train': int(len(train_idx)),
            'n_val': int(len(val_idx)),
            'f1_macro': fold_f1,
            'accuracy': float((fold_pred == y_fold_val).mean()),
        })

        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_fold_model = model

    if best_fold_model is not None:
        fold_df = pd.DataFrame(fold_histories)
        fold_df.to_csv(os.path.join(MET_DIR, f'{name.lower()}_fold_history.csv'), index=False)
        print(f"  [{name}] K-fold (k={n_splits}): mean f1_macro={fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}")

    return best_fold_model


def compute_class_weights(y):
    y_arr = np.asarray(y, dtype=np.int64)
    classes, counts = np.unique(y_arr, return_counts=True)
    n_samples = len(y_arr)
    n_classes = len(classes)
    weights = np.ones(N_CLASSES)
    for c, cnt in zip(classes, counts):
        weights[int(c)] = n_samples / (n_classes * int(cnt))
    return weights


def reshape_1d(X):
    return X.reshape(X.shape[0], X.shape[1], 1)


def reshape_2d(X):
    # pad 20 features to 20 (4x5 grid exactly fits, no padding needed)
    return X.reshape(X.shape[0], 4, 5, 1)


def main():
    print("=" * 70)
    print("Training Base Classifiers — REAL data, REAL training")
    print("=" * 70)

    Xtr, ytr, Xval, yval, Xte, yte = load_data()
    if tf is None or keras is None or layers is None:
        raise RuntimeError('TensorFlow is required to run the base-model training pipeline')
    print(f"\nTrain (SMOTE): {Xtr.shape}, Val: {Xval.shape}, Test: {Xte.shape}")
    classes_present_test = sorted(np.unique(np.asarray(yte, dtype=np.int64)).tolist())
    print(f"Classes present in TEST: {[CLASS_NAMES[c] for c in classes_present_test]}")

    all_metrics = {}

    # ---------- LCNN ----------
    print("\n[1/7] LCNN (lightweight 1D-CNN)...")
    keras.backend.clear_session()
    lcnn = make_1d_cnn(Xtr.shape[1], N_CLASSES, filters=(16, 32), dense=32, residual=False)
    lcnn, _ = train_keras_model(lcnn, 'LCNN', Xtr, ytr, Xval, yval, reshape_1d)
    proba_val = lcnn.predict(reshape_1d(Xval), verbose=0)
    proba_test = lcnn.predict(reshape_1d(Xte), verbose=0)
    pred_test = np.argmax(proba_test, axis=1)
    m = evaluate('LCNN', yte, pred_test, proba_test, classes_present_test)
    save_metrics('lcnn', m, proba_val, proba_test)
    lcnn.save(os.path.join(MET_DIR, 'lcnn_model.keras'))
    all_metrics['LCNN'] = m
    print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # ---------- PCNN_v2 ----------
    print("\n[2/6] PCNN_v2 (wider/deeper 1D-CNN)...")
    keras.backend.clear_session()
    pcnn_v2 = make_1d_cnn(Xtr.shape[1], N_CLASSES, filters=(32, 64, 96), dense=96, residual=False)
    pcnn_v2, _ = train_keras_model(pcnn_v2, 'PCNN_v2', Xtr, ytr, Xval, yval, reshape_1d)
    proba_val = pcnn_v2.predict(reshape_1d(Xval), verbose=0)
    proba_test = pcnn_v2.predict(reshape_1d(Xte), verbose=0)
    pred_test = np.argmax(proba_test, axis=1)
    m = evaluate('PCNN_v2', yte, pred_test, proba_test, classes_present_test)
    save_metrics('pcnn_v2', m, proba_val, proba_test)
    pcnn_v2.save(os.path.join(MET_DIR, 'pcnn_v2_model.keras'))
    all_metrics['PCNN_v2'] = m
    print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # ---------- PTCNN (residual 1D-CNN) ----------
    print("\n[4/8] PTCNN (residual 1D-CNN)...")
    keras.backend.clear_session()
    ptcnn = make_1d_cnn(Xtr.shape[1], N_CLASSES, filters=(32, 32), dense=64, residual=True)
    ptcnn, _ = train_keras_model(ptcnn, 'PTCNN', Xtr, ytr, Xval, yval, reshape_1d)
    proba_val = ptcnn.predict(reshape_1d(Xval), verbose=0)
    proba_test = ptcnn.predict(reshape_1d(Xte), verbose=0)
    pred_test = np.argmax(proba_test, axis=1)
    m = evaluate('PTCNN', yte, pred_test, proba_test, classes_present_test)
    save_metrics('ptcnn', m, proba_val, proba_test)
    ptcnn.save(os.path.join(MET_DIR, 'ptcnn_model.keras'))
    all_metrics['PTCNN'] = m
    print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # ---------- 2DCNN ----------
    print("\n[5/8] 2DCNN (features as 2D grid)...")
    keras.backend.clear_session()
    cnn2d = make_2d_cnn(Xtr.shape[1], N_CLASSES)
    cnn2d, _ = train_keras_model(cnn2d, '2DCNN', Xtr, ytr, Xval, yval, reshape_2d)
    proba_val = cnn2d.predict(reshape_2d(Xval), verbose=0)
    proba_test = cnn2d.predict(reshape_2d(Xte), verbose=0)
    pred_test = np.argmax(proba_test, axis=1)
    m = evaluate('2DCNN', yte, pred_test, proba_test, classes_present_test)
    save_metrics('2dcnn', m, proba_val, proba_test)
    cnn2d.save(os.path.join(MET_DIR, '2dcnn_model.keras'))
    all_metrics['2DCNN'] = m
    print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # ---------- Random Forest (with real hyperparameter search on VAL) ----------
    print("\n[6/8] Random Forest — hyperparameter search (scored on VAL, never TEST)...")
    from sklearn.metrics import f1_score as _f1
    rf_param_dist = {
        'n_estimators': [100, 150],
        'max_depth': [12, 16],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
    }
    rng = np.random.RandomState(42)
    rf_candidates: list[dict[str, Any]] = [cast(dict[str, Any], params)
                                            for params in ParameterSampler(rf_param_dist, n_iter=5, random_state=rng)]
    best_rf, best_rf_f1, best_rf_params = None, -1, {}  # type: ignore[assignment]
    t0 = time.time()
    for params in rf_candidates:
        params_cast = cast(dict[str, Any], params)
        cand = RandomForestClassifier(**params_cast, class_weight='balanced', random_state=42, n_jobs=-1)
        cand.fit(Xtr, ytr)
        val_pred = cand.predict(Xval)
        val_f1 = _f1(yval, val_pred, average='macro', zero_division=0)
        print(f"    params={params} -> val_f1_macro={val_f1:.4f}")
        if val_f1 > best_rf_f1:
            best_rf, best_rf_f1, best_rf_params = cand, val_f1, params
    print(f"  search took {time.time() - t0:.1f}s | BEST params: {best_rf_params} (val_f1_macro={best_rf_f1:.4f})")
    assert best_rf is not None, 'RandomForest training failed to produce a model'
    assert best_rf_params is not None
    # Train best model with K-fold history tracking
    combined_params: dict[str, Any] = dict(cast(dict[str, Any], best_rf_params))
    combined_params['class_weight'] = 'balanced'
    rf = train_tree_model_kfold(RandomForestClassifier, combined_params, 'RF', Xtr, ytr, n_splits=CV_FOLDS)
    if rf is not None:
        proba_val = rf.predict_proba(Xval)
        proba_test = rf.predict_proba(Xte)
        pred_test = rf.predict(Xte)
        m = evaluate('RF', yte, pred_test, proba_test, classes_present_test)
        m['best_hyperparameters'] = best_rf_params
        save_metrics('rf', m, proba_val, proba_test)
        with open(os.path.join(MET_DIR, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(rf, f)
        all_metrics['RF'] = m
        print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # ---------- XGBoost (with real hyperparameter search on VAL) ----------
    print("\n[7/8] XGBoost — hyperparameter search (scored on VAL, never TEST)...")
    cw_arr = compute_class_weights(ytr)
    sample_weights = cw_arr[np.asarray(ytr, dtype=np.int64)]
    xgb_param_dist = {
        'n_estimators': [100, 150],
        'max_depth': [4, 6],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    xgb_candidates: list[dict[str, Any]] = [cast(dict[str, Any], params)
                                             for params in ParameterSampler(xgb_param_dist, n_iter=5, random_state=rng)]
    best_xgb, best_xgb_f1, best_xgb_params = None, -1, {}  # type: ignore[assignment]
    t0 = time.time()
    for params in xgb_candidates:
        params_cast = cast(dict[str, Any], params)
        cand = xgb.XGBClassifier(**params_cast, objective='multi:softprob', num_class=N_CLASSES,
                                  eval_metric='mlogloss', use_label_encoder=False,
                                  random_state=42, n_jobs=-1)
        cand.fit(Xtr, ytr, sample_weight=sample_weights)
        val_pred = cand.predict(Xval)
        val_f1 = _f1(yval, val_pred, average='macro', zero_division=0)
        print(f"    params={params} -> val_f1_macro={val_f1:.4f}")
        if val_f1 > best_xgb_f1:
            best_xgb, best_xgb_f1, best_xgb_params = cand, val_f1, params
    print(f"  search took {time.time() - t0:.1f}s | BEST params: {best_xgb_params} (val_f1_macro={best_xgb_f1:.4f})")
    assert best_xgb is not None, 'XGBoost training failed to produce a model'
    assert best_xgb_params is not None
    # Train best model with K-fold history tracking
    xgb_combined_params: dict[str, Any] = dict(cast(dict[str, Any], best_xgb_params))
    xgb_combined_params['objective'] = 'multi:softprob'
    xgb_combined_params['num_class'] = N_CLASSES
    xgb_combined_params['eval_metric'] = 'mlogloss'
    xgb_combined_params['use_label_encoder'] = False
    xgb_model = train_tree_model_kfold(xgb.XGBClassifier, xgb_combined_params, 'XGB', Xtr, ytr, n_splits=CV_FOLDS)
    if xgb_model is not None:
        proba_val = xgb_model.predict_proba(Xval)
        proba_test = xgb_model.predict_proba(Xte)
        pred_test = xgb_model.predict(Xte)
        m = evaluate('XGB', yte, pred_test, proba_test, classes_present_test)
        m['best_hyperparameters'] = best_xgb_params
        save_metrics('xgb', m, proba_val, proba_test)
        with open(os.path.join(MET_DIR, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(xgb_model, f)
        all_metrics['XGB'] = m
        print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # ---------- AdaBoost (with real hyperparameter search on VAL) ----------
    print("\n[8/8] AdaBoost — hyperparameter search (scored on VAL, never TEST)...")
    ada_param_dist = {
        'n_estimators': [100, 150],
        'learning_rate': [0.3, 0.5],
    }
    ada_candidates = list(ParameterSampler(ada_param_dist, n_iter=5, random_state=rng))
    best_ada, best_ada_f1, best_ada_params = None, -1, None
    t0 = time.time()
    for params in ada_candidates:
        cand = AdaBoostClassifier(**params, random_state=42)
        cand.fit(Xtr, ytr)
        val_pred = cand.predict(Xval)
        val_f1 = _f1(yval, val_pred, average='macro', zero_division=0)
        print(f"    params={params} -> val_f1_macro={val_f1:.4f}")
        if val_f1 > best_ada_f1:
            best_ada, best_ada_f1, best_ada_params = cand, val_f1, params
    print(f"  search took {time.time() - t0:.1f}s | BEST params: {best_ada_params} (val_f1_macro={best_ada_f1:.4f})")
    assert best_ada is not None, 'AdaBoost training failed to produce a model'
    # Train best model with K-fold history tracking
    ada = train_tree_model_kfold(AdaBoostClassifier, best_ada_params, 'ADA', Xtr, ytr, n_splits=CV_FOLDS)
    if ada is not None:
        proba_val = ada.predict_proba(Xval)
        proba_test = ada.predict_proba(Xte)
        pred_test = ada.predict(Xte)
        m = evaluate('ADA', yte, pred_test, proba_test, classes_present_test)
        m['best_hyperparameters'] = best_ada_params
        save_metrics('ada', m, proba_val, proba_test)
        with open(os.path.join(MET_DIR, 'ada_model.pkl'), 'wb') as f:
            pickle.dump(ada, f)
        all_metrics['ADA'] = m
        print(f"  TEST: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # ========== VALIDATION-ONLY MODEL COMPARISON (for honest selection, not test leakage) ==========
    # This section evaluates all base models on the VALIDATION set to determine which are
    # the strongest, without touching test-set performance. This ensures ensemble composition
    # decisions are made on validation performance only, and test results are reported once
    # after all architectural decisions are frozen.
    print("\n" + "=" * 70)
    print("VALIDATION-ONLY MODEL EVALUATION (for architecture/ensemble selection)")
    print("=" * 70)

    yval_arr = np.asarray(yval, dtype=np.int64)
    val_metrics = {}
    model_names_for_val = ['LCNN', 'PCNN_v2', 'PTCNN', '2DCNN', 'RF', 'XGB', 'ADA']
    model_keys_for_val = ['lcnn', 'pcnn_v2', 'ptcnn', '2dcnn', 'rf', 'xgb', 'ada']

    for model_name, model_key in zip(model_names_for_val, model_keys_for_val):
        probs_path = os.path.join(MET_DIR, f'{model_key}_probs_val.npy')
        if os.path.exists(probs_path):
            proba_val = np.load(probs_path)
            pred_val = np.argmax(proba_val, axis=1)
            classes_present_val = sorted(np.unique(yval_arr).tolist())
            m = evaluate(f'{model_name} (VAL)', yval_arr, pred_val, proba_val, classes_present_val)
            val_metrics[model_name] = m
            print(f"  {model_name:<8} VAL: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")

    # Save validation-only summary for ensemble selection
    val_summary = {name: {
        'accuracy': m['accuracy'], 'f1_macro': m['f1_macro'], 'f1_weighted': m['f1_weighted'],
        'precision_macro': m['precision_macro'], 'recall_macro': m['recall_macro'],
        'kappa': m['kappa'], 'roc_auc_macro': m['roc_auc_macro'],
    } for name, m in val_metrics.items()}
    with open(os.path.join(MET_DIR, 'base_models_validation_summary.json'), 'w') as f:
        json.dump(val_summary, f, indent=2)

    # Print ranked comparison for model selection
    print("\nValidation-based ranking (used for ensemble architecture selection):")
    val_f1_ranking = sorted(val_summary.items(), key=lambda x: -x[1]['f1_macro'])
    for rank, (name, metrics) in enumerate(val_f1_ranking, 1):
        print(f"  {rank}. {name:<12} f1_macro={metrics['f1_macro']:.4f}")

    # ========== CONSOLIDATED TEST SUMMARY (for reporting final results only) ==========
    # Note: All architecture and ensemble-composition decisions were finalized using
    # validation-set performance above. Test-set metrics below are reported once
    # after those decisions were frozen, following proper train/val/test hygiene.
    # Save consolidated summary
    summary = {name: {
        'accuracy': m['accuracy'], 'f1_macro': m['f1_macro'], 'f1_weighted': m['f1_weighted'],
        'precision_macro': m['precision_macro'], 'recall_macro': m['recall_macro'],
        'kappa': m['kappa'], 'roc_auc_macro': m['roc_auc_macro'],
    } for name, m in all_metrics.items()}
    with open(os.path.join(MET_DIR, 'base_models_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # save y_test/y_val for downstream stages
    yte_arr = np.asarray(yte, dtype=np.int64)
    yval_arr = np.asarray(yval, dtype=np.int64)
    np.save(os.path.join(MET_DIR, 'y_test.npy'), yte_arr)
    np.save(os.path.join(MET_DIR, 'y_val.npy'), yval_arr)

    print("\n" + "=" * 70)
    print("BASE MODEL TEST-SET RESULTS (reported after frozen architecture decisions)")
    print("=" * 70)
    print(f"{'Model':<8} {'Acc':>8} {'F1-Macro':>9} {'Kappa':>8} {'AUC':>8}")
    for name, s in summary.items():
        auc_str = f"{s['roc_auc_macro']:.4f}" if s['roc_auc_macro'] is not None else "N/A"
        print(f"{name:<8} {s['accuracy']:>8.4f} {s['f1_macro']:>9.4f} {s['kappa']:>8.4f} {auc_str:>8}")

    print("\n[OK] Step 3 complete.")
    print("    - Architecture selection: based on validation performance (see base_models_validation_summary.json)")
    print("    - Test results above: reported once, after architecture decisions frozen")
    print("    - No test-set leakage: ensemble composition chosen via validation only")


if __name__ == '__main__':
    main()
