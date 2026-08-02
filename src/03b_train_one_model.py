"""
Step 3 (parameterized): Train all base models for ONE protocol at a time.
Usage: python 03b_train_one_model.py beatwise <model_name>
    protocol: beatwise
  model_name: lcnn | pcnn_v2 | ptcnn | 2dcnn | rf | xgb | ada

Each call trains and evaluates exactly ONE model, to fit within compute/time budget
per call on this single-core environment. Real training, real evaluation — RF/XGB/ADA
use fixed, reasonable hyperparameters (informed by the lighter, full hyperparameter
search already done in the earlier 23-record and 48-record/single-protocol runs) rather
than a fresh exhaustive search per protocol, in the interest of covering both protocols
completely. This tradeoff is stated explicitly in the final report.
"""
import os
import sys
import json
import time
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Any, Dict, Tuple, cast
from sklearn.preprocessing import label_binarize
import xgboost as xgb

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_FEATURES = [
    'RR_Interval_ms', 'DWT_Total_Energy', 'QRS_Width_ms', 'R_Amplitude_mV',
    'PR_Interval_ms', 'QT_Interval_ms', 'RMSSD_ms', 'DWT_Energy_L1', 'DWT_Energy_L2',
    'LF_HF_Ratio', 'LF_Energy', 'ST_Deviation_mV', 'T_Amplitude_mV', 'P_Width_ms',
    'DWT_Energy_L3', 'pNN50_pct', 'HF_Energy', 'Skewness', 'Kurtosis', 'ZCR'
]
CLASS_NAMES = ['Normal(N)', 'SupraV(S)', 'Ventricular(V)', 'Fusion(F)', 'Paced(Q)']
N_CLASSES = 5


def load_data(protocol) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proc_dir = os.path.join(BASE, 'data', f'processed_{protocol}')
    train = pd.read_csv(os.path.join(proc_dir, 'ecg_train_smote.csv'))
    val = pd.read_csv(os.path.join(proc_dir, 'ecg_val_norm.csv'))
    test = pd.read_csv(os.path.join(proc_dir, 'ecg_test_norm.csv'))
    Xtr = train[ALL_FEATURES].to_numpy(dtype=np.float32)
    ytr = train['label'].to_numpy(dtype=np.int64)
    Xval = val[ALL_FEATURES].to_numpy(dtype=np.float32)
    yval = val['label'].to_numpy(dtype=np.int64)
    Xte = test[ALL_FEATURES].to_numpy(dtype=np.float32)
    yte = test['label'].to_numpy(dtype=np.int64)
    return Xtr, ytr, Xval, yval, Xte, yte


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    weights = np.ones(N_CLASSES)
    for c, cnt in zip(classes, counts):
        weights[c] = n_samples / (len(classes) * cnt)
    return weights


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
    report_dict = report if isinstance(report, dict) else {}
    per_class = {
        name: {
            'precision': float(report_dict.get(name, {}).get('precision', 0)),
            'recall': float(report_dict.get(name, {}).get('recall', 0)),
            'f1': float(report_dict.get(name, {}).get('f1-score', 0)),
            'support': int(report_dict.get(name, {}).get('support', 0)),
        }
        for name in CLASS_NAMES
    }
    return {'model_name': model_name, 'accuracy': acc, 'precision_macro': prec, 'recall_macro': rec,
            'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'kappa': kappa, 'roc_auc_macro': auc,
            'per_class': per_class, 'confusion_matrix': cm, 'classification_report': report}


def reshape_1d(X):
    return X.reshape(X.shape[0], X.shape[1], 1)


def reshape_2d(X):
    return X.reshape(X.shape[0], 4, 5, 1)


def make_1d_cnn(input_dim, n_classes, filters=(32, 64), dense=64, residual=False, dropout=0.3):  # type: ignore[no-untyped-def]
    try:  # type: ignore[no-untyped-def]
        from tensorflow import keras  # type: ignore[import-not-found]
        from tensorflow.keras import layers  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError('TensorFlow is required to build CNN models')
    inp = keras.Input(shape=(input_dim, 1))
    n_stages = len(filters)
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
    return keras.Model(inp, out)


def make_2d_cnn(input_dim, n_classes):  # type: ignore[no-untyped-def]
    try:  # type: ignore[no-untyped-def]
        from tensorflow import keras  # type: ignore[import-not-found]
        from tensorflow.keras import layers  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError('TensorFlow is required to build CNN models')
    inp = keras.Input(shape=(4, 5, 1))
    x = layers.Conv2D(16, (2, 2), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(32, (2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inp, out)


def train_cnn_chunked(protocol, model_key, chunk_epochs=10, max_epochs=80):  # type: ignore[no-untyped-def]
    """Train in small epoch chunks, checkpointing model + history + early-stop state to disk
    after each chunk, so training can resume across multiple calls if a single call's wall-clock
    budget is exceeded. Always makes forward progress regardless of how many calls it takes."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        from tensorflow import keras  # type: ignore[import-not-found]
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError('TensorFlow is required to train CNN models')
    tf.random.set_seed(42)
    np.random.seed(42)

    Xtr, ytr, Xval, yval, Xte, yte = load_data(protocol)
    met_dir = os.path.join(BASE, f'results_{protocol}', 'metrics')
    os.makedirs(met_dir, exist_ok=True)
    ckpt_path = os.path.join(met_dir, f'{model_key}_checkpoint.keras')
    state_path = os.path.join(met_dir, f'{model_key}_train_state.json')

    cfg = {
        'lcnn': dict(filters=(16, 32), dense=32, residual=False, reshape=reshape_1d, is2d=False),
        'pcnn_v2': dict(filters=(64, 128, 128), dense=128, residual=False, reshape=reshape_1d, is2d=False),
        'ptcnn': dict(filters=(32, 32), dense=64, residual=True, reshape=reshape_1d, is2d=False),
        '2dcnn': dict(reshape=reshape_2d, is2d=True),
    }[model_key]
    reshape_fn = cfg['reshape']

    if os.path.exists(ckpt_path) and os.path.exists(state_path):
        model = keras.models.load_model(ckpt_path)
        with open(state_path) as f:
            state = json.load(f)
        print(f"  resuming {model_key} from epoch {state['epoch']} "
              f"(best_val_loss={state['best_val_loss']:.4f}, no_improve={state['no_improve']})")
    else:
        if cfg['is2d']:
            model = make_2d_cnn(Xtr.shape[1], N_CLASSES)
        else:
            model = make_1d_cnn(Xtr.shape[1], N_CLASSES, filters=cfg['filters'], dense=cfg['dense'],
                                 residual=cfg['residual'])
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        state = {'epoch': 0, 'best_val_loss': float('inf'), 'no_improve': 0, 'history': [],
                  'lr': 1e-3, 'lr_no_improve': 0}

    cw: dict[int, float] = {int(i): float(w) for i, w in enumerate(compute_class_weights(ytr))}
    patience, lr_patience = 7, 3
    Xtr_r, Xval_r = reshape_fn(Xtr), reshape_fn(Xval)

    t0 = time.time()
    stopped_early = False
    while state['epoch'] < max_epochs and state['no_improve'] < patience:
        h = model.fit(Xtr_r, ytr, validation_data=(Xval_r, yval), epochs=1, batch_size=512,
                       class_weight=cw, verbose=0)
        state['epoch'] += 1
        val_loss = float(h.history['val_loss'][0])
        state['history'].append({'epoch': state['epoch'], 'loss': float(h.history['loss'][0]),
                                  'accuracy': float(h.history['accuracy'][0]),
                                  'val_loss': val_loss, 'val_accuracy': float(h.history['val_accuracy'][0])})
        if val_loss < state['best_val_loss'] - 1e-5:
            state['best_val_loss'] = val_loss
            state['no_improve'] = 0
            state['lr_no_improve'] = 0
            model.save(os.path.join(met_dir, f'{model_key}_best.keras'))
        else:
            state['no_improve'] += 1
            state['lr_no_improve'] += 1
            if state['lr_no_improve'] >= lr_patience:
                state['lr'] *= 0.5
                optimizer = getattr(model, 'optimizer', None)
                lr_attr = getattr(optimizer, 'learning_rate', None) if optimizer is not None else None
                if lr_attr is not None and hasattr(lr_attr, 'assign'):
                    lr_attr.assign(float(state['lr']))
                state['lr_no_improve'] = 0
        # checkpoint every epoch so progress is never lost to a call timeout
        model.save(ckpt_path)
        with open(state_path, 'w') as f:
            json.dump(state, f)
        if time.time() - t0 > 110:  # conservative — leave headroom for save overhead
            break
    else:
        if state['no_improve'] >= patience:
            stopped_early = True

    done = stopped_early or state['epoch'] >= max_epochs
    print(f"  [{model_key}] now at epoch {state['epoch']}/{max_epochs}, "
          f"no_improve={state['no_improve']}/{patience}, done={done} ({time.time()-t0:.1f}s this call)")

    if done:
        best_model = keras.models.load_model(os.path.join(met_dir, f'{model_key}_best.keras'))
        _finalize_cnn(best_model, model_key, met_dir, Xval, yval, Xte, yte, reshape_fn, state['history'])
    return done


def _finalize_cnn(model, model_key, met_dir, Xval, yval, Xte, yte, reshape_fn, history_list):
    hist_df = pd.DataFrame(history_list)
    hist_df.to_csv(os.path.join(met_dir, f'{model_key}_training_history.csv'), index=False)

    proba_val = model.predict(reshape_fn(Xval), verbose=0)
    proba_test = model.predict(reshape_fn(Xte), verbose=0)
    pred_test = np.argmax(proba_test, axis=1)
    m = evaluate(model_key.upper(), yte, pred_test, proba_test)
    with open(os.path.join(met_dir, f'{model_key}_test_metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    np.save(os.path.join(met_dir, f'{model_key}_probs_val.npy'), proba_val)
    np.save(os.path.join(met_dir, f'{model_key}_probs_test.npy'), proba_test)
    model.save(os.path.join(met_dir, f'{model_key}_model.keras'))
    np.save(os.path.join(met_dir, 'y_test.npy'), yte)
    np.save(os.path.join(met_dir, 'y_val.npy'), yval)
    print(f"  FINAL [{model_key}] TEST acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")
    # clean up checkpoint/state so a future re-run starts fresh, not resumes a finished run
    for fn in [f'{model_key}_checkpoint.keras', f'{model_key}_train_state.json', f'{model_key}_best.keras']:
        p = os.path.join(met_dir, fn)
        if os.path.exists(p):
            os.remove(p)


def train_cnn(protocol, model_key):  # type: ignore[no-untyped-def]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        from tensorflow import keras  # type: ignore[import-not-found]
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError('TensorFlow is required to train CNN models')
    tf.random.set_seed(42)
    np.random.seed(42)

    Xtr, ytr, Xval, yval, Xte, yte = load_data(protocol)
    met_dir = os.path.join(BASE, f'results_{protocol}', 'metrics')
    os.makedirs(met_dir, exist_ok=True)

    cfg = {
        'lcnn': dict(filters=(16, 32), dense=32, residual=False, reshape=reshape_1d, is2d=False),
        'pcnn_v2': dict(filters=(64, 128, 128), dense=128, residual=False, reshape=reshape_1d, is2d=False),
        'ptcnn': dict(filters=(32, 32), dense=64, residual=True, reshape=reshape_1d, is2d=False),
        '2dcnn': dict(reshape=reshape_2d, is2d=True),
    }[model_key]

    if cfg['is2d']:
        model = make_2d_cnn(Xtr.shape[1], N_CLASSES)
    else:
        model = make_1d_cnn(Xtr.shape[1], N_CLASSES, filters=cfg['filters'], dense=cfg['dense'],
                             residual=cfg['residual'])

    cw = {int(i): float(w) for i, w in enumerate(compute_class_weights(ytr))}
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    reshape_fn = cfg['reshape']
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    ]
    t0 = time.time()
    history = model.fit(reshape_fn(Xtr), ytr, validation_data=(reshape_fn(Xval), yval),
                         epochs=80, batch_size=512, class_weight=cw, callbacks=callbacks, verbose=0)
    print(f"  [{model_key}] trained {len(history.history['loss'])} epochs in {time.time()-t0:.1f}s")

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, 'epoch', range(1, len(hist_df) + 1))
    hist_df.to_csv(os.path.join(met_dir, f'{model_key}_training_history.csv'), index=False)

    proba_val = model.predict(reshape_fn(Xval), verbose=0)
    proba_test = model.predict(reshape_fn(Xte), verbose=0)
    pred_test = np.argmax(proba_test, axis=1)
    m = evaluate(model_key.upper(), yte, pred_test, proba_test)
    with open(os.path.join(met_dir, f'{model_key}_test_metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    np.save(os.path.join(met_dir, f'{model_key}_probs_val.npy'), proba_val)
    np.save(os.path.join(met_dir, f'{model_key}_probs_test.npy'), proba_test)
    model.save(os.path.join(met_dir, f'{model_key}_model.keras'))
    np.save(os.path.join(met_dir, 'y_test.npy'), np.asarray(yte, dtype=np.int64))
    np.save(os.path.join(met_dir, 'y_val.npy'), np.asarray(yval, dtype=np.int64))
    print(f"  TEST acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")


def select_rf_params(Xtr, ytr, Xval, yval) -> tuple[dict[str, Any], float]:
    candidates: list[dict[str, Any]] = [
        {'n_estimators': 250, 'max_depth': 18, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        {'n_estimators': 400, 'max_depth': 22, 'min_samples_leaf': 1, 'max_features': 'log2'},
        {'n_estimators': 500, 'max_depth': 24, 'min_samples_leaf': 1, 'max_features': 'log2'},
    ]
    best_params: dict[str, Any] = {}
    best_score = -1.0
    for params in candidates:
        params_cast = cast(dict[str, Any], params)
        model = RandomForestClassifier(**params_cast, class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        pred_val = model.predict(Xval)
        score = f1_score(yval, pred_val, average='macro')
        if score > best_score:
            best_score = float(score)
            best_params = params
    return best_params, best_score


def select_xgb_params(Xtr, ytr, Xval, yval) -> tuple[dict[str, Any], float]:
    cw_arr = compute_class_weights(ytr)
    sample_weights = cw_arr[ytr]
    candidates: list[dict[str, Any]] = [
        {'n_estimators': 250, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 1.0, 'colsample_bytree': 0.9},
        {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 1.0},
    ]
    best_params: dict[str, Any] = {}
    best_score = -1.0
    for params in candidates:
        params_cast = cast(dict[str, Any], params)
        model = xgb.XGBClassifier(**params_cast, objective='multi:softprob', num_class=N_CLASSES,
                                  eval_metric='mlogloss', random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr, sample_weight=sample_weights)
        pred_val = model.predict(Xval)
        score = f1_score(yval, pred_val, average='macro')
        if score > best_score:
            best_score = float(score)
            best_params = params
    return best_params, best_score


def train_rf(protocol):  # type: ignore[no-untyped-def]
    Xtr, ytr, Xval, yval, Xte, yte = load_data(protocol)
    met_dir = os.path.join(BASE, f'results_{protocol}', 'metrics')
    os.makedirs(met_dir, exist_ok=True)
    params, _ = select_rf_params(Xtr, ytr, Xval, yval)
    params = cast(dict[str, Any], params)
    t0 = time.time()
    from sklearn.model_selection import StratifiedKFold
    fold_histories = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_fold_model = None
    best_fold_f1 = -1.0

    Xtr_arr = np.asarray(Xtr, dtype=np.float32)
    ytr_arr = np.asarray(ytr, dtype=np.int64)
    Xval_arr = np.asarray(Xval, dtype=np.float32)
    Xte_arr = np.asarray(Xte, dtype=np.float32)
    yte_arr = np.asarray(yte, dtype=np.int64)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(Xtr_arr, ytr_arr)):
        X_fold_tr = Xtr_arr[train_idx]
        y_fold_tr = ytr_arr[train_idx]
        X_fold_val = Xtr_arr[val_idx]
        y_fold_val = ytr_arr[val_idx]
        fold_model = RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1)
        fold_model.fit(X_fold_tr, y_fold_tr)
        fold_pred = fold_model.predict(X_fold_val)
        fold_f1 = float(f1_score(y_fold_val, fold_pred, average='macro', zero_division=0))
        fold_acc = float((fold_pred == y_fold_val).mean())
        fold_histories.append({'fold': fold_idx + 1, 'f1_macro': fold_f1, 'accuracy': fold_acc})
        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_fold_model = fold_model
    rf = best_fold_model
    if rf is not None:
        fold_df = pd.DataFrame(fold_histories)
        fold_df.to_csv(os.path.join(met_dir, 'rf_fold_history.csv'), index=False)
        print(f"  [RF] K-fold (k=5): mean f1_macro={fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}, trained in {time.time()-t0:.1f}s, params={params}")
        proba_val = rf.predict_proba(Xval_arr)
        proba_test = rf.predict_proba(Xte_arr)
        pred_test = rf.predict(Xte_arr)
        m = evaluate('RF', yte_arr, pred_test, proba_test)
        m['hyperparameters'] = params
        with open(os.path.join(met_dir, 'rf_test_metrics.json'), 'w') as f:
            json.dump(m, f, indent=2)
        np.save(os.path.join(met_dir, 'rf_probs_val.npy'), proba_val)
        np.save(os.path.join(met_dir, 'rf_probs_test.npy'), proba_test)
        with open(os.path.join(met_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(rf, f)
        print(f"  TEST acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")


def train_xgb(protocol):  # type: ignore[no-untyped-def]
    Xtr, ytr, Xval, yval, Xte, yte = load_data(protocol)
    met_dir = os.path.join(BASE, f'results_{protocol}', 'metrics')
    os.makedirs(met_dir, exist_ok=True)
    params, _ = select_xgb_params(Xtr, ytr, Xval, yval)
    cw_arr = compute_class_weights(ytr)
    t0 = time.time()
    from sklearn.model_selection import StratifiedKFold
    fold_histories = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_fold_model = None
    best_fold_f1 = -1.0

    Xtr_arr = np.asarray(Xtr, dtype=np.float32)
    ytr_arr = np.asarray(ytr, dtype=np.int64)
    Xval_arr = np.asarray(Xval, dtype=np.float32)
    Xte_arr = np.asarray(Xte, dtype=np.float32)
    yte_arr = np.asarray(yte, dtype=np.int64)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(Xtr_arr, ytr_arr)):
        X_fold_tr = Xtr_arr[train_idx]
        y_fold_tr = ytr_arr[train_idx]
        X_fold_val = Xtr_arr[val_idx]
        y_fold_val = ytr_arr[val_idx]
        fold_weights = cw_arr[y_fold_tr]
        fold_model = xgb.XGBClassifier(**params, objective='multi:softprob', num_class=N_CLASSES,
                                       eval_metric='mlogloss', random_state=42, n_jobs=-1)
        fold_model.fit(X_fold_tr, y_fold_tr, sample_weight=fold_weights)
        fold_pred = fold_model.predict(X_fold_val)
        fold_f1 = float(f1_score(y_fold_val, fold_pred, average='macro', zero_division=0))
        fold_acc = float((fold_pred == y_fold_val).mean())
        fold_histories.append({'fold': fold_idx + 1, 'f1_macro': fold_f1, 'accuracy': fold_acc})
        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_fold_model = fold_model
    model = best_fold_model
    if model is not None:
        fold_df = pd.DataFrame(fold_histories)
        fold_df.to_csv(os.path.join(met_dir, 'xgb_fold_history.csv'), index=False)
        print(f"  [XGB] K-fold (k=5): mean f1_macro={fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}, trained in {time.time()-t0:.1f}s, params={params}")
        proba_val = model.predict_proba(Xval_arr)
        proba_test = model.predict_proba(Xte_arr)
        pred_test = model.predict(Xte_arr)
        m = evaluate('XGB', yte_arr, pred_test, proba_test)
        m['hyperparameters'] = params
        with open(os.path.join(met_dir, 'xgb_test_metrics.json'), 'w') as f:
            json.dump(m, f, indent=2)
        np.save(os.path.join(met_dir, 'xgb_probs_val.npy'), proba_val)
        np.save(os.path.join(met_dir, 'xgb_probs_test.npy'), proba_test)
        with open(os.path.join(met_dir, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print(f"  TEST acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")


def train_ada(protocol):  # type: ignore[no-untyped-def]
    Xtr, ytr, Xval, yval, Xte, yte = load_data(protocol)
    met_dir = os.path.join(BASE, f'results_{protocol}', 'metrics')
    os.makedirs(met_dir, exist_ok=True)
    params = {'n_estimators': 100, 'learning_rate': 0.5}
    t0 = time.time()
    from sklearn.model_selection import StratifiedKFold
    fold_histories = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_fold_model = None
    best_fold_f1 = -1.0

    Xtr_arr = np.asarray(Xtr, dtype=np.float32)
    ytr_arr = np.asarray(ytr, dtype=np.int64)
    Xval_arr = np.asarray(Xval, dtype=np.float32)
    Xte_arr = np.asarray(Xte, dtype=np.float32)
    yte_arr = np.asarray(yte, dtype=np.int64)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(Xtr_arr, ytr_arr)):
        X_fold_tr = Xtr_arr[train_idx]
        y_fold_tr = ytr_arr[train_idx]
        X_fold_val = Xtr_arr[val_idx]
        y_fold_val = ytr_arr[val_idx]
        fold_model = AdaBoostClassifier(**params, random_state=42)
        fold_model.fit(X_fold_tr, y_fold_tr)
        fold_pred = fold_model.predict(X_fold_val)
        fold_f1 = float(f1_score(y_fold_val, fold_pred, average='macro', zero_division=0))
        fold_acc = float((fold_pred == y_fold_val).mean())
        fold_histories.append({'fold': fold_idx + 1, 'f1_macro': fold_f1, 'accuracy': fold_acc})
        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_fold_model = fold_model
    model = best_fold_model
    if model is not None:
        fold_df = pd.DataFrame(fold_histories)
        fold_df.to_csv(os.path.join(met_dir, 'ada_fold_history.csv'), index=False)
        print(f"  [ADA] K-fold (k=5): mean f1_macro={fold_df['f1_macro'].mean():.4f} ± {fold_df['f1_macro'].std():.4f}, trained in {time.time()-t0:.1f}s, params={params}")
        proba_val = model.predict_proba(Xval_arr)
        proba_test = model.predict_proba(Xte_arr)
        pred_test = model.predict(Xte_arr)
        m = evaluate('ADA', yte_arr, pred_test, proba_test)
        m['hyperparameters'] = params
        with open(os.path.join(met_dir, 'ada_test_metrics.json'), 'w') as f:
            json.dump(m, f, indent=2)
        np.save(os.path.join(met_dir, 'ada_probs_val.npy'), proba_val)
        np.save(os.path.join(met_dir, 'ada_probs_test.npy'), proba_test)
        with open(os.path.join(met_dir, 'ada_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print(f"  TEST acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} kappa={m['kappa']:.4f}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Usage: python 03b_train_one_model.py beatwise <model_name>")
    protocol = sys.argv[1]
    model_key = sys.argv[2]
    if protocol != 'beatwise':
        raise ValueError("Only 'beatwise' protocol is supported in this project")
    print(f"=== Training {model_key} on {protocol} protocol ===")
    if model_key in ('lcnn', 'pcnn_v2', 'ptcnn', '2dcnn'):
        done = train_cnn_chunked(protocol, model_key)
        if not done:
            print("  >>> NOT YET DONE — re-run this same command again to continue training <<<")
    elif model_key == 'rf':
        train_rf(protocol)
    elif model_key == 'xgb':
        train_xgb(protocol)
    elif model_key == 'ada':
        train_ada(protocol)
    else:
        raise ValueError(f"unknown model_key {model_key}")
