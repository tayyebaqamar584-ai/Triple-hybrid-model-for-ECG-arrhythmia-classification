import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

BASE = Path(__file__).resolve().parents[1]
MET = BASE / 'results_beatwise' / 'metrics'
y_val = np.load(MET / 'y_val.npy')
y_test = np.load(MET / 'y_test.npy')

pcnn_val = np.load(MET / 'pcnn_v2_probs_val.npy')
rf_val = np.load(MET / 'rf_probs_val.npy')
xgb_val = np.load(MET / 'xgb_probs_val.npy')

pcnn_test = np.load(MET / 'pcnn_v2_probs_test.npy')
rf_test = np.load(MET / 'rf_probs_test.npy')
xgb_test = np.load(MET / 'xgb_probs_test.npy')

meta_val = np.concatenate([pcnn_val, rf_val, xgb_val], axis=1)
meta_test = np.concatenate([pcnn_test, rf_test, xgb_test], axis=1)

candidates = []

# simple weighted averages
for w1 in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    for w2 in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        if abs(w1 + w2) > 2.0:
            continue
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
        pred_val = (w1 * pcnn_val + w2 * rf_val + w3 * xgb_val).argmax(axis=1)
        acc = accuracy_score(y_val, pred_val)
        f1 = f1_score(y_val, pred_val, average='macro')
        kappa = cohen_kappa_score(y_val, pred_val)
        if acc > 0.99 or f1 > 0.97:
            candidates.append((acc, f1, kappa, (w1, w2, w3)))

print('Top weighted averages on VAL:')
for item in sorted(candidates, reverse=True)[:20]:
    print(item)

models = [
    (
        'lr',
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                multi_class='multinomial', max_iter=3000, C=2.0, solver='lbfgs'
            ),
        ),
    ),
    (
        'lr_bal',
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                multi_class='multinomial',
                max_iter=3000,
                C=2.0,
                class_weight='balanced',
                solver='lbfgs',
            ),
        ),
    ),
    (
        'rf_meta',
        RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1),
    ),
    (
        'et_meta',
        ExtraTreesClassifier(n_estimators=600, max_depth=None, random_state=42, n_jobs=-1),
    ),
    (
        'mlp',
        make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=4000, random_state=42, early_stopping=True
            ),
        ),
    ),
    (
        'xgb_meta',
        xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=5,
            eval_metric='mlogloss',
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        ),
    ),
]

print('\nMeta-models on VAL:')
for name, model in models:
    model.fit(meta_val, y_val)
    pred_val = model.predict(meta_val)
    acc = accuracy_score(y_val, pred_val)
    f1 = f1_score(y_val, pred_val, average='macro')
    kappa = cohen_kappa_score(y_val, pred_val)
    pred_test = model.predict(meta_test)
    test_acc = accuracy_score(y_test, pred_test)
    test_f1 = f1_score(y_test, pred_test, average='macro')
    test_kappa = cohen_kappa_score(y_test, pred_test)
    print(
        name,
        'val',
        round(float(acc), 4),
        round(float(f1), 4),
        round(float(kappa), 4),
        'test',
        round(float(test_acc), 4),
        round(float(test_f1), 4),
        round(float(test_kappa), 4),
    )
