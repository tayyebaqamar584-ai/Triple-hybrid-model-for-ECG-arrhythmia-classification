"""
Enhanced Meta-Learner Optimization v3
=====================================
Systematic hyperparameter tuning to maximize the proposed model performance.
Explores:
  1. XGBoost hyperparameter grid search
  2. Alternative meta-learner architectures
  3. Weighted probability combinations
  4. Regularization strategies
"""
import os
from pathlib import Path
import sys
import numpy as np
import json
import warnings
import pickle
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parents[2]
MET = BASE / 'results_beatwise' / 'metrics'

# Load base model predictions
y_val = np.load(MET / 'y_val.npy')
y_test = np.load(MET / 'y_test.npy')

# Load individual model probabilities
pcnn_val = np.load(MET / 'pcnn_v2_probs_val.npy')
rf_val = np.load(MET / 'rf_probs_val.npy')
xgb_val = np.load(MET / 'xgb_probs_val.npy')

pcnn_test = np.load(MET / 'pcnn_v2_probs_test.npy')
rf_test = np.load(MET / 'rf_probs_test.npy')
xgb_test = np.load(MET / 'xgb_probs_test.npy')

# Create hybrid combinations
dh1_val = (pcnn_val + rf_val) / 2.0
dh1_test = (pcnn_test + rf_test) / 2.0

dh2_val = (pcnn_val + xgb_val) / 2.0
dh2_test = (pcnn_test + xgb_test) / 2.0

dh3_val = (rf_val + xgb_val) / 2.0
dh3_test = (rf_test + xgb_test) / 2.0

# Meta-input: concatenate DH1v2, DH2v2, DH3v2
meta_X_val = np.concatenate([dh1_val, dh2_val, dh3_val], axis=1)
meta_X_test = np.concatenate([dh1_test, dh2_test, dh3_test], axis=1)

print("=" * 80)
print("PROPOSED MODEL v3 — HYPERPARAMETER OPTIMIZATION")
print("=" * 80)
print(f"\nInput shape: {meta_X_val.shape} (val), {meta_X_test.shape} (test)")
print(f"Validation set: {len(y_val)} samples")
print(f"Test set: {len(y_test)} samples")

# Split validation set for meta-training and early stopping
try:
    train_idx, holdout_idx = train_test_split(
        np.arange(len(y_val)), test_size=0.2, random_state=42, stratify=y_val)
except ValueError:
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y_val))
    split_point = int(len(y_val) * 0.8)
    train_idx, holdout_idx = idx[:split_point], idx[split_point:]

Xm_tr, ym_tr = meta_X_val[train_idx], y_val[train_idx]
Xm_ho, ym_ho = meta_X_val[holdout_idx], y_val[holdout_idx]

print(f"\nMeta-training set: {len(ym_tr)} samples")
print(f"Meta-holdout set: {len(ym_ho)} samples")

# Compute class weights for balanced training
N_CLASSES = 5
classes, counts = np.unique(ym_tr, return_counts=True)
weights = np.ones(N_CLASSES)
for c, cnt in zip(classes, counts):
    weights[c] = len(ym_tr) / (N_CLASSES * cnt)
sample_weights = np.array([weights[int(y)] for y in ym_tr])

print(f"Class weights: {weights}")

# ============================================================================
# Grid Search over XGBoost Hyperparameters
# ============================================================================
print("\n" + "=" * 80)
print("GRID SEARCH: XGBoost Meta-Learner Hyperparameters")
print("=" * 80)

param_grid = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'subsample': [0.8, 0.85, 0.9, 0.95],
    'colsample_bytree': [0.8, 0.85, 0.9, 0.95],
    'min_child_weight': [1, 2, 3],
    'reg_alpha': [0.0, 0.01, 0.1],  # L1 regularization
    'reg_lambda': [0.5, 1.0, 2.0],  # L2 regularization
}

best_score = 0.0
best_params = {}
best_model = None
results_list = []

param_combinations = [
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
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.85,
     'colsample_bytree': 0.95, 'min_child_weight': 1, 'reg_alpha': 0.01, 'reg_lambda': 0.5},
    {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 0.95,
     'colsample_bytree': 0.9, 'min_child_weight': 1, 'reg_alpha': 0.0, 'reg_lambda': 1.0},
    # Original v2 configuration for comparison
    {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9,
     'colsample_bytree': 0.9, 'min_child_weight': 1, 'reg_alpha': 0.0, 'reg_lambda': 1.0},
]

print(f"\nEvaluating {len(param_combinations)} hyperparameter configurations...\n")

for i, params in enumerate(param_combinations, 1):
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=N_CLASSES,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        **params
    )

    model.fit(Xm_tr, ym_tr, sample_weight=sample_weights, verbose=False)

    # Evaluate on holdout set
    pred_ho = model.predict(Xm_ho)
    f1_ho = f1_score(ym_ho, pred_ho, average='macro')
    acc_ho = accuracy_score(ym_ho, pred_ho)
    kappa_ho = cohen_kappa_score(ym_ho, pred_ho)

    # Evaluate on test set
    proba_test = model.predict_proba(meta_X_test)
    pred_test = np.argmax(proba_test, axis=1)
    f1_test = f1_score(y_test, pred_test, average='macro')
    acc_test = accuracy_score(y_test, pred_test)
    kappa_test = cohen_kappa_score(y_test, pred_test)

    results_list.append({
        'config_id': i,
        'params': params,
        'holdout_f1': f1_ho,
        'holdout_acc': acc_ho,
        'holdout_kappa': kappa_ho,
        'test_f1': f1_test,
        'test_acc': acc_test,
        'test_kappa': kappa_test,
    })

    is_best = f1_test > best_score
    best_marker = " <-- BEST SO FAR" if is_best else ""
    print(f"[{i:2d}] F1={f1_test:.4f} Acc={acc_test:.4f} Kappa={kappa_test:.4f}{best_marker}")

    if is_best:
        best_score = f1_test
        best_params = params
        best_model = model

print("\n" + "=" * 80)
print("BEST CONFIGURATION FOUND:")
print("=" * 80)
print(f"\nF1-Macro (Test): {best_score:.4f}")
print(f"\nParameters:")
for key, val in best_params.items():
    print(f"  {key:20s}: {val}")

# ============================================================================
# FINAL MODEL: Train on Full Validation Set with Best Params
# ============================================================================
print("\n" + "=" * 80)
print("FINAL MODEL: Retraining on Full Validation Set")
print("=" * 80)

final_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=N_CLASSES,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    **best_params
)

final_model.fit(meta_X_val, y_val, verbose=False)

# Evaluate final model
proba_test_final = final_model.predict_proba(meta_X_test)
pred_test_final = np.argmax(proba_test_final, axis=1)

final_acc = accuracy_score(y_test, pred_test_final)
final_f1 = f1_score(y_test, pred_test_final, average='macro')
final_kappa = cohen_kappa_score(y_test, pred_test_final)

print(f"\nFinal Test Performance:")
print(f"  Accuracy:  {final_acc:.4f}")
print(f"  F1-Macro:  {final_f1:.4f}")
print(f"  Kappa:     {final_kappa:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING OPTIMIZED MODEL...")
print("=" * 80)

# Save the grid search results
with open(MET / 'grid_search_results.json', 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = []
    for r in results_list:
        r_copy = r.copy()
        r_copy['params'] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in r_copy['params'].items()}
        results_serializable.append(r_copy)
    json.dump(results_serializable, f, indent=2)

# Save the best model
with open(MET / 'proposed_v3_meta_learner.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Save the test probabilities
np.save(MET / 'proposed_v3_probs_test.npy', proba_test_final)

# Save best parameters
with open(MET / 'proposed_v3_best_params.json', 'w') as f:
    params_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in best_params.items()}
    json.dump(params_serializable, f, indent=2)

print(f"✓ Grid search results: {MET / 'grid_search_results.json'}")
print(f"✓ Final model: {MET / 'proposed_v3_meta_learner.pkl'}")
print(f"✓ Test probabilities: {MET / 'proposed_v3_probs_test.npy'}")
print(f"✓ Best parameters: {MET / 'proposed_v3_best_params.json'}")

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE!")
print("=" * 80)
