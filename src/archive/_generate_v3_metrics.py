"""
Generate comprehensive metrics for PROPOSED_V3 model
"""
import os
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              cohen_kappa_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import label_binarize

BASE = Path(__file__).resolve().parents[2]
MET = BASE / 'results_beatwise' / 'metrics'

CLASS_NAMES = ['Normal(N)', 'SupraV(S)', 'Ventricular(V)', 'Fusion(F)', 'Paced(Q)']
N_CLASSES = 5

# Load true labels and predictions
y_test = np.load(MET / 'y_test.npy')
proba_test = np.load(MET / 'proposed_v3_probs_test.npy')
pred_test = np.argmax(proba_test, axis=1)

# Compute metrics
acc = float(accuracy_score(y_test, pred_test))
prec = float(precision_score(y_test, pred_test, average='macro', zero_division=0))
rec = float(recall_score(y_test, pred_test, average='macro', zero_division=0))
f1_macro = float(f1_score(y_test, pred_test, average='macro', zero_division=0))
f1_weighted = float(f1_score(y_test, pred_test, average='weighted', zero_division=0))
kappa = float(cohen_kappa_score(y_test, pred_test))

try:
    y_test_bin = np.asarray(label_binarize(y_test, classes=list(range(N_CLASSES))), dtype=np.float64)
    present_mask = np.asarray(y_test_bin.sum(axis=0) > 0)
    auc = float(roc_auc_score(y_test_bin[:, present_mask], proba_test[:, present_mask],
                               average='macro', multi_class='ovr')) if int(present_mask.sum()) >= 2 else None
except Exception as e:
    print(f"AUC calculation error: {e}")
    auc = None

cm = confusion_matrix(y_test, pred_test, labels=list(range(N_CLASSES))).tolist()
report = classification_report(y_test, pred_test, labels=list(range(N_CLASSES)),
                                target_names=CLASS_NAMES, output_dict=True, zero_division=0)

report_dict = report if isinstance(report, dict) else {}
per_class = {name: {'precision': float(report_dict.get(name, {}).get('precision', 0)),
                     'recall': float(report_dict.get(name, {}).get('recall', 0)),
                     'f1': float(report_dict.get(name, {}).get('f1-score', 0)),
                     'support': int(report_dict.get(name, {}).get('support', 0))}
             for name in CLASS_NAMES}

# Create metrics dictionary
metrics = {
    'model_name': 'PROPOSED_V3',
    'accuracy': acc,
    'precision_macro': prec,
    'recall_macro': rec,
    'f1_macro': f1_macro,
    'f1_weighted': f1_weighted,
    'kappa': kappa,
    'roc_auc_macro': auc,
    'per_class': per_class,
    'confusion_matrix': cm,
    'classification_report': report
}

# Save metrics
with open(MET / 'proposed_v3_test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("=" * 80)
print("PROPOSED_V3 COMPREHENSIVE METRICS")
print("=" * 80)
print(f"\nOverall Performance:")
print(f"  Accuracy:           {acc:.4f}")
print(f"  Precision (Macro):  {prec:.4f}")
print(f"  Recall (Macro):     {rec:.4f}")
print(f"  F1-Macro:           {f1_macro:.4f}")
print(f"  F1-Weighted:        {f1_weighted:.4f}")
print(f"  Cohen's Kappa:      {kappa:.4f}")
print(f"  ROC-AUC (Macro):    {auc:.4f}" if auc else "  ROC-AUC (Macro):    N/A")

print(f"\nPer-Class Performance:")
print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 70)
for name in CLASS_NAMES:
    p = per_class[name]['precision']
    r = per_class[name]['recall']
    f = per_class[name]['f1']
    s = per_class[name]['support']
    print(f"{name:<20} {p:<12.4f} {r:<12.4f} {f:<12.4f} {s:<10}")

print(f"\n✓ Full metrics saved to: {MET / 'proposed_v3_test_metrics.json'}")
