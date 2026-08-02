"""
Comprehensive Model Comparison Report
Compare PROPOSED_V3 with all existing models
"""
import os
from pathlib import Path
import json
import numpy as np

BASE = Path(__file__).resolve().parents[2]
MET = BASE / 'results_beatwise' / 'metrics'

# All available models
all_models = [
    'lcnn', 'ada', '2dcnn', 'ptcnn', 'pcnn', 'pcnn_v2', 'xgb', 'rf',
    'dh1', 'dh2', 'dh3', 'dh1v2', 'dh2v2', 'dh3v2',
    'proposed', 'proposed_v2', 'proposed_v3'
]

results = {}
for model_name in all_models:
    metrics_file = MET / f'{model_name}_test_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            results[model_name] = {
                'accuracy': metrics.get('accuracy'),
                'f1_macro': metrics.get('f1_macro'),
                'precision_macro': metrics.get('precision_macro'),
                'recall_macro': metrics.get('recall_macro'),
                'kappa': metrics.get('kappa'),
                'auc_macro': metrics.get('roc_auc_macro'),
            }


def get_model_type(name):
    if 'v2' in name or 'v3' in name:
        if 'proposed' in name:
            return 'Meta-Learner (Optimized)'
        elif 'dh' in name:
            return 'Double Hybrid (v2)'
    if 'proposed' in name:
        return 'Meta-Learner'
    elif 'dh' in name:
        return 'Double Hybrid'
    elif 'cnn' in name or '2dcnn' in name or 'lcnn' in name:
        return 'CNN'
    elif 'xgb' in name:
        return 'XGBoost'
    elif 'rf' in name:
        return 'Random Forest'
    elif 'ada' in name:
        return 'AdaBoost'
    return 'Classifier'


# Sort by F1-Macro
sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)

print("=" * 100)
print("COMPREHENSIVE MODEL RANKING — ECG ARRHYTHMIA CLASSIFICATION (BEAT-WISE)")
print("=" * 100)
print("\nRanking by F1-Macro (Primary Metric):")
print("-" * 100)
print(f"{'Rank':<6} {'Model':<20} {'Type':<25} {'F1-Macro':<12} {'Accuracy':<12} {'Kappa':<12} {'Precision':<12}")
print("-" * 100)

for rank, (model_name, metrics) in enumerate(sorted_results, 1):
    model_type = get_model_type(model_name)
    marker = " ⭐ BEST" if rank == 1 else ""
    is_v3 = " ← V3 OPTIMIZED" if model_name == 'proposed_v3' else ""

    print(f"{rank:<6} {model_name:<20} {model_type:<25} "
          f"{metrics['f1_macro']:.4f}       {metrics['accuracy']:.4f}       "
          f"{metrics['kappa']:.4f}       {metrics['precision_macro']:.4f}{marker}{is_v3}")

print("\n" + "=" * 100)
print("PROPOSED MODEL EVOLUTION")
print("=" * 100)

proposed_versions = {
    'proposed': results.get('proposed', {}),
    'proposed_v2': results.get('proposed_v2', {}),
    'proposed_v3': results.get('proposed_v3', {}),
}

print(f"\n{'Metric':<25} {'PROPOSED v1':<18} {'PROPOSED v2':<18} {'PROPOSED v3':<18} {'Improvement (v1→v3)':<20}")
print("-" * 100)

for metric_name in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'kappa', 'auc_macro']:
    v1 = proposed_versions['proposed'].get(metric_name, 0)
    v2 = proposed_versions['proposed_v2'].get(metric_name, 0)
    v3 = proposed_versions['proposed_v3'].get(metric_name, 0)
    improvement = ((v3 - v1) / v1 * 100) if v1 != 0 else 0

    print(f"{metric_name:<25} {v1:.4f}           {v2:.4f}           {v3:.4f}           {improvement:+.2f}%")

print("\n" + "=" * 100)
print("TOP 3 MODELS DETAILED COMPARISON")
print("=" * 100)

for rank, (model_name, metrics) in enumerate(sorted_results[:3], 1):
    print(f"\n#{rank}: {model_name.upper()}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  F1-Macro:       {metrics['f1_macro']:.4f}")
    print(f"  Precision:      {metrics['precision_macro']:.4f}")
    print(f"  Recall:         {metrics['recall_macro']:.4f}")
    print(f"  Kappa:          {metrics['kappa']:.4f}")
    print(f"  ROC-AUC:        {metrics['auc_macro']:.4f}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

best_model, best_metrics = sorted_results[0]
print(f"\n✓ BEST MODEL: {best_model.upper()}")
print(f"  F1-Macro:        {best_metrics['f1_macro']:.4f}")
print(f"  Accuracy:        {best_metrics['accuracy']:.4f}")
print(f"  Kappa:           {best_metrics['kappa']:.4f}")

if 'proposed_v3' in results:
    v3_metrics = results['proposed_v3']
    v3_rank = next((i + 1 for i, (m, _) in enumerate(sorted_results) if m == 'proposed_v3'), None)
    print(f"\n✓ PROPOSED_V3 RANKING")
    print(f"  Overall Rank:    #{v3_rank} out of {len(sorted_results)} models")
    print(f"  F1-Macro:        {v3_metrics['f1_macro']:.4f}")
    print(f"  Accuracy:        {v3_metrics['accuracy']:.4f}")
    print(f"  Kappa:           {v3_metrics['kappa']:.4f}")

# Save ranking to file
ranking_list = [
    {'rank': rank, 'model': model_name, 'f1_macro': metrics['f1_macro'],
     'accuracy': metrics['accuracy'], 'kappa': metrics['kappa']}
    for rank, (model_name, metrics) in enumerate(sorted_results, 1)
]

with open(MET / 'full_ranking_all_models.json', 'w') as f:
    json.dump(ranking_list, f, indent=2)

print(f"\n✓ Full ranking saved to: {MET / 'full_ranking_all_models.json'}")
