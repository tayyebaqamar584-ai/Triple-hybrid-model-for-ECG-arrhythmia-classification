"""
Step 4: Double-Hybrid Ensembles + Proposed Meta-Learner (archived copy)
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              cohen_kappa_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import label_binarize
# TensorFlow-based meta-learner logic omitted in archive copy for brevity

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL = 'beatwise'
MET_DIR = os.path.join(BASE, f'results_{PROTOCOL}', 'metrics')

def load_probs(name, split):
    return np.load(os.path.join(MET_DIR, f'{name}_probs_{split}.npy'))

def main():
    print('Archived ensemble script — kept for reference')

if __name__ == '__main__':
    main()
