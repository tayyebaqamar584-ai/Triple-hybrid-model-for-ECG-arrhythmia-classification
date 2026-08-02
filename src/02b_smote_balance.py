"""
Step 2 (parameterized): SMOTE Class Balancing — TRAIN split only, beatwise protocol.
Usage: python 02b_smote_balance.py beatwise
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_FEATURES = [
    'RR_Interval_ms', 'DWT_Total_Energy', 'QRS_Width_ms', 'R_Amplitude_mV',
    'PR_Interval_ms', 'QT_Interval_ms', 'RMSSD_ms', 'DWT_Energy_L1', 'DWT_Energy_L2',
    'LF_HF_Ratio', 'LF_Energy', 'ST_Deviation_mV', 'T_Amplitude_mV', 'P_Width_ms',
    'DWT_Energy_L3', 'pNN50_pct', 'HF_Energy', 'Skewness', 'Kurtosis', 'ZCR'
]
CLASS_NAMES = {0: 'Normal(N)', 1: 'SupraV(S)', 2: 'Ventricular(V)', 3: 'Fusion(F)', 4: 'Paced(Q)'}
MIN_SAMPLES_FOR_SMOTE = 15


def run(protocol):
    proc_dir = os.path.join(BASE, 'data', f'processed_{protocol}')
    print("=" * 70)
    print(f"SMOTE Class Balancing — {protocol.upper()} protocol, TRAIN split only")
    print("=" * 70)

    train_df = pd.read_csv(os.path.join(proc_dir, 'ecg_train_norm.csv'))
    X = train_df[ALL_FEATURES].to_numpy(dtype=np.float32)
    y = train_df['label'].to_numpy(dtype=np.int64)

    print("\nBefore SMOTE:")
    before = Counter(y)
    for c in sorted(before):
        print(f"  {CLASS_NAMES[c]:<16}: {before[c]:>6,}")

    class_counts = Counter(y)
    eligible = {c: n for c, n in class_counts.items() if n >= MIN_SAMPLES_FOR_SMOTE}
    ineligible = {c: n for c, n in class_counts.items() if n < MIN_SAMPLES_FOR_SMOTE}
    if ineligible:
        ineligible_names = {CLASS_NAMES[c]: n for c, n in ineligible.items()}
        print(f"\n[WARNING] Classes with <{MIN_SAMPLES_FOR_SMOTE} real samples NOT oversampled: "
              f"{ineligible_names}")

    majority_count = max(class_counts.values(), default=0)
    majority_class = max(class_counts.items(), key=lambda item: item[1])[0]
    target_per_class = {c: majority_count for c in eligible if c != majority_class}

    min_eligible_minority = min([n for c, n in eligible.items() if c in target_per_class], default=None)
    k_neighbors = max(1, min(5, min_eligible_minority - 1)) if min_eligible_minority else 5
    print(f"\nSMOTE k_neighbors = {k_neighbors}")

    if target_per_class:
        smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
        mask_elig = np.isin(y, list(eligible.keys()))
        X_elig, y_elig = X[mask_elig], y[mask_elig]
        resample_result = smote.fit_resample(X_elig, y_elig)
        X_res, y_res = resample_result[0], resample_result[1]
        if ineligible:
            mask_inelig = np.isin(y, list(ineligible.keys()))
            X_res = np.vstack([np.asarray(X_res, dtype=np.float32), np.asarray(X[mask_inelig], dtype=np.float32)])
            y_res = np.concatenate([np.asarray(y_res, dtype=np.int64), np.asarray(y[mask_inelig], dtype=np.int64)])
    else:
        X_res, y_res = X, y

    print("\nAfter SMOTE:")
    y_res_int = np.asarray(y_res, dtype=np.int64).astype(int).tolist()
    after = Counter(y_res_int)
    for c in sorted(int(k) for k in after):
        print(f"  {CLASS_NAMES[c]:<16}: {after[c]:>6,}")

    smote_df = pd.DataFrame(X_res, columns=ALL_FEATURES)
    smote_df['label'] = y_res.astype(int)
    smote_df = smote_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    smote_df.to_csv(os.path.join(proc_dir, 'ecg_train_smote.csv'), index=False)

    summary = {
        'before': {CLASS_NAMES[c]: int(n) for c, n in before.items()},
        'after': {CLASS_NAMES[c]: int(after[c]) for c in sorted(int(k) for k in after)},
        'ineligible_classes': {CLASS_NAMES[c]: int(n) for c, n in ineligible.items()},
        'k_neighbors_used': k_neighbors,
    }
    with open(os.path.join(proc_dir, 'smote_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved ecg_train_smote.csv ({len(smote_df):,} rows) to {proc_dir}")


if __name__ == '__main__':
    protocol = sys.argv[1] if len(sys.argv) > 1 else 'beatwise'
    if protocol != 'beatwise':
        raise ValueError("Only 'beatwise' protocol is supported in this project")
    run(protocol)
