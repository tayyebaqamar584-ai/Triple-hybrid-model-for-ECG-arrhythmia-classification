"""
Step 1b: Rebuild Splits — Beat-wise only, 75:15:10
=======================================================================
Reuses the already-extracted real per-beat feature table (data/raw/ecg_all_raw.csv,
109,494 real beats from all 48 MIT-BIH records, computed by 01_extract_features.py) and
derives a single beat-wise 75:15:10 train:val:test split. Split stratification is by
class so every split retains representative proportions of the five rhythm labels.

This is an intra-patient protocol: beats from one record can occur in multiple
splits. It must not be described as patient-independent evaluation. The official
pipeline currently uses this protocol; the record-level outputs from
01_extract_features.py are a separate experimental protocol.
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TABLE = os.path.join(BASE, 'data', 'raw', 'ecg_all_raw.csv')

CLASS_NAMES = {0: 'Normal(N)', 1: 'SupraV(S)', 2: 'Ventricular(V)', 3: 'Fusion(F)', 4: 'Paced(Q)'}
ALL_FEATURES = [
    'RR_Interval_ms', 'DWT_Total_Energy', 'QRS_Width_ms', 'R_Amplitude_mV',
    'PR_Interval_ms', 'QT_Interval_ms', 'RMSSD_ms', 'DWT_Energy_L1', 'DWT_Energy_L2',
    'LF_HF_Ratio', 'LF_Energy', 'ST_Deviation_mV', 'T_Amplitude_mV', 'P_Width_ms',
    'DWT_Energy_L3', 'pNN50_pct', 'HF_Energy', 'Skewness', 'Kurtosis', 'ZCR'
]

def build_beatwise_splits(df, seed=42):
    out_dir = os.path.join(BASE, 'data', 'splits_beatwise')
    proc_dir = os.path.join(BASE, 'data', 'processed_beatwise')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    # First split off TEST (10%), then split remaining 90% into TRAIN/VAL (75/15 of total
    # = 83.33/16.67 of the remaining 90%). Stratify by label throughout.
    train_val_df, test_df = train_test_split(
        df, test_size=0.10, random_state=seed, stratify=df['label'])
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15 / 0.90, random_state=seed, stratify=train_val_df['label'])

    splits = {'train': train_df.reset_index(drop=True), 'val': val_df.reset_index(drop=True),
              'test': test_df.reset_index(drop=True)}

    for name, sdf in splits.items():
        sdf.to_csv(os.path.join(out_dir, f'ecg_{name}_raw.csv'), index=False)

    _fit_and_save_normalisation(splits, proc_dir)
    _print_summary('BEAT-WISE (stratified random, intra-patient)', splits, None)
    _save_summary(splits, proc_dir, None, protocol='beat-wise')
    return splits


def _fit_and_save_normalisation(splits, proc_dir):
    """Z-score scaler fit on TRAIN only, applied to all splits — same discipline as the
    original pipeline, just parameterised by output directory now."""
    train_df = splits['train']
    scaler_params = {}
    for feat in ALL_FEATURES:
        mu = float(train_df[feat].mean())
        sigma = float(train_df[feat].std())
        scaler_params[feat] = {'mean': mu, 'std': sigma if sigma > 1e-9 else 1.0}
    with open(os.path.join(proc_dir, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f, indent=2)

    for name, df in splits.items():
        norm = df.copy()
        for feat in ALL_FEATURES:
            mu, sigma = scaler_params[feat]['mean'], scaler_params[feat]['std']
            norm[feat] = (norm[feat] - mu) / sigma
        norm.to_csv(os.path.join(proc_dir, f'ecg_{name}_norm.csv'), index=False)


def _print_summary(label, splits, records):
    total = sum(len(df) for df in splits.values())
    print(f"\n[{label}]")
    if records:
        for name in ['train', 'val', 'test']:
            print(f"  {name.upper()} records: {sorted(records[name])}")
    print(f"  {'Split':<8} {'Beats':>8} {'%':>7}  {'N':>7} {'S':>6} {'V':>6} {'F':>5} {'Q':>5}")
    for name in ['train', 'val', 'test']:
        df = splits[name]
        cnts = {c: int((df['label'] == c).sum()) for c in range(5)}
        pct = 100 * len(df) / total
        print(f"  {name:<8} {len(df):>8,} {pct:>6.1f}%  {cnts[0]:>7,} {cnts[1]:>6,} "
              f"{cnts[2]:>6,} {cnts[3]:>5,} {cnts[4]:>5,}")


def _save_summary(splits, proc_dir, records, protocol):
    total = sum(len(df) for df in splits.values())
    summary = {
        'protocol': protocol,
        'total_beats': int(total),
        'splits': {name: len(df) for name, df in splits.items()},
        'split_pct': {name: round(100 * len(df) / total, 1) for name, df in splits.items()},
        'class_counts_per_split': {
            name: {CLASS_NAMES[c]: int((df['label'] == c).sum()) for c in range(5)}
            for name, df in splits.items()
        },
    }
    if records:
        summary['records_used'] = records
    with open(os.path.join(proc_dir, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    print("=" * 70)
    print("Rebuilding splits: BEAT-WISE only, 75:15:10")
    print("Reusing existing real feature table (no signal re-extraction needed)")
    print("=" * 70)

    df = pd.read_csv(RAW_TABLE)
    print(f"\nLoaded {len(df):,} real beats from {df['record'].nunique()} records "
        f"(from {RAW_TABLE})")

    build_beatwise_splits(df)

    print("\n[OK] Beat-wise protocol built. val and test are DISTINCT sets.")
    print("     (val for tuning/early stopping, test touched only for final reporting).")


if __name__ == '__main__':
    main()
