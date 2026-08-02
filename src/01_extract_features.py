"""
Step 1: Extract Real Per-Beat Features from Raw MIT-BIH WFDB Signals
=======================================================================
Reads raw .dat/.atr/.hea files (MIT-BIH Arrhythmia Database) and computes
genuine per-beat features directly from the signal — no precomputed/fake data.

Available records (26 of the standard 48 — subset provided):
  100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,
  117,118,119,121,122,123,124,200,201,202

Records 102, 104, 107 are excluded (paced-rhythm records, standard practice
in AAMI-protocol ECG classification studies, e.g. de Chazal et al. 2004).

Inter-patient split (AAMI/ANSI EC57 protocol, de Chazal DS1/DS2), restricted
to records actually available in this dataset:
  DS1 (available): 101,106,108,109,112,114,115,116,118,119,122,124,201
  DS2 (available): 100,103,105,111,113,117,121,123,200,202

DS1 is further split by patient into TRAIN / VAL. DS2 is held out as TEST.
No patient's beats ever appear in more than one split.

5-Class AAMI mapping:
  N (Normal)     <- N, L, R, e, j
  S (SupraV)     <- A, a, J, S
  V (Ventricular)<- V, E
  F (Fusion)     <- F
  Q (Unknown/paced)<- /, f, Q
Non-beat annotations (+, ~, |, x, etc.) are discarded.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import wfdb
import pywt
from scipy import stats

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_SIGNAL_DIR = os.path.join(BASE, 'raw_data', 'mit-bih-arrhythmia-database-1.0.0')
SPLITS_DIR = os.path.join(BASE, 'data', 'splits')
PROC_DIR   = os.path.join(BASE, 'data', 'processed')
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

FS = 360  # MIT-BIH sampling rate (Hz)

# AAMI EC57 5-class mapping
SYMBOL_TO_CLASS = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,         # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                  # Supraventricular
    'V': 2, 'E': 2,                                  # Ventricular
    'F': 3,                                          # Fusion
    '/': 4, 'f': 4, 'Q': 4,                          # Unknown/paced
}
CLASS_NAMES = {0: 'Normal(N)', 1: 'SupraV(S)', 2: 'Ventricular(V)', 3: 'Fusion(F)', 4: 'Paced(Q)'}

# Standard AAMI inter-patient split (de Chazal et al. 2004) as a starting point, FULL
# 48-record database. Per explicit project decision, this is a genuine 5-class AAMI task
# (N/S/V/F/Q), so 102/104/107/217 are kept rather than excluded as in the de Chazal
# convention. However, nearly ALL real Paced(Q) beats in the entire database live inside
# exactly these 4 records (102, 104, 107, 217 — combined >99% of all Q-class beats; no
# other record has more than a handful). The standard DS1/DS2 assignment puts all 4 on the
# DS1 (train-pool) side, which would leave the TEST set with ~7 real Q beats — making 5-class
# TEST evaluation on Q meaningless. To report real, defensible 5-class metrics on a genuinely
# held-out test set, we deliberately move ONE paced record into DS2 (test) and split the
# remaining 3 between TRAIN and VAL, so all three splits contain real Q-class examples. This
# is a stated, deliberate deviation from the literal de Chazal DS1/DS2 list, made necessary by
# this project's explicit choice to keep the 5-class (not 4-class) formulation.
DS1_BASE = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201,
            203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_BASE = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
            210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# Paced-bearing records distributed: 104 -> stays DS1/train pool, 217 -> DS2/test,
# 102 & 107 -> DS1/train pool (one of which is reserved for VAL below).
DS1 = sorted(DS1_BASE + [102, 104, 107])
DS2 = sorted(DS2_BASE + [217])

AVAILABLE = sorted(DS1 + DS2)
assert len(AVAILABLE) == 48, f"expected 48 records, got {len(AVAILABLE)}"
assert sorted(set(AVAILABLE)) == AVAILABLE, "duplicate record assignment detected"

# DS1 patient-level split into TRAIN (~73%) / VAL (~27%). 102 (paced-heavy) is held out for
# VAL so VAL also contains real Q-class examples; 104/107 (also paced-heavy) stay in TRAIN.
DS1_VAL_RECORDS = [102, 116, 119, 124, 205, 220]
DS1_TRAIN_RECORDS = [r for r in DS1 if r not in DS1_VAL_RECORDS]
DS2_TEST_RECORDS = DS2


def wavelet_features(beat_signal):
    """3-level DWT decomposition; returns energy of each band + total."""
    coeffs = pywt.wavedec(beat_signal, 'db4', level=3)
    # coeffs = [cA3, cD3, cD2, cD1]
    energies = [float(np.sum(np.square(c))) for c in coeffs]
    total = float(np.sum(np.square(beat_signal)))
    return {
        'DWT_Energy_L1': energies[3],  # cD1 (highest freq detail)
        'DWT_Energy_L2': energies[2],  # cD2
        'DWT_Energy_L3': energies[1],  # cD3
        'DWT_Total_Energy': total,
    }


def freq_band_energy(beat_signal, fs=FS):
    """LF (0.04-0.15Hz proxy via beat-scale) / HF energy ratio using Welch PSD
    on a windowed beat segment. For single-beat morphology we approximate using
    intra-beat spectral energy split (low vs high half of beat-relevant band)."""
    if len(beat_signal) < 8:
        return {'LF_Energy': 0.0, 'HF_Energy': 0.0, 'LF_HF_Ratio': 0.0}
    freqs = np.fft.rfftfreq(len(beat_signal), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(beat_signal)) ** 2
    lf_mask = (freqs >= 0.5) & (freqs < 10)
    hf_mask = (freqs >= 10) & (freqs < 40)
    lf_energy = float(np.sum(spectrum[lf_mask]))
    hf_energy = float(np.sum(spectrum[hf_mask]))
    ratio = lf_energy / hf_energy if hf_energy > 1e-9 else 0.0
    return {'LF_Energy': lf_energy, 'HF_Energy': hf_energy, 'LF_HF_Ratio': ratio}


def extract_beat_window(signal, r_idx, pre=99, post=160):
    """Extract a window around the R-peak. Default ~0.72s window at 360Hz
    (99 samples before, 160 after), standard for MIT-BIH beat segmentation."""
    start = max(0, r_idx - pre)
    end = min(len(signal), r_idx + post)
    return signal[start:end]


def detect_fiducials(beat_signal, r_local_idx, fs=FS):
    """Lightweight fiducial point estimation within a beat window for
    QRS width, P/T wave amplitudes, and interval estimation.
    This is a simplified rule-based detector (not a full Pan-Tompkins QRS
    delineator) operating on a pre-segmented beat window centered on R."""
    n = len(beat_signal)
    r_amp = float(beat_signal[r_local_idx]) if 0 <= r_local_idx < n else float(np.max(beat_signal))

    # QRS onset/offset: scan outward from R until signal returns near baseline
    baseline = float(np.median(beat_signal))
    thresh = 0.1 * abs(r_amp - baseline) if abs(r_amp - baseline) > 1e-6 else 0.05

    q_idx = r_local_idx
    while q_idx > 0 and abs(beat_signal[q_idx] - baseline) > thresh:
        q_idx -= 1
    s_idx = r_local_idx
    while s_idx < n - 1 and abs(beat_signal[s_idx] - baseline) > thresh:
        s_idx += 1
    qrs_width_ms = (s_idx - q_idx) / fs * 1000.0

    # P wave: search window before QRS onset
    p_search_start = max(0, q_idx - int(0.20 * fs))
    p_region = beat_signal[p_search_start:q_idx] if q_idx > p_search_start else np.array([baseline])
    p_amp = float(np.max(p_region) - baseline) if len(p_region) else 0.0
    p_width_ms = len(p_region) / fs * 1000.0 * 0.4  # approximate active P duration fraction

    # T wave: search window after QRS offset
    t_search_end = min(n, s_idx + int(0.40 * fs))
    t_region = beat_signal[s_idx:t_search_end] if t_search_end > s_idx else np.array([baseline])
    t_amp = float(np.max(t_region) - baseline) if len(t_region) else 0.0

    # ST deviation: signal level ~80ms after S point vs baseline
    st_idx = min(n - 1, s_idx + int(0.08 * fs))
    st_dev = float(beat_signal[st_idx] - baseline)

    pr_interval_ms = (q_idx - p_search_start) / fs * 1000.0 if q_idx > p_search_start else 0.0
    qt_interval_ms = (t_search_end - q_idx) / fs * 1000.0

    return {
        'QRS_Width_ms': qrs_width_ms,
        'R_Amplitude_mV': r_amp - baseline,
        'P_Width_ms': p_width_ms,
        'T_Amplitude_mV': t_amp,
        'ST_Deviation_mV': st_dev,
        'PR_Interval_ms': pr_interval_ms,
        'QT_Interval_ms': qt_interval_ms,
    }


def process_record(record_id, signal_dir=RAW_SIGNAL_DIR):
    """Extract all valid beats + features from a single record."""
    path = os.path.join(signal_dir, str(record_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, 'atr')

    rec_any = rec
    p_signal = getattr(rec_any, 'p_signal', None)
    if p_signal is None:
        raise ValueError(f"Record {record_id} did not contain any signal data")
    sig = np.asarray(p_signal, dtype=np.float64)[:, 0]  # lead 0 (MLII for all these records)
    fs = int(getattr(rec_any, 'fs', FS) or FS)

    samples = np.asarray(getattr(ann, 'sample', []), dtype=np.int64)
    symbols = list(getattr(ann, 'symbol', []) or [])

    rows = []
    valid_idx = [i for i, s in enumerate(symbols) if s in SYMBOL_TO_CLASS]

    for pos, i in enumerate(valid_idx):
        r_sample = samples[i]
        label = SYMBOL_TO_CLASS[symbols[i]]

        # RR interval (to previous beat of ANY type, in ms) — needs raw sample index, not filtered
        # find previous annotation sample overall (any symbol) for true RR
        if i > 0:
            rr_prev_ms = (samples[i] - samples[i - 1]) / fs * 1000.0
        else:
            rr_prev_ms = np.nan

        beat_win = extract_beat_window(sig, r_sample, pre=99, post=160)
        r_local = min(99, r_sample)  # local index of R within window (clamped at signal start)

        fiducials = detect_fiducials(beat_win, r_local, fs=fs)
        wave_feats = wavelet_features(beat_win)
        freq_feats = freq_band_energy(beat_win, fs=fs)

        skew = float(stats.skew(beat_win)) if len(beat_win) > 2 else 0.0
        kurt = float(stats.kurtosis(beat_win)) if len(beat_win) > 2 else 0.0
        zero_crossings = int(np.sum(np.diff(np.sign(beat_win - np.median(beat_win))) != 0))
        zcr = zero_crossings / len(beat_win) if len(beat_win) else 0.0

        rows.append({
            'record': record_id,
            'label': label,
            'RR_Interval_ms': rr_prev_ms,
            **fiducials,
            **wave_feats,
            **freq_feats,
            'Skewness': skew,
            'Kurtosis': kurt,
            'ZCR': zcr,
        })

    df = pd.DataFrame(rows)

    # RMSSD and pNN50: computed per-record over consecutive RR intervals (any beat type)
    rr_all_ms = np.diff(samples) / fs * 1000.0
    if len(rr_all_ms) > 1:
        diffs = np.diff(rr_all_ms)
        rmssd = float(np.sqrt(np.mean(diffs ** 2)))
        pnn50 = float(np.mean(np.abs(diffs) > 50) * 100.0)
    else:
        rmssd, pnn50 = 0.0, 0.0
    df['RMSSD_ms'] = rmssd
    df['pNN50_pct'] = pnn50

    # First beat in record has no valid RR_Interval -> drop
    df = df.dropna(subset=['RR_Interval_ms']).reset_index(drop=True)
    return df


ALL_FEATURES = [
    'RR_Interval_ms', 'DWT_Total_Energy', 'QRS_Width_ms', 'R_Amplitude_mV',
    'PR_Interval_ms', 'QT_Interval_ms', 'RMSSD_ms', 'DWT_Energy_L1', 'DWT_Energy_L2',
    'LF_HF_Ratio', 'LF_Energy', 'ST_Deviation_mV', 'T_Amplitude_mV', 'P_Width_ms',
    'DWT_Energy_L3', 'pNN50_pct', 'HF_Energy', 'Skewness', 'Kurtosis', 'ZCR'
]


def main():
    print("=" * 70)
    print("ECG Feature Extraction — REAL signals, REAL annotations")
    print("MIT-BIH Arrhythmia Database | Inter-Patient AAMI DS1/DS2 Split")
    print("=" * 70)
    print(f"\nDS1 (train pool): {DS1}")
    print(f"  -> TRAIN records: {DS1_TRAIN_RECORDS}")
    print(f"  -> VAL records:   {DS1_VAL_RECORDS}")
    print(f"DS2 (test, held out): {DS2_TEST_RECORDS}")
    print("Note: full 5-class task — 102/104/107/217 (paced-heavy) deliberately distributed")
    print("across TRAIN/VAL/TEST (one in VAL, one in TEST, two in TRAIN) so all splits carry")
    print("real Paced(Q) examples. See in-code comment for rationale.")

    splits = {
        'train': DS1_TRAIN_RECORDS,
        'val': DS1_VAL_RECORDS,
        'test': DS2_TEST_RECORDS,
    }

    split_dfs = {}
    for split_name, records in splits.items():
        print(f"\n[{split_name.upper()}] Processing {len(records)} records...")
        dfs = []
        for r in records:
            df_r = process_record(r)
            dfs.append(df_r)
            counts = df_r['label'].value_counts().sort_index().to_dict()
            print(f"  Record {r}: {len(df_r):,} beats | {counts}")
        split_df = pd.concat(dfs, ignore_index=True)
        split_dfs[split_name] = split_df
        print(f"  {split_name.upper()} TOTAL: {len(split_df):,} beats")

    # Save raw (un-normalized) splits
    for name, df in split_dfs.items():
        df.to_csv(os.path.join(SPLITS_DIR, f'ecg_{name}_raw.csv'), index=False)

    all_raw = pd.concat(split_dfs.values(), ignore_index=True)
    all_raw.to_csv(os.path.join(BASE, 'data', 'raw', 'ecg_all_raw.csv'), index=False)

    # Fit z-score scaler on TRAIN ONLY
    train_df = split_dfs['train']
    scaler_params = {}
    for feat in ALL_FEATURES:
        mu = float(train_df[feat].mean())
        sigma = float(train_df[feat].std())
        scaler_params[feat] = {'mean': mu, 'std': sigma if sigma > 1e-9 else 1.0}

    with open(os.path.join(PROC_DIR, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f, indent=2)

    def apply_zscore(df):
        out = df.copy()
        for feat in ALL_FEATURES:
            mu, sigma = scaler_params[feat]['mean'], scaler_params[feat]['std']
            out[feat] = (out[feat] - mu) / sigma
        return out

    print("\n[Normalisation] Z-score fit on TRAIN only, applied to VAL/TEST via transform")
    for name, df in split_dfs.items():
        norm = apply_zscore(df)
        norm.to_csv(os.path.join(PROC_DIR, f'ecg_{name}_norm.csv'), index=False)

    # Dataset summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY (REAL, extracted from raw signals)")
    print("=" * 70)
    print(f"{'Split':<10} {'Beats':>8}  {'N':>7}  {'S':>7}  {'V':>7}  {'F':>7}  {'Q':>7}")
    print("-" * 60)
    totals = {c: 0 for c in range(5)}
    for name, df in split_dfs.items():
        cnts = {c: int((df['label'] == c).sum()) for c in range(5)}
        for c in range(5):
            totals[c] += cnts[c]
        print(f"{name:<10} {len(df):>8,}  {cnts[0]:>7,}  {cnts[1]:>7,}  {cnts[2]:>7,}  {cnts[3]:>7,}  {cnts[4]:>7,}")
    total_beats = sum(totals.values())
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_beats:>8,}  {totals[0]:>7,}  {totals[1]:>7,}  {totals[2]:>7,}  {totals[3]:>7,}  {totals[4]:>7,}")

    summary = {
        'total_beats': int(total_beats),
        'class_counts': {CLASS_NAMES[c]: int(totals[c]) for c in range(5)},
        'splits': {name: len(df) for name, df in split_dfs.items()},
        'records_used': {'train': DS1_TRAIN_RECORDS, 'val': DS1_VAL_RECORDS, 'test': DS2_TEST_RECORDS},
        'paced_heavy_records_distribution': {'train': [104, 107], 'val': [102], 'test': [217]},
        'note': 'Full 48-record MIT-BIH database. 5-class AAMI task (N/S/V/F/Q), all records '
                'retained per project decision. Paced-heavy records (102/104/107/217) deliberately '
                'distributed across all three splits so each split has real Paced(Q) examples.',
    }
    with open(os.path.join(PROC_DIR, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Step 1 complete. Real features extracted from raw WFDB signals.")


if __name__ == '__main__':
    main()
