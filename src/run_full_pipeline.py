import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = BASE
PROC_BEAT = os.path.join(ROOT, 'data', 'processed_beatwise')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')

os.makedirs(PROC_BEAT, exist_ok=True)

def resolve_python_executable():
    candidates = []

    if sys.executable and os.path.exists(sys.executable):
        candidates.append(sys.executable)

    env_override = os.environ.get('ECG_PYTHON')
    if env_override:
        candidates.append(env_override)

    for env_dir in ('.venv', '.venv2'):
        local_env = os.path.join(ROOT, env_dir, 'bin', 'python')
        if os.path.exists(local_env):
            candidates.append(local_env)

    for name in ('python', 'python3'):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)

    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen and os.path.exists(candidate):
            seen.add(candidate)
            return candidate

    raise FileNotFoundError('No usable Python executable was found for the ECG pipeline')


PYTHON = resolve_python_executable()

STEPS = [
    ('Feature extraction', [PYTHON, 'src/01_extract_features.py']),
    ('Rebuild beatwise splits', [PYTHON, 'src/01b_rebuild_splits.py']),
    ('Preprocessing visualization', [PYTHON, 'src/preprocessing_visualization.py', '--record', '100', '--output', 'results_beatwise/plots/preprocessing_signal_demo.png']),
    ('SMOTE balance (train)', [PYTHON, 'src/02b_smote_balance.py', 'beatwise']),
    # Train base models individually to keep resource usage predictable
    ('Train RF (base)', [PYTHON, 'src/03b_train_one_model.py', 'beatwise', 'rf']),
    ('Train XGB (base)', [PYTHON, 'src/03b_train_one_model.py', 'beatwise', 'xgb']),
    ('Train ADA (base)', [PYTHON, 'src/03b_train_one_model.py', 'beatwise', 'ada']),
    ('Ensemble v2', [PYTHON, 'src/04c_ensemble_v2.py', 'beatwise']),
    ('RFO optimization', [PYTHON, 'src/07_red_fox_optimization.py']),
    ('Visualize', [PYTHON, 'src/05b_visualize.py', 'beatwise']),
    ('Generate reports', [PYTHON, 'src/06b_generate_reports.py', 'beatwise']),
]


def set_env_threads():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def run_step(name, cmd):
    print(f"\n=== {name} ===")
    print(' '.join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Step '{name}' failed (exit {proc.returncode})")


def copy_filtered_and_normalized():
    # Source locations produced by scripts
    src_proc = os.path.join(ROOT, 'data', 'processed')
    src_proc_bw = os.path.join(ROOT, 'data', 'processed_beatwise')
    src_splits = os.path.join(ROOT, 'data', 'splits')

    # Copy raw (filtered) splits -> processed_beatwise/filtered_*.csv
    for split in ('train', 'val', 'test'):
        src_raw = os.path.join(src_splits, f'ecg_{split}_raw.csv')
        if os.path.exists(src_raw):
            dst = os.path.join(PROC_BEAT, f'ecg_{split}_filtered.csv')
            shutil.copyfile(src_raw, dst)

    # Copy normalized files if present
    for split in ('train', 'val', 'test'):
        candidates = [
            os.path.join(src_proc_bw, f'ecg_{split}_norm.csv'),
            os.path.join(src_proc, f'ecg_{split}_norm.csv'),
        ]
        for src in candidates:
            if os.path.exists(src):
                dst = os.path.join(PROC_BEAT, f'ecg_{split}_norm.csv')
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copyfile(src, dst)
                break

    # Ensure SMOTE train exists
    smote_src = os.path.join(src_proc_bw, 'ecg_train_smote.csv')
    smote_dst = os.path.join(PROC_BEAT, 'ecg_train_smote.csv')
    if os.path.exists(smote_src) and os.path.abspath(smote_src) != os.path.abspath(smote_dst):
        shutil.copyfile(smote_src, smote_dst)


def write_features_description():
    # Load ALL_FEATURES from 01_extract_features.py by importing the module
    import importlib.util
    spec = importlib.util.spec_from_file_location('featmod', os.path.join(ROOT, 'src', '01_extract_features.py'))
    if spec is None or spec.loader is None:
        raise ImportError('Unable to load feature definitions from 01_extract_features.py')
    featmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(featmod)
    features = getattr(featmod, 'ALL_FEATURES', [])

    # Basic mapping (short descriptions inferred from code names)
    descs = {
        'RR_Interval_ms': 'Previous RR interval in milliseconds',
        'DWT_Total_Energy': 'Total energy of beat from DWT',
        'QRS_Width_ms': 'Estimated QRS complex width (ms)',
        'R_Amplitude_mV': 'R-peak amplitude relative to baseline (mV)',
        'PR_Interval_ms': 'Estimated PR interval duration (ms)',
        'QT_Interval_ms': 'Estimated QT interval duration (ms)',
        'RMSSD_ms': 'Root mean square of successive RR differences (ms)',
        'DWT_Energy_L1': 'DWT detail energy level 1 (highest freq band)',
        'DWT_Energy_L2': 'DWT detail energy level 2',
        'LF_HF_Ratio': 'Ratio of low/high frequency energy bands',
        'LF_Energy': 'Low-frequency band energy (proxy)',
        'ST_Deviation_mV': 'ST segment deviation (mV)',
        'T_Amplitude_mV': 'T-wave amplitude (mV)',
        'P_Width_ms': 'Estimated P-wave width (ms)',
        'DWT_Energy_L3': 'DWT detail energy level 3 (low frequency band)',
        'pNN50_pct': 'Percentage of successive RR diffs >50ms',
        'HF_Energy': 'High-frequency band energy (proxy)',
        'Skewness': 'Signal skewness for beat window',
        'Kurtosis': 'Signal kurtosis for beat window',
        'ZCR': 'Zero-crossing rate normalized by window length',
    }

    out_path = os.path.join(PROC_BEAT, 'features_description.csv')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('feature,description\n')
        for feat in features:
            desc = descs.get(feat, '')
            f.write(f'"{feat}","{desc}"\n')


def main():
    set_env_threads()
    # Stop immediately so a partial run cannot be mistaken for a completed experiment.
    for name, cmd in STEPS:
        run_step(name, cmd)

    # Copy filtered/normalized/smote outputs into processed_beatwise for easy access
    copy_filtered_and_normalized()
    write_features_description()
    print('\nPipeline finished successfully. Outputs under data/processed_beatwise and results_beatwise')


if __name__ == '__main__':
    main()
