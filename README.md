# Triple-Hybrid ECG Arrhythmia Classification

Research code for five-class ECG beat classification using MIT-BIH-derived signals,
engineered beat features, classical models, neural models, probability ensembles, and
Red Fox Optimization (RFO) for the XGBoost meta-learner.

## Scientific status

The executable pipeline currently uses a **beat-wise, stratified 75:15:10 split**.
This is an intra-patient evaluation protocol: beats from the same record may occur in
train, validation, and test. Current results must therefore not be described as
patient-independent or inter-patient generalization. The record-level split logic in
`src/01_extract_features.py` is retained for investigation but is not the active
protocol until the paper and pipeline are reconciled.

RFO selects eight XGBoost meta-learner hyperparameters using macro-F1 on a fixed
80/20 holdout carved from validation meta-inputs. It then refits on the complete
validation split and evaluates once on test. See [docs/rfo.md](docs/rfo.md).

## Installation on Fedora/Linux

Python 3.10 and 3.11 are supported by the CI matrix. Python 3.11 is the recommended
Fedora/Linux interpreter because the dependency stack, including TensorFlow, is not
currently verified on Python 3.14. GPU support is not claimed by this repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

## Running

The source database files are not part of the intended public release. Obtain the
MIT-BIH Arrhythmia Database from its official PhysioNet source, verify its terms, and
place the required records under `raw_data/mit-bih-arrhythmia-database-1.0.0/`.

```bash
python src/preprocessing_visualization.py --record 100 \
  --output results_beatwise/plots/preprocessing_signal_demo.png
python src/run_full_pipeline.py
```

The full pipeline extracts features, rebuilds the active beat-wise splits, applies
training-only SMOTE, trains base models, builds the ensemble, runs RFO, and writes
generated outputs under `data/` and `results_beatwise/`.

## Repository layout

- `src/`: preprocessing, training, ensemble, optimization, visualization, and reports
- `tests/`: regression and scientific-behavior tests
- `scripts/`: small repository utilities
- `docs/`: methodology, reproducibility, data, and RFO notes
- `data/` and `results_beatwise/`: generated local artifacts, not release source

## Testing

```bash
pytest -q tests
```

## Data and licensing

MIT-BIH data have separate terms and citation requirements from this software. Do not
redistribute downloaded records without verifying the current PhysioNet conditions.
The software license is documented separately in [LICENSE](LICENSE).

## Limitations

The current active evaluation is beat-wise and can be optimistic for patient-level
generalization. Feature extraction also contains simplified fiducial estimation.
Results should be treated as research outputs until the split protocol, paper claims,
and independent rerun have been reconciled.

## Security

Do not commit credentials, private recordings, PowerLab material, virtual environments,
or generated private results. See [SECURITY.md](SECURITY.md).
