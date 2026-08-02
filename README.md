# Triple-Hybrid ECG Arrhythmia Classification

This repository contains a complete end-to-end pipeline for ECG arrhythmia classification using the MIT-BIH Arrhythmia Database. The project combines signal preprocessing, beat-wise feature extraction, class balancing, model training, ensemble construction, visualization, and automated reporting in one reproducible workflow.

## What this project does

- Preprocesses raw ECG signals and prepares beat-wise datasets
- Extracts clinically meaningful beat-level features
- Balances class distribution for training data
- Trains and evaluates multiple machine learning and hybrid models
- Generates metrics, plots, and report artifacts for analysis

## Repository structure

- src/ — main pipeline scripts for preprocessing, training, ensembling, plotting, and reporting
- data/ — processed and split datasets used throughout the workflow
- raw_data/ — original MIT-BIH database files used as input data
- results_beatwise/ — generated performance metrics, figures, and reports
- tests/ — regression tests for the main workflow components
- scripts/ — support utilities for data checks and cleanup

## Requirements

The project targets Python 3.10+ and uses the dependencies listed in requirements.txt.

## Quick start

1. Create and activate a Python virtual environment.
2. Install dependencies:
   - `python -m pip install -r requirements.txt`
3. Run the preprocessing demo:
   - `python src/preprocessing_visualization.py --record 100 --output results_beatwise/plots/preprocessing_signal_demo.png`
4. Reproduce the full pipeline:
   - `python src/run_full_pipeline.py`

## Testing

Run the regression tests with:

- `pytest tests`

## Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Security note

This repository does not contain hard-coded secrets or credentials. Keep API keys, access tokens, and local configuration values out of version control by storing them in local environment files or secret stores.
