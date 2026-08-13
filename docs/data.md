# Data Policy

The repository uses the MIT-BIH Arrhythmia Database. Download it from the official
PhysioNet source and follow the database license, access terms, and required citation.
Downloaded records, extracted feature tables, normalized CSV files, SMOTE outputs, and
prediction arrays are generated artifacts and should not be committed to the public
source repository.

This project has a separate private PowerLab repository. PowerLab recordings,
ADInstruments exports, private validation data, and related scripts are outside the
scope of this public research implementation.

## Reproducibility and hosting large artifacts

Large artifacts (raw recordings, preprocessed CSVs, model checkpoints, and numpy
arrays) are intentionally excluded from this repository to keep history small and
to comply with data licensing. To reproduce experiments and obtain these artifacts,
use one of the following approaches:

- Host artifacts on Zenodo or Figshare and add DOI links here. Example:
	- `https://zenodo.org/record/<DOI>`
- Host on a private S3 bucket or institutional file server and provide a stable
	download URL (signed URLs for restricted access).
- Provide artifacts as a separate Git LFS or release asset if size and licensing
	permit.

Example download and preparation commands (replace URLs with real links):

```bash
# create a local data folder (keeps repo clean)
mkdir -p data/processed

# download release artifact (example: zipped processed CSVs)
curl -L -o /tmp/processed_data.zip "https://example.org/processed_data.zip"
unzip /tmp/processed_data.zip -d data/processed

# optional: restore scaler params and numpy arrays
python3 scripts/restore_artifacts.py --src data/processed --dest data/processed
```

Guidelines
- Never commit raw datasets, model weights, or large numpy arrays to this repo.
- If you must distribute model files for reviewers, upload them as GitHub Release
	assets or to Zenodo and reference them in `docs/data.md`.
- To reproduce the exact preprocessing used in the paper, run the provided
	scripts in `scripts/` in the order documented in `docs/reproducibility.md`.

If you'd like, I can help upload the current local artifacts to Zenodo and add
the DOI links here.