#!/usr/bin/env python3
"""Create a release package (zip) of selected artifacts and draft Zenodo metadata.

Usage:
  python3 scripts/prepare_zenodo_package.py --paths data/processed results_beatwise --out release

The script writes a ZIP under `release/` and a metadata draft `docs/zenodo_metadata.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


def gather_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
        else:
            for f in p.rglob("*"):
                if f.is_file():
                    files.append(f)
    return files


def make_zip(files: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_zip = out_dir / f"artifacts_{ts}.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arcname = f.relative_to(Path.cwd())
            z.write(f, arcname)
    return out_zip


def write_metadata(out_dir: Path, zip_path: Path) -> Path:
    meta = {
        "title": "Processed ECG data and model artifacts for Triple-hybrid-model-for-ECG-arrhythmia-classification",
        "creators": [],
        "description": "Processed datasets, scaler parameters, and trained model checkpoints used for reproducing experiments in the associated paper. This deposit DOES NOT include raw patient records; obtain raw MIT-BIH data from PhysioNet.",
        "license": "MIT",
        "keywords": ["ECG", "arrhythmia", "machine learning", "reproducibility"],
        "upload_file": str(zip_path.name),
    }
    out_path = Path("docs") / "zenodo_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="+", default=["data/processed"], help="Paths to include in the package")
    p.add_argument("--out", default="release", help="Output directory for the zip")
    args = p.parse_args()

    paths = [Path(x) for x in args.paths]
    files = gather_files(paths)
    if not files:
        print("No files found to package. Check the provided paths.")
        return 2

    out_dir = Path(args.out)
    zip_path = make_zip(files, out_dir)
    meta_path = write_metadata(out_dir, zip_path)

    print(f"Created package: {zip_path}")
    print(f"Wrote metadata draft: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
