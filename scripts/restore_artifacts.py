#!/usr/bin/env python3
"""Restore downloaded artifact archives into the repository's data layout.

Usage examples:
  python3 scripts/restore_artifacts.py --src /path/to/downloaded.zip --dest data/processed
  python3 scripts/restore_artifacts.py --src /tmp/processed_data --dest data/processed

The script is intentionally conservative: it will extract common archives
(.zip, .tar.gz, .tgz, .tar) or copy directories, then report which expected
files are present.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

EXPECTED_FILES = [
    "ecg_train_norm.csv",
    "ecg_val_norm.csv",
    "ecg_test_norm.csv",
    "scaler_params.json",
]


def extract_archive(src: Path, dest: Path) -> None:
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(dest)
        return
    try:
        with tarfile.open(src, "r:*") as t:
            t.extractall(dest)
        return
    except tarfile.ReadError:
        pass
    raise RuntimeError(f"Unknown archive format: {src}")


def copy_tree(src: Path, dest: Path) -> None:
    if src.is_dir():
        for item in src.iterdir():
            s = item
            d = dest / item.name
            if item.is_dir():
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        raise RuntimeError(f"Expected directory for copy: {src}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Path to downloaded archive or folder")
    p.add_argument("--dest", default="data/processed", help="Destination folder in repo")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    try:
        if src.is_file():
            extract_archive(src, dest)
        elif src.is_dir():
            copy_tree(src, dest)
        else:
            print(f"Source not found: {src}")
            return 2
    except Exception as e:
        print(f"Failed to restore artifacts: {e}")
        return 3

    print(f"Restored artifacts into: {dest}")
    print("Checking for expected files:")
    for fname in EXPECTED_FILES:
        path = dest / fname
        print(f" - {fname}: {'OK' if path.exists() else 'MISSING'}")

    print("If files are missing, check the downloaded package contents and retry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
