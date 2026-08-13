#!/usr/bin/env bash
# Helper to upload a prepared zip package to Zenodo using the REST API.
# Usage:
#   ZENODO_TOKEN=... PACKAGE=/path/to/artifacts.zip ./scripts/zenodo_upload.sh

set -euo pipefail

if [ -z "${ZENODO_TOKEN:-}" ]; then
  echo "ERROR: Set ZENODO_TOKEN environment variable with your Zenodo API token." >&2
  exit 1
fi

if [ -z "${PACKAGE:-}" ]; then
  echo "ERROR: Set PACKAGE environment variable to the path of the zip to upload." >&2
  exit 1
fi

if [ ! -f "$PACKAGE" ]; then
  echo "ERROR: PACKAGE file not found: $PACKAGE" >&2
  exit 1
fi

echo "Creating a new deposition on zenodo.org..."
RESP=$(curl -sH "Content-Type: application/json" \
  -H "Authorization: Bearer ${ZENODO_TOKEN}" \
  -X POST "https://zenodo.org/api/deposit/depositions" -d '{"metadata": {"title": "Repository artifacts", "upload_type": "dataset"}}')

ID=$(echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo "Created deposition id: $ID"

echo "Uploading file..."
curl -sH "Authorization: Bearer ${ZENODO_TOKEN}" \
  -X PUT "https://zenodo.org/api/deposit/depositions/${ID}/files?filename=$(basename $PACKAGE)" \
  -F "file=@${PACKAGE}" >/dev/null

echo "File uploaded. Finalize the deposition on Zenodo's website to publish or set visibility."
