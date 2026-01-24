#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/release.sh [-t|--test-pypi]

Runs release checks, builds artifacts, verifies them, and uploads to PyPI.
Use --test-pypi to upload to TestPyPI instead.
USAGE
}

test_pypi=false

for arg in "$@"; do
  case "$arg" in
    -t|--test-pypi)
      test_pypi=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      usage
      exit 1
      ;;
  esac
done

echo "Running ruff..."
python -m ruff check .

echo "Running pytest..."
python -m pytest

echo "Building artifacts..."
python -m build

echo "Checking artifacts..."
python -m twine check dist/*

if [[ "${test_pypi}" == "true" ]]; then
  echo "Uploading to TestPyPI..."
  python -m twine upload --repository testpypi dist/*
else
  read -r -p "Upload to PyPI (https://pypi.org)? Type 'yes' to continue: " confirm
  if [[ "${confirm}" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
  echo "Uploading to PyPI..."
  python -m twine upload dist/*
fi
