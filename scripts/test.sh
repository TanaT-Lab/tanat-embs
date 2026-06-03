#!/usr/bin/env bash
###
# Run pytest for tanat-embeddings
# Usage:
#   bash scripts/test.sh
###
set -euo pipefail

# Get the project root directory
SELF=$(readlink -f "${BASH_SOURCE[0]}")
DIR=${SELF%/*/*}

cd -- "$DIR"

# Setup virtual environment if needed
if [[ ! -e ./venv ]]; then
  echo "Creating virtual environment..."
  bash ./scripts/install_in_venv.sh
fi

# Activate the virtual environment
source ./venv/bin/activate

echo "Running tests..."
pytest -vv --import-mode=importlib --cov=src/tanat_embs --cov-report=xml --cov-report=term tests/
