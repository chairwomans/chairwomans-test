#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

DATA_ROOT=${1:?"Usage: $0 DATA_ROOT [BACKBONE] [DATASETS]"}
BACKBONE=${2:-"ViT-B/16"}
DATASETS=${3:-"all"}
PYTHON=${PYTHON:-python}

"$PYTHON" run_final_evaluation.py \
  --data-root "$DATA_ROOT" \
  --backbone "$BACKBONE" \
  --datasets "$DATASETS"
