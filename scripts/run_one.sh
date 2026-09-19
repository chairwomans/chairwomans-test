#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

DATA_ROOT=${1:?"Usage: $0 DATA_ROOT DATASET_CONFIG [BACKBONE]"}
DATASET_CONFIG=${2:?"Usage: $0 DATA_ROOT DATASET_CONFIG [BACKBONE]"}
BACKBONE=${3:-"ViT-B/16"}

python run_final_evaluation.py \
  --data-root "$DATA_ROOT" \
  --dataset-config "$DATASET_CONFIG" \
  --backbone "$BACKBONE"
