#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

DATA_ROOT=${1:?"Usage: $0 DATA_ROOT [BACKBONE]"}
BACKBONE=${2:-"ViT-B/16"}

for DATASET_CONFIG in \
  configs/imagenet.yaml \
  configs/imagenet_a.yaml \
  configs/imagenet_r.yaml \
  configs/imagenetv2.yaml \
  configs/imagenet_sketch.yaml \
  configs/caltech101.yaml \
  configs/dtd.yaml \
  configs/eurosat.yaml \
  configs/fgvc_aircraft.yaml \
  configs/food101.yaml \
  configs/oxford_flowers.yaml \
  configs/oxford_pets.yaml \
  configs/stanford_cars.yaml \
  configs/sun397.yaml \
  configs/ucf101.yaml
do
  python run_final_evaluation.py \
    --data-root "$DATA_ROOT" \
    --dataset-config "$DATASET_CONFIG" \
    --backbone "$BACKBONE"
done
