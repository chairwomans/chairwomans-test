import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.final_evaluation import run_domain_hint_templated

NON_IMAGENET_DATASETS = [
    "caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101",
    "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101",
]
IMAGENET_DATASETS = ["imagenet", "imagenetv2", "imagenet_a", "imagenet_r", "imagenet_sketch"]
DEFAULT_DATASETS = NON_IMAGENET_DATASETS + IMAGENET_DATASETS

DATASET_PRESETS = {
    "all": DEFAULT_DATASETS,
    "no-imagenet": NON_IMAGENET_DATASETS,
    "imagenet-only": IMAGENET_DATASETS,
}


def main():
    parser = argparse.ArgumentParser(
        description="Templated domain-hint evaluation: domain phrases combined with each "
                     "dataset's own CLIP prompt templates."
    )
    parser.add_argument("--datasets", default="all",
                        help="Comma-separated cached datasets, or a preset: "
                             + ", ".join(DATASET_PRESETS))
    parser.add_argument("--backbone", default="ViT-B/16",
                        choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate cached raw features without loading CLIP or evaluating.")
    args = parser.parse_args()

    if args.datasets in DATASET_PRESETS:
        datasets = DATASET_PRESETS[args.datasets]
    else:
        datasets = [name.strip() for name in args.datasets.split(",")]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_domain_hint_templated(
        args.backbone, args.data_root, datasets,
        batch_size=args.batch_size, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
