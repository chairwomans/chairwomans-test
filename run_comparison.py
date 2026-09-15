import argparse
import json
from pathlib import Path

from data.dataset_registry import DATASET_REGISTRY
from pipeline.comparison import run_comparison


def main():
    parser = argparse.ArgumentParser(description="Run our final local CLIP comparison on one complete test set")
    parser.add_argument("--data-root", required=True, help="Root containing the dataset directories")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY), help="Exactly one dataset ID")
    parser.add_argument("--backbone", default="ViT-B/16", choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", default="auto", help="Numeric beta or auto to read the dataset/backbone calibration result")
    parser.add_argument("--beta-config", default="configs/betas.json")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--hf-path", default=None, help="Optional Hugging Face dataset path used if local data is absent")
    parser.add_argument("--hf-split", default=None, help="Hugging Face split, e.g. test or validation")
    parser.add_argument("--hf-image-column", default=None)
    parser.add_argument("--hf-label-column", default=None)
    args = parser.parse_args()
    if args.beta == "auto":
        config = Path(args.beta_config)
        if not config.exists():
            raise FileNotFoundError(f"No calibrated beta found at {config}. Run calibrate_beta.py first or pass --beta explicitly.")
        values = json.loads(config.read_text(encoding="utf-8"))
        try:
            beta = values[args.dataset][args.backbone]["beta"]
        except KeyError as error:
            raise KeyError(f"No beta for dataset={args.dataset}, backbone={args.backbone}. Run calibrate_beta.py first.") from error
    else:
        beta = float(args.beta)
    print("Selected dataset:", args.dataset, "beta:", beta)
    run_comparison(
        args.backbone, args.data_root, args.dataset, args.seed, beta,
        args.max_test_samples, hf_path=args.hf_path, hf_split=args.hf_split,
        hf_image_column=args.hf_image_column, hf_label_column=args.hf_label_column,
    )


if __name__ == "__main__":
    main()
