import argparse
import json
from pathlib import Path

from data.dataset_registry import DATASET_REGISTRY
from pipeline.comparison import calibrate_beta


def main():
    parser = argparse.ArgumentParser(description="Calibrate beta on a separate validation/calibration dataset")
    parser.add_argument("--calibration-root", required=True, help="Root containing a train/validation split; never point this at the final test set")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    parser.add_argument("--backbone", required=True, choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"])
    parser.add_argument("--split", default=None, help="Local split folder, e.g. train or val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-grid", default="0.0,0.2,0.35,0.5")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="configs/betas.json")
    parser.add_argument("--hf-path", default=None)
    parser.add_argument("--hf-split", default=None)
    parser.add_argument("--hf-image-column", default=None)
    parser.add_argument("--hf-label-column", default=None)
    args = parser.parse_args()
    grid = tuple(float(x.strip()) for x in args.beta_grid.split(",") if x.strip())
    result = calibrate_beta(
        args.backbone, args.calibration_root, args.dataset, seed=args.seed,
        beta_grid=grid, max_samples=args.max_samples, split_override=args.split,
        hf_path=args.hf_path, hf_split=args.hf_split,
        hf_image_column=args.hf_image_column, hf_label_column=args.hf_label_column,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    values.setdefault(result["dataset"], {})[result["backbone"]] = result
    output.write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved calibrated beta to {output}")


if __name__ == "__main__":
    main()
