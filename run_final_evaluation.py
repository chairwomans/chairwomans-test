import argparse
from utils.final_evaluation import run_final_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run the final CLIP evaluation on an official test split")
    parser.add_argument("--data-root", required=True, help="Root containing the dataset directories")
    parser.add_argument("--dataset-config", required=True, help="Dataset settings YAML, e.g. configs/eurosat.yaml")
    parser.add_argument("--backbone", default="ViT-B/16", choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"])
    args = parser.parse_args()
    print("Selected config:", args.dataset_config)
    run_final_evaluation(
        args.backbone, args.data_root, args.dataset_config,
    )


if __name__ == "__main__":
    main()
