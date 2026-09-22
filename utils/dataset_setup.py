"""Dataset and pipeline-setting loaders."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from datasets import build_dataset


_REQUIRED_SPLIT_FILES = {
    "caltech101": "caltech-101/split_zhou_Caltech101.json",
    "dtd": "dtd/split_zhou_DescribableTextures.json",
    "eurosat": "eurosat/split_zhou_EuroSAT.json",
    "food101": "food-101/split_zhou_Food101.json",
    "oxford_flowers": "oxford_flowers/split_zhou_OxfordFlowers.json",
    "oxford_pets": "oxford_pets/split_zhou_OxfordPets.json",
    "stanford_cars": "stanford_cars/split_zhou_StanfordCars.json",
    "sun397": "sun397/split_zhou_SUN397.json",
    "ucf101": "ucf101/split_zhou_UCF101.json",
}


@dataclass(frozen=True)
class PipelineSettings:
    n_clusters: int | None
    beta: float


def _load_settings(config_path):
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    recenter = values.get("recenter", {})
    n_clusters = recenter.get("K")
    if n_clusters is not None and (not isinstance(n_clusters, int) or n_clusters < 1):
        raise ValueError(f"recenter.K must be a positive integer or null: {config_path}")
    beta = float(recenter.get("beta", 0.0))
    return PipelineSettings(n_clusters=n_clusters, beta=beta)


def load_evaluation_dataset(data_root, dataset_config):
    """Build one dataset's official test split and read its fixed settings."""
    config_path = Path(dataset_config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    root = Path(data_root).expanduser()
    split_file = _REQUIRED_SPLIT_FILES.get(config_path.stem)
    if split_file and not (root / split_file).is_file():
        raise FileNotFoundError(
            f"Required official split file is missing: {root / split_file}. "
            "Prepare the dataset exactly as specified in DATASETS.md."
        )

    dataset = build_dataset(config_path.stem, root)
    return dataset.name, dataset, _load_settings(config_path)
