"""Local dataset loading with this repository's directory conventions."""
from pathlib import Path
from torchvision.datasets import ImageFolder
from data.dataset_registry import DATASET_REGISTRY


def _find_root(data_root, spec, split_override=None):
    base = Path(data_root) / spec.directory
    candidates = [base]
    split = split_override or spec.split
    if split != "auto":
        candidates.insert(0, base / split)
    candidates += [base / "test", base / "val", base / "images"]
    for candidate in candidates:
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"No class-folder dataset found for {spec.id}: {candidates}")


def load_dataset(dataset_id, data_root, transform=None, hf_path=None, hf_split=None,
                 hf_image_column=None, hf_label_column=None, split_override=None):
    try:
        return ImageFolder(_find_root(data_root, DATASET_REGISTRY[dataset_id], split_override), transform=transform)
    except FileNotFoundError as local_error:
        from data.hf_datasets import load_hf_dataset
        try:
            return load_hf_dataset(dataset_id, hf_path, hf_split, hf_image_column, hf_label_column)
        except Exception as hf_error:
            raise FileNotFoundError(
                f"Could not load {dataset_id} locally or from Hugging Face.\n"
                f"Local error: {local_error}\nHF error: {hf_error}"
            ) from hf_error


def class_names(dataset):
    return [name.replace("_", " ").replace("-", " ") for name in dataset.classes]
