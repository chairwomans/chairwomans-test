"""Hugging Face fallback datasets used when a local copy is unavailable."""

from dataclasses import dataclass

from datasets import load_dataset


@dataclass(frozen=True)
class HFDatasetSpec:
    path: str
    split: str
    image_column: str = "image"
    label_column: str = "label"


# Keep these explicit and easy to audit. Additional datasets can be supplied
# from the CLI with --hf-path/--hf-split without changing the pipeline.
HF_DATASETS = {
    "R": HFDatasetSpec("axiong/imagenet-r", "test", "image", "class_name"),
    "K": HFDatasetSpec("vaughankraska/imagenet_sketch", "train", "image", "label"),
}


class HuggingFaceImageDataset:
    """Small ImageFolder-like adapter for a Hugging Face Dataset split."""

    def __init__(self, dataset, image_column, label_column):
        self.dataset = dataset
        self.image_column = image_column
        self.label_column = label_column
        values = [row[label_column] for row in dataset]
        feature = dataset.features.get(label_column)
        if hasattr(feature, "names") and feature.names:
            self.classes = list(feature.names)
            self.targets = [int(value) for value in values]
        else:
            names = sorted({str(value) for value in values})
            self.classes = names
            index = {name: i for i, name in enumerate(names)}
            self.targets = [index[str(value)] for value in values]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        image = row[self.image_column]
        if not hasattr(image, "convert"):
            raise TypeError(f"HF column '{self.image_column}' is not an image")
        return image.convert("RGB"), self.targets[index]


def load_hf_dataset(dataset_id, path=None, split=None, image_column=None, label_column=None):
    spec = HF_DATASETS.get(dataset_id)
    if spec is None and path is None:
        raise FileNotFoundError(
            f"No built-in Hugging Face fallback for {dataset_id}. "
            "Pass --hf-path and --hf-split explicitly."
        )
    path = path or spec.path
    split = split or spec.split
    image_column = image_column or spec.image_column
    label_column = label_column or spec.label_column
    print(f"Local dataset not found; loading Hugging Face dataset '{path}' split='{split}'")
    dataset = load_dataset(path, split=split)
    return HuggingFaceImageDataset(dataset, image_column, label_column)
