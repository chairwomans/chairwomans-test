"""Dataset choices for our local experiments.

Class names are read from ImageFolder directories at runtime; this module has
no dependency on the reference MTA implementation.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    directory: str
    description: str
    split: str = "auto"


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "I": DatasetSpec("I", "ImageNet", "ImageNet validation", "val"),
    "A": DatasetSpec("A", "imagenet-a", "ImageNet-A", "images"),
    "R": DatasetSpec("R", "imagenet-r", "ImageNet-R", "images"),
    "V": DatasetSpec("V", "imagenetv2-matched-frequency-format-val", "ImageNet-V2", "images"),
    "K": DatasetSpec("K", "ImageNet-Sketch", "ImageNet-Sketch", "images"),
    "DTD": DatasetSpec("DTD", "DTD", "Describable Textures"),
    "Flower102": DatasetSpec("Flower102", "Flower102", "Oxford Flowers 102"),
    "Food101": DatasetSpec("Food101", "Food101", "Food-101"),
    "Cars": DatasetSpec("Cars", "StanfordCars", "Stanford Cars"),
    "SUN397": DatasetSpec("SUN397", "SUN397", "SUN397"),
    "Aircraft": DatasetSpec("Aircraft", "fgvc_aircraft", "FGVC Aircraft"),
    "Pets": DatasetSpec("Pets", "OxfordPets", "Oxford-IIIT Pets"),
    "Caltech101": DatasetSpec("Caltech101", "Caltech101", "Caltech-101"),
    "UCF101": DatasetSpec("UCF101", "UCF101", "UCF-101"),
    "eurosat": DatasetSpec("eurosat", "eurosat", "EuroSAT"),
}
