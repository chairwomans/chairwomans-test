"""Small TDA-style dataset primitives used by the local test loaders."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Datum:
    impath: str
    label: int
    classname: str


class DatasetBase:
    def __init__(self, test):
        if not test:
            raise ValueError("The official test split is empty.")
        self.test = test


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def listdir_nohidden(path, sort=False):
    items = [entry.name for entry in Path(path).iterdir() if not entry.name.startswith(".")]
    return sorted(items) if sort else items
