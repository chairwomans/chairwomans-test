"""TDA-style local loaders that expose only each dataset's official test set."""

from pathlib import Path

from .utils import DatasetBase, Datum, listdir_nohidden, read_json


_JSON_DATASETS = {
    "caltech101": ("Caltech101", "caltech-101", "101_ObjectCategories", "split_zhou_Caltech101.json"),
    "dtd": ("DescribableTextures", "dtd", "images", "split_zhou_DescribableTextures.json"),
    "eurosat": ("EuroSAT", "eurosat", "2750", "split_zhou_EuroSAT.json"),
    "food101": ("Food101", "food-101", "images", "split_zhou_Food101.json"),
    "oxford_flowers": ("OxfordFlowers", "oxford_flowers", "jpg", "split_zhou_OxfordFlowers.json"),
    "oxford_pets": ("OxfordPets", "oxford_pets", "images", "split_zhou_OxfordPets.json"),
    "stanford_cars": ("StanfordCars", "stanford_cars", ".", "split_zhou_StanfordCars.json"),
    "sun397": ("SUN397", "sun397", "SUN397", "split_zhou_SUN397.json"),
    "ucf101": ("UCF101", "ucf101", "UCF-101-midframes", "split_zhou_UCF101.json"),
}

_IMAGENET_VARIANTS = {
    "imagenet_a": ("ImageNetA", "imagenet-adversarial", "imagenet-a"),
    "imagenet_r": ("ImageNetR", "imagenet-rendition", "imagenet-r"),
    "imagenet_sketch": ("ImageNetSketch", "imagenet-sketch", "images"),
}


class TestOnlyDataset(DatasetBase):
    def __init__(self, name, test):
        self.name = name
        super().__init__(test=test)


def _read_split(split_path, image_root):
    values = read_json(split_path)
    return [
        Datum(str(Path(image_root) / impath), int(label), classname)
        for impath, label, classname in values["test"]
    ]


def _read_classnames(path):
    names = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        folder, classname = line.split(" ", 1)
        names[folder] = classname
    return names


def _folder_dataset(name, image_root, classnames):
    items = []
    folders = [folder for folder in sorted(classnames) if (Path(image_root) / folder).is_dir()]
    for label, folder in enumerate(folders):
        folder_path = Path(image_root) / folder
        for filename in listdir_nohidden(folder_path, sort=True):
            items.append(Datum(str(folder_path / filename), label, classnames[folder]))
    return TestOnlyDataset(name, items)


def _load_imagenet(root):
    dataset_root = Path(root) / "imagenet"
    names = _read_classnames(dataset_root / "classnames.txt")
    return _folder_dataset("ImageNet", dataset_root / "images" / "val", names)


def _load_imagenetv2(root):
    dataset_root = Path(root) / "imagenetv2"
    names = _read_classnames(dataset_root / "classnames.txt")
    folders = list(names)
    image_root = dataset_root / "imagenetv2-matched-frequency-format-val"
    items = []
    for label, folder in enumerate(folders):
        class_root = image_root / str(label)
        for filename in listdir_nohidden(class_root, sort=True):
            items.append(Datum(str(class_root / filename), label, names[folder]))
    return TestOnlyDataset("ImageNetV2", items)


def _load_imagenet_variant(config_name, root):
    name, dataset_dir, image_dir = _IMAGENET_VARIANTS[config_name]
    dataset_root = Path(root) / dataset_dir
    names = _read_classnames(dataset_root / "classnames.txt")
    return _folder_dataset(name, dataset_root / image_dir, names)


def _load_fgvc(root):
    dataset_root = Path(root) / "fgvc_aircraft"
    classnames = (dataset_root / "variants.txt").read_text(encoding="utf-8").splitlines()
    label_by_name = {name: index for index, name in enumerate(classnames)}
    items = []
    for line in (dataset_root / "images_variant_test.txt").read_text(encoding="utf-8").splitlines():
        image_id, classname = line.split(" ", 1)
        items.append(Datum(str(dataset_root / "images" / f"{image_id}.jpg"), label_by_name[classname], classname))
    return TestOnlyDataset("FGVCAircraft", items)


def build_dataset(config_name, root):
    """Build the fixed official test split for one config filename."""
    if config_name == "imagenet":
        return _load_imagenet(root)
    if config_name == "imagenetv2":
        return _load_imagenetv2(root)
    if config_name in _IMAGENET_VARIANTS:
        return _load_imagenet_variant(config_name, root)
    if config_name == "fgvc_aircraft":
        return _load_fgvc(root)
    try:
        name, dataset_dir, image_dir, split_file = _JSON_DATASETS[config_name]
    except KeyError as error:
        raise KeyError(f"No local test loader for dataset config: {config_name}") from error
    dataset_root = Path(root) / dataset_dir
    return TestOnlyDataset(name, _read_split(dataset_root / split_file, dataset_root / image_dir))
