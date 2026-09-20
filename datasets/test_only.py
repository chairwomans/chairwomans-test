"""TDA-style local loaders for the fixed dataset splits used by this project."""

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


class LocalDataset(DatasetBase):
    def __init__(self, name, test, val=None, train=None):
        self.name = name
        super().__init__(test=test, val=val, train=train)


def _read_split(split_path, image_root, split):
    values = read_json(split_path)
    return [
        Datum(str(Path(image_root) / impath), int(label), classname)
        for impath, label, classname in values[split]
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
    return LocalDataset(name, items)


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
    return LocalDataset("ImageNetV2", items)


def _load_imagenet_variant(config_name, root):
    name, dataset_dir, image_dir = _IMAGENET_VARIANTS[config_name]
    dataset_root = Path(root) / dataset_dir
    names = _read_classnames(dataset_root / "classnames.txt")
    return _folder_dataset(name, dataset_root / image_dir, names)


def _read_fgvc_split(dataset_root, split, label_by_name):
    items = []
    for line in (dataset_root / f"images_variant_{split}.txt").read_text(encoding="utf-8").splitlines():
        image_id, classname = line.split(" ", 1)
        items.append(Datum(str(dataset_root / "images" / f"{image_id}.jpg"), label_by_name[classname], classname))
    return items


def _load_fgvc(root):
    dataset_root = Path(root) / "fgvc_aircraft"
    classnames = (dataset_root / "variants.txt").read_text(encoding="utf-8").splitlines()
    label_by_name = {name: index for index, name in enumerate(classnames)}
    return LocalDataset(
        "FGVCAircraft",
        test=_read_fgvc_split(dataset_root, "test", label_by_name),
        val=_read_fgvc_split(dataset_root, "val", label_by_name),
        train=_read_fgvc_split(dataset_root, "train", label_by_name),
    )


def build_dataset(config_name, root):
    """Build fixed splits for one dataset config filename."""
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
    split_path = dataset_root / split_file
    return LocalDataset(
        name,
        test=_read_split(split_path, dataset_root / image_dir, "test"),
        val=_read_split(split_path, dataset_root / image_dir, "val"),
        train=_read_split(split_path, dataset_root / image_dir, "train"),
    )
