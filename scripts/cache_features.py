"""Create deterministic CLIP image-feature caches for one dataset split."""

import argparse
import json
from pathlib import Path
import sys

import clip
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import build_dataset


def _load_images(items, indices):
    images = []
    for index in indices:
        with Image.open(items[index].impath) as image:
            images.append(image.convert("RGB").copy())
    return images


def _encode(model, preprocess, images, device):
    batch = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.inference_mode():
        return F.normalize(model.encode_image(batch).float(), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, help="Dataset config stem, e.g. eurosat")
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; refusing to create a CPU cache.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "RTX 3070" not in gpu_name.upper():
        raise RuntimeError(f"RTX 3070 GPU required, but cuda:0 is {gpu_name!r}.")

    dataset = build_dataset(args.dataset, args.data_root)
    items = getattr(dataset, args.split)
    if not items:
        raise ValueError(f"{args.dataset} has an empty {args.split} split")

    cache_dir = args.data_root / "cache" / args.backbone.replace("/", "-") / args.dataset
    if args.split != "test":
        cache_dir = cache_dir / args.split
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = [cache_dir / "raw.pt", cache_dir / "avg.pt", cache_dir / "metadata.json"]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Cache already exists in {cache_dir}; use --overwrite to replace it.")

    model_cache = args.data_root / "cache" / "clip_models"
    model, preprocess = clip.load(
        args.backbone, device=device, download_root=str(model_cache)
    )
    model.eval()
    raw, avg = [], []
    for start in tqdm(range(0, len(items), args.batch_size), desc=f"Caching {dataset.name}"):
        indices = range(start, min(start + args.batch_size, len(items)))
        images = _load_images(items, indices)
        original = _encode(model, preprocess, images, device)
        mirrored = _encode(model, preprocess, [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images], device)
        raw.append(original.cpu().half())
        avg.append(F.normalize((original + mirrored) / 2, dim=-1).cpu().half())

    torch.save(torch.cat(raw), cache_dir / "raw.pt")
    torch.save(torch.cat(avg), cache_dir / "avg.pt")
    metadata = {
        "dataset": args.dataset,
        "dataset_name": dataset.name,
        "split": args.split,
        "backbone": args.backbone,
        "device": gpu_name,
        "feature_dtype": "float16",
        "feature_count": len(items),
        "feature_dimension": raw[0].shape[1],
        "raw_definition": "normalize(encode_image(original))",
        "avg_definition": "normalize((raw + normalize(encode_image(horizontal_flip(original)))) / 2)",
        "items": [
            {"path": str(Path(item.impath).relative_to(args.data_root)), "label": item.label, "classname": item.classname}
            for item in items
        ],
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created cache: {cache_dir}")


if __name__ == "__main__":
    main()
