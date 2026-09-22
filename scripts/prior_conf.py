import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clip
from utils.dataset_setup import _load_settings
from utils.final_evaluation import _cache_directory, _dataset_templates, _fit_recenter, _load_cached_avg, _text_proto


DATASETS = [
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "food101",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
    "imagenet",
    "imagenetv2",
    "imagenet_a",
    "imagenet_r",
    "imagenet_sketch",
]


def load_cache_items(data_root, backbone, dataset, split):
    metadata_path = _cache_directory(data_root, backbone, dataset, split) / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {metadata_path.parent}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    items = [
        SimpleNamespace(
            impath=str(Path(data_root) / row["path"]),
            label=row["label"],
            classname=row["classname"],
        )
        for row in metadata["items"]
    ]

    return metadata["dataset_name"], items


def _load_raw(data_root, backbone, dataset, split, device):
    cache_dir = _cache_directory(data_root, backbone, dataset, split)
    raw_path = cache_dir / "raw.pt"
    metadata_path = cache_dir / "metadata.json"

    if not raw_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{split} feature cache not found: {cache_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("backbone") != backbone:
        raise ValueError(
            f"Cache backbone mismatch in {cache_dir}: "
            f"expected {backbone}, found {metadata.get('backbone')}"
        )

    raw = torch.load(raw_path, map_location=device, weights_only=True).float()

    if raw.ndim != 2 or raw.shape[0] != len(metadata["items"]):
        raise ValueError(f"Invalid cached raw feature shape {tuple(raw.shape)} in {raw_path}")

    return F.normalize(raw, dim=-1)


def load_fit_embeds(data_root, backbone, dataset, device):
    val_dir = _cache_directory(data_root, backbone, dataset, "val")

    if val_dir.is_dir():
        return _load_raw(data_root, backbone, dataset, "val", device), "val"

    return None, "test"


def build_conf_prior(logits):
    prob = F.softmax(logits, dim=-1)
    num_classes = prob.shape[1]

    entropy = -(prob * torch.log(prob + 1e-12)).sum(dim=-1)
    confidence = 1.0 - entropy / np.log(num_classes)

    prior = (confidence.unsqueeze(1) * prob).sum(dim=0)
    prior = prior / (confidence.sum() + 1e-12)

    prior = prior.clamp_min(1e-12)
    prior = prior / prior.sum()

    correction = -torch.log(prior + 1e-12)
    correction = correction - correction.mean()

    return prior, correction


def accuracy(logits, labels):
    pred = logits.argmax(dim=-1)
    acc = (pred == labels).float().mean().item() * 100.0
    return pred, acc


@torch.no_grad()
def run_dataset(model, data_root, config_dir, backbone, dataset, device, seed):
    dataset_config = Path(config_dir) / f"{dataset}.yaml"
    settings = _load_settings(dataset_config)

    dataset_name, items = load_cache_items(data_root, backbone, dataset, "test")

    label_to_name = {}

    for item in sorted(items, key=lambda item: item.label):
        label_to_name.setdefault(item.label, item.classname)

    label_order = list(label_to_name.keys())
    names = list(label_to_name.values())
    label_to_index = {label: idx for idx, label in enumerate(label_order)}

    labels = np.array([label_to_index[item.label] for item in items])
    labels_t = torch.as_tensor(labels, device=device, dtype=torch.long)

    text_proto = _text_proto(
        model,
        names,
        _dataset_templates(dataset_name),
        device,
    )

    avg = _load_cached_avg(
        data_root,
        backbone,
        dataset_config,
        items,
        device,
        "test",
    )

    fit_embeds, fit_source = load_fit_embeds(
        data_root,
        backbone,
        dataset,
        device,
    )

    if settings.n_clusters is None:
        def recenter(values, beta):
            return F.normalize(values, dim=-1)
    else:
        recenter = _fit_recenter(
            avg,
            seed=seed,
            n_clusters=settings.n_clusters,
        )

    test_rm = recenter(avg, settings.beta)
    test_logits = test_rm @ text_proto.t() * 100.0

    if fit_source == "val":
        fit_rm = recenter(fit_embeds, settings.beta)
        prior_logits = fit_rm @ text_proto.t() * 100.0
    else:
        prior_logits = test_logits

    prior, correction = build_conf_prior(prior_logits)

    corrected_logits = test_logits + correction.unsqueeze(0)

    rm_pred, rm_acc = accuracy(test_logits, labels_t)
    final_pred, final_acc = accuracy(corrected_logits, labels_t)

    num_classes = len(names)

    true_freq = torch.bincount(labels_t, minlength=num_classes).float()
    true_freq = true_freq / true_freq.sum()

    rm_freq = torch.bincount(rm_pred, minlength=num_classes).float()
    rm_freq = rm_freq / rm_freq.sum()

    final_freq = torch.bincount(final_pred, minlength=num_classes).float()
    final_freq = final_freq / final_freq.sum()

    rm_l1 = torch.abs(rm_freq - true_freq).sum().item()
    final_l1 = torch.abs(final_freq - true_freq).sum().item()

    rm_ok = rm_pred == labels_t
    final_ok = final_pred == labels_t

    w2r = ((~rm_ok) & final_ok).sum().item()
    r2w = (rm_ok & (~final_ok)).sum().item()
    changed = (rm_pred != final_pred).sum().item()

    return {
        "dataset": dataset,
        "fit_source": fit_source,
        "rm_acc": rm_acc,
        "final_acc": final_acc,
        "gain": final_acc - rm_acc,
        "rm_l1": rm_l1,
        "final_l1": final_l1,
        "l1_change": final_l1 - rm_l1,
        "w2r": int(w2r),
        "r2w": int(r2w),
        "changed": int(changed),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", default="all")
    parser.add_argument(
        "--backbone",
        default="ViT-B/16",
        choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"],
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    datasets = (
        DATASETS
        if args.datasets == "all"
        else [name.strip() for name in args.datasets.split(",")]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = clip.load(args.backbone, device=device)
    model.eval()

    rows = [
        run_dataset(
            model,
            args.data_root,
            args.config_dir,
            args.backbone,
            dataset,
            device,
            args.seed,
        )
        for dataset in datasets
    ]

    print("\n" + "=" * 124)
    print("R+M + CONFIDENCE-WEIGHTED PRIOR")

    print(
        f"{'dataset':<18}"
        f"{'R+M':>10}"
        f"{'Conf':>10}"
        f"{'gain':>9}"
        f"{'L1 before':>12}"
        f"{'L1 after':>11}"
        f"{'ΔL1':>10}"
        f"{'W→R':>7}"
        f"{'R→W':>7}"
        f"{'changed':>10}"
    )

    for row in rows:
        print(
            f"{row['dataset']:<18}"
            f"{row['rm_acc']:>9.2f}%"
            f"{row['final_acc']:>9.2f}%"
            f"{row['gain']:>+9.2f}"
            f"{row['rm_l1']:>12.4f}"
            f"{row['final_l1']:>11.4f}"
            f"{row['l1_change']:>+10.4f}"
            f"{row['w2r']:>7}"
            f"{row['r2w']:>7}"
            f"{row['changed']:>10}"
        )

    mean_rm = np.mean([row["rm_acc"] for row in rows])
    mean_final = np.mean([row["final_acc"] for row in rows])
    mean_gain = np.mean([row["gain"] for row in rows])

    positive = sum(row["gain"] > 0 for row in rows)
    non_negative = sum(row["gain"] >= 0 for row in rows)
    l1_improved = sum(row["l1_change"] < 0 for row in rows)

    print()
    print(f"Mean R+M accuracy     : {mean_rm:.3f}%")
    print(f"Mean Conf accuracy    : {mean_final:.3f}%")
    print(f"Mean Conf gain        : {mean_gain:+.3f} pp")
    print(f"Positive datasets     : {positive}/{len(rows)}")
    print(f"Non-negative datasets : {non_negative}/{len(rows)}")
    print(f"Frequency L1 improved : {l1_improved}/{len(rows)}")


if __name__ == "__main__":
    main()