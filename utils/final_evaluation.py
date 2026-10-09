import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import clip

from utils import templates
from utils.domain_prompts import DOMAIN_PROMPTS


def _cache_directory(data_root, model_name, dataset, split="test"):
    cache_backbone = model_name.replace("/", "-")
    directory = Path(data_root) / "cache" / cache_backbone / Path(dataset).stem
    return directory if split == "test" else directory / split


def load_cached_raw(data_root, backbone, dataset, split, device):
    cache_dir = _cache_directory(data_root, backbone, dataset, split)
    raw_path = cache_dir / "raw.pt"
    metadata_path = cache_dir / "metadata.json"
    if not raw_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{split} feature cache not found: {cache_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("backbone") != backbone:
        raise ValueError(
            f"Cache backbone mismatch in {cache_dir}: expected {backbone}, "
            f"found {metadata.get('backbone')}"
        )

    raw = torch.load(raw_path, map_location=device, weights_only=True).float()
    items = metadata.get("items", [])
    if raw.ndim != 2 or raw.shape[0] != len(items):
        raise ValueError(f"Invalid cached raw feature shape {tuple(raw.shape)} in {raw_path}")
    return F.normalize(raw, dim=-1), items, metadata


def build_label_space(items):
    label_to_name = {}
    for item in sorted(items, key=lambda item: item["label"]):
        label_to_name.setdefault(item["label"], item["classname"])
    label_to_index = {label: index for index, label in enumerate(label_to_name)}
    labels = torch.tensor([label_to_index[item["label"]] for item in items], dtype=torch.long)
    return list(label_to_name.values()), labels


@torch.no_grad()
def build_domain_embeddings(model, device):
    embeddings = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        encoded = model.encode_text(clip.tokenize(prompts).to(device)).float()
        encoded = F.normalize(encoded, dim=-1)
        embeddings[domain] = encoded.mean(dim=0)
    domain_names = list(embeddings)
    matrix = torch.stack([embeddings[name] for name in domain_names]).to(device)
    return domain_names, matrix


@torch.no_grad()
def evaluate_text_only(features, labels, domain_matrix, proto_tensor, logit_scale, device, batch_size):
    correct = 0
    total = len(labels)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        values = features[start:end].to(device)
        target = labels[start:end].to(device)
        domain_weights = F.softmax((values @ domain_matrix.T) * logit_scale, dim=-1)
        text_star = torch.einsum("bd,dcx->bcx", domain_weights, proto_tensor)
        text_star = F.normalize(text_star, dim=-1)
        logits = torch.einsum("bx,bcx->bc", values, text_star)
        correct += (logits.argmax(dim=-1) == target).sum().item()
    return correct / total if total else 0.0, correct, total


_TEMPLATES_BY_DATASET = {
    "ImageNet": templates.imagenet_templates,
    "ImageNetA": templates.imagenet_templates,
    "ImageNetR": templates.imagenet_templates,
    "ImageNetV2": templates.imagenet_templates,
    "ImageNetSketch": templates.imagenet_templates,
    "Caltech101": templates.caltech101_templates,
    "DescribableTextures": templates.dtd_templates,
    "EuroSAT": templates.eurosat_templates,
    "FGVCAircraft": templates.aircraft_templates,
    "Food101": templates.food101_templates,
    "OxfordFlowers": templates.flowers_templates,
    "OxfordPets": templates.pets_templates,
    "SUN397": templates.sun397_templates,
    "StanfordCars": templates.cars_templates,
    "UCF101": templates.ucf101_templates,
}


def _dataset_templates(dataset_name):
    try:
        return _TEMPLATES_BY_DATASET[dataset_name]
    except KeyError as error:
        raise KeyError(f"No prompt templates registered for CoOp dataset: {dataset_name}") from error


def _domain_class_prompts(domain, class_name, dataset_templates, max_prompts):
    combined = [
        f"{domain_phrase}, {template.format(class_name)}"
        for domain_phrase in DOMAIN_PROMPTS[domain]
        for template in dataset_templates
    ]
    if len(combined) <= max_prompts:
        return combined
    stride = len(combined) / max_prompts
    return [combined[int(index * stride)] for index in range(max_prompts)]


@torch.no_grad()
def build_domain_text_prototypes_templated(model, class_names, domain_names, dataset_templates, device, max_prompts=40):
    per_domain = []
    for domain in domain_names:
        prototypes = []
        for class_name in tqdm(class_names, desc=f"{domain} text prototypes", leave=False):
            prompts = _domain_class_prompts(domain, class_name.replace("_", " "), dataset_templates, max_prompts)
            encoded = model.encode_text(clip.tokenize(prompts, truncate=True).to(device)).float()
            encoded = F.normalize(encoded, dim=-1)
            prototypes.append(F.normalize(encoded.mean(dim=0), dim=0))
        per_domain.append(torch.stack(prototypes))
    return torch.stack(per_domain)


def run_dataset_templated(model, data_root, backbone, dataset, device, domain_names, domain_matrix,
                           logit_scale, batch_size, max_prompts, dry_run=False):
    features, items, metadata = load_cached_raw(data_root, backbone, dataset, "test", device)
    class_names, labels = build_label_space(items)
    dataset_name = metadata.get("dataset_name", dataset)

    if dry_run:
        return {"dataset": dataset, "samples": len(items), "classes": len(class_names)}

    dataset_templates = _dataset_templates(dataset_name)
    proto_tensor = build_domain_text_prototypes_templated(
        model, class_names, domain_names, dataset_templates, device, max_prompts
    )
    accuracy, correct, total = evaluate_text_only(
        features, labels, domain_matrix, proto_tensor, logit_scale, device, batch_size
    )
    print(f"{dataset_name}: templated domain hint accuracy {accuracy * 100:.2f}% ({correct}/{total})")
    return {
        "dataset": dataset,
        "samples": total,
        "classes": len(class_names),
        "templated": accuracy * 100.0,
    }


def run_domain_hint_templated(backbone, data_root, datasets, batch_size=512, max_prompts=40, dry_run=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dry_run:
        rows = [
            run_dataset_templated(None, data_root, backbone, dataset, device, [], None, 0.0, batch_size,
                                   max_prompts, dry_run=True)
            for dataset in datasets
        ]
        print("\n" + "=" * 54)
        print(f"{'dataset':<18}{'samples':>12}{'classes':>12}")
        for row in rows:
            print(f"{row['dataset']:<18}{row['samples']:>12}{row['classes']:>12}")
        return rows

    print("device:", device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    model, _ = clip.load(backbone, device=device)
    model.eval()

    domain_names, domain_matrix = build_domain_embeddings(model, device)
    logit_scale = model.logit_scale.exp().item()
    print("Domain prompts:", ", ".join(domain_names))
    print(f"Domain logit scale: {logit_scale:.2f}")
    print(f"Ensemble cap per domain/class: {max_prompts}")

    rows = [
        run_dataset_templated(
            model, data_root, backbone, dataset, device, domain_names,
            domain_matrix, logit_scale, batch_size, max_prompts,
        )
        for dataset in datasets
    ]

    print("\n" + "=" * 54)
    print("TEMPLATED DOMAIN HINT")
    print(f"{'dataset':<18}{'samples':>12}{'classes':>12}{'accuracy':>12}")
    for row in rows:
        print(
            f"{row['dataset']:<18}{row['samples']:>12}{row['classes']:>12}"
            f"{row['templated']:>11.2f}%"
        )
    mean_accuracy = sum(row["templated"] for row in rows) / len(rows) if rows else 0.0
    print()
    print(f"Mean templated accuracy: {mean_accuracy:.3f}%")
    return rows
