"""Final CLIP evaluation pipeline on each CoOp dataset's fixed test split."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

import clip
from utils.dataset_setup import load_evaluation_dataset
from utils import templates

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


def _encode(model, preprocess, images, device):
    batch = torch.stack([preprocess(image.convert("RGB")) for image in images]).to(device)
    with torch.no_grad():
        emb = model.encode_image(batch).float()
    return F.normalize(emb, dim=-1)


def _open_images(items, indices):
    """Load copies so PIL closes each source file before the GPU batch is built."""
    images = []
    for index in indices:
        with Image.open(items[index].impath) as image:
            images.append(image.copy())
    return images


def _cache_directory(data_root, model_name, dataset_config, split="test"):
    """Return the cache location associated with one backbone and config stem."""
    cache_backbone = model_name.replace("/", "-")
    directory = Path(data_root) / "cache" / cache_backbone / Path(dataset_config).stem
    return directory if split == "test" else directory / split


def _load_cached_avg(data_root, model_name, dataset_config, items, device, split="test"):
    """Load a compatible cached flip-average feature matrix, if it exists."""
    cache_dir = _cache_directory(data_root, model_name, dataset_config, split)
    avg_path = cache_dir / "avg.pt"
    metadata_path = cache_dir / "metadata.json"
    if not avg_path.is_file() or not metadata_path.is_file():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split", "test") != split:
        raise ValueError(f"Cache split mismatch in {cache_dir}: expected {split}")
    if metadata.get("backbone") != model_name:
        raise ValueError(
            f"Cache backbone mismatch in {cache_dir}: "
            f"expected {model_name}, found {metadata.get('backbone')}"
        )
    cached_items = metadata.get("items", [])
    if len(cached_items) != len(items):
        raise ValueError(
            f"Cache item count mismatch in {cache_dir}: "
            f"expected {len(items)}, found {len(cached_items)}"
        )

    root = Path(data_root)
    for index, (cached, item) in enumerate(zip(cached_items, items)):
        expected = {
            "path": str(Path(item.impath).relative_to(root)),
            "label": item.label,
            "classname": item.classname,
        }
        if cached != expected:
            raise ValueError(f"Cache metadata differs from the current test split at index {index}: {cache_dir}")

    avg = torch.load(avg_path, map_location=device, weights_only=True).float()
    if avg.ndim != 2 or avg.shape[0] != len(items):
        raise ValueError(f"Invalid cached avg feature shape {tuple(avg.shape)} in {avg_path}")
    print(f"Using cached image features: {cache_dir}")
    return F.normalize(avg, dim=-1)


def _dataset_templates(dataset_name):
    try:
        return _TEMPLATES_BY_DATASET[dataset_name]
    except KeyError as error:
        raise KeyError(f"No prompt templates registered for CoOp dataset: {dataset_name}") from error


def _text_proto(model, class_names, prompt_templates, device, prompt_batch_size=128):
    """Encode template ensembles in batches while preserving per-class averaging."""
    rows = []
    templates_per_class = len(prompt_templates)
    classes_per_batch = max(1, prompt_batch_size // templates_per_class)
    for start in tqdm(range(0, len(class_names), classes_per_batch), desc="text prototypes"):
        names = class_names[start:start + classes_per_batch]
        prompts = [
            template.format(name.replace("_", " "))
            for name in names
            for template in prompt_templates
        ]
        tokens = clip.tokenize(prompts, truncate=True).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(tokens).float().reshape(len(names), templates_per_class, -1)
        rows.append(F.normalize(embeddings.mean(1), dim=-1))
    return torch.cat(rows)


def _fit_recenter(embeds, seed, n_clusters):
    mean_np = embeds.cpu().numpy().mean(0)
    pca = PCA(n_components=min(16, embeds.shape[1], len(embeds)), random_state=seed).fit(embeds.cpu().numpy() - mean_np)
    reduced = pca.transform(embeds.cpu().numpy() - mean_np)
    gmm = GaussianMixture(n_components=min(n_clusters, len(embeds)), covariance_type="diag", random_state=seed, n_init=5).fit(reduced)
    hard = gmm.predict(reduced)
    means = torch.stack([embeds[hard == k].mean(0) if (hard == k).any() else embeds.mean(0) for k in range(gmm.n_components)])
    def recenter(values, beta):
        probs = torch.tensor(gmm.predict_proba(pca.transform(values.detach().cpu().numpy() - mean_np)), device=values.device, dtype=values.dtype)
        return F.normalize(values - beta * (probs @ means), dim=-1)
    return recenter


def _load_cached_raw(data_root, model_name, dataset_config, items, device, split):
    """Load normalized, non-flip image features after cache compatibility checks."""
    # _load_cached_avg validates the metadata and item ordering shared by raw.pt.
    _load_cached_avg(data_root, model_name, dataset_config, items, device, split)
    raw_path = _cache_directory(data_root, model_name, dataset_config, split) / "raw.pt"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw feature cache is missing: {raw_path}")
    raw = torch.load(raw_path, map_location=device, weights_only=True).float()
    if raw.ndim != 2 or raw.shape[0] != len(items):
        raise ValueError(f"Invalid cached raw feature shape {tuple(raw.shape)} in {raw_path}")
    return F.normalize(raw, dim=-1)


def _confidence_prior(logits):
    """Return the confidence-weighted class-prior logit correction."""
    probabilities = F.softmax(logits, dim=-1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
    confidence = 1.0 - entropy / np.log(probabilities.shape[1])
    prior = (confidence[:, None] * probabilities).sum(dim=0)
    prior = (prior / confidence.sum().clamp_min(1e-12)).clamp_min(1e-12)
    prior = prior / prior.sum()
    correction = -torch.log(prior)
    return prior, correction - correction.mean()


def run_dataset(model, preprocess, data_root, dataset_config, model_name, device, batch_size=64, split="test"):
    dataset_id, dataset, settings = load_evaluation_dataset(data_root, dataset_config)
    items = getattr(dataset, split)
    if not items:
        raise ValueError(f"Dataset has no usable {split} split: {dataset_id}")
    # Some ImageNet-derived datasets omit classes but retain their original
    # ImageNet label IDs (e.g. 0, 1, 3, ...).  Build prototypes in label order
    # and remap their sparse IDs to the contiguous prototype indices.
    label_to_name = {}
    for item in sorted(items, key=lambda item: item.label):
        label_to_name.setdefault(item.label, item.classname)
    names = list(label_to_name.values())
    label_to_index = {label: index for index, label in enumerate(label_to_name)}
    prompt_templates = _dataset_templates(dataset_id)
    text_proto = _text_proto(model, names, prompt_templates, device)
    avg = _load_cached_avg(data_root, model_name, dataset_config, items, device, split)
    if avg is None:
        original, mirrored = [], []
        for start in tqdm(range(0, len(items), batch_size), desc=f"{dataset_id} test", leave=False):
            batch = _open_images(items, range(start, min(start + batch_size, len(items))))
            original.append(_encode(model, preprocess, batch, device))
            mirrored.append(_encode(model, preprocess, [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in batch], device))
        avg = F.normalize((torch.cat(original) + torch.cat(mirrored)) / 2, dim=-1)
    # Hyperparameters and the prior are fit only from validation features.
    fit_items = getattr(dataset, "val", None) or items
    fit_split = "val" if getattr(dataset, "val", None) else split
    fit_avg = _load_cached_avg(data_root, model_name, dataset_config, fit_items, device, fit_split)
    if fit_avg is None:
        raise FileNotFoundError(f"Feature cache is required for the fitting split ({fit_split})")
    if settings.n_clusters is None:
        recenter = lambda values, beta: F.normalize(values, dim=-1)
    else:
        recenter = _fit_recenter(fit_avg, seed=42, n_clusters=settings.n_clusters)
    rec = recenter(avg, settings.beta)
    fit_raw = _load_cached_raw(data_root, model_name, dataset_config, fit_items, device, fit_split)
    prior_logits = recenter(fit_raw, settings.beta) @ text_proto.t() * 100
    _, correction = _confidence_prior(prior_logits)
    final_pred = (rec @ text_proto.t() * 100 + correction).argmax(-1).cpu().numpy()
    labels = np.array([label_to_index[item.label] for item in items])
    return {"dataset": dataset_id, "final": float((final_pred == labels).mean() * 100), "fit_split": fit_split}


def run_final_evaluation(model_name, data_root, dataset_config, split="test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    results = [run_dataset(
        model, preprocess, data_root, dataset_config, model_name, device, split=split,
    )]
    print("\n" + "=" * 30)
    print(f"{'dataset':<18}{'FINAL':>12}")
    for row in results:
        print(f"{row['dataset']:<18}{row['final']:>11.2f}%")
    return results
