"""Grid-search a dataset's evaluation settings using the existing feature cache."""

import argparse
from pathlib import Path
import sys

import clip
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset_setup import load_evaluation_dataset
from utils.final_evaluation import (
    _fit_recenter,
    _load_cached_avg,
    _style_banks,
    _text_proto,
    _dataset_templates,
)


def accuracy(probabilities, labels):
    return float((probabilities.argmax(-1).cpu().numpy() == labels).mean() * 100)


def style_probabilities(rec, avg, rm_logits, text_bank, style_proto, style_class_proto):
    batch_size = 256
    top2 = rm_logits.topk(2, dim=-1).indices
    support = []
    for start in range(0, len(rec), batch_size):
        embeds = rec[start:start + batch_size]
        first = text_bank[top2[start:start + batch_size, 0]]
        second = text_bank[top2[start:start + batch_size, 1]]
        score_first = torch.einsum("bd,bmd->bm", embeds, first)
        score_second = torch.einsum("bd,bmd->bm", embeds, second)
        support.append((score_first > score_second).float().mean(-1))
    uncertainty = 1 - torch.cat(support)

    style_weights = F.softmax(avg @ style_proto.t() * 100, dim=-1)
    probabilities = []
    for start in range(0, len(rec), batch_size):
        embeds = rec[start:start + batch_size]
        logits = torch.einsum("bd,skd->bsk", embeds, style_class_proto) * 100
        per_style = F.softmax(logits, dim=-1)
        probabilities.append((per_style * style_weights[start:start + batch_size].unsqueeze(-1)).sum(1))
    return uncertainty, torch.cat(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/imagenet.yaml")
    parser.add_argument("--clusters", required=True, help="Comma-separated K values; use 'none' for no recentering")
    parser.add_argument("--betas", required=True, help="Comma-separated beta values")
    parser.add_argument("--lambdas", required=True, help="Comma-separated style lambda values")
    args = parser.parse_args()
    clusters = tuple(None if value.strip().lower() == "none" else int(value) for value in args.clusters.split(","))
    betas = tuple(float(value) for value in args.betas.split(","))
    style_lambdas = tuple(float(value) for value in args.lambdas.split(","))
    data_root = "data"
    config = args.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("ImageNet tuning requires CUDA")

    dataset_id, dataset, _ = load_evaluation_dataset(data_root, config)
    items = dataset.test
    label_to_name = {}
    for item in sorted(items, key=lambda item: item.label):
        label_to_name.setdefault(item.label, item.classname)
    names = list(label_to_name.values())
    label_to_index = {label: index for index, label in enumerate(label_to_name)}
    labels = np.array([label_to_index[item.label] for item in items])

    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    templates = _dataset_templates(dataset_id)
    text_proto = _text_proto(model, names, templates, device)
    text_bank, style_proto, style_class_proto = _style_banks(model, names, templates, device)
    avg = _load_cached_avg(data_root, "ViT-B/16", config, items, device)
    if avg is None:
        raise FileNotFoundError("ImageNet avg feature cache is required")

    candidates = []
    recenter_cache = {None: avg}
    for n_clusters in clusters:
        if n_clusters is not None:
            print(f"Fitting recenter model: K={n_clusters}", flush=True)
            recenter_cache[n_clusters] = _fit_recenter(avg, seed=42, n_clusters=n_clusters)

    for n_clusters in clusters:
        beta_values = (0.0,) if n_clusters is None else betas
        for beta in beta_values:
            rec = avg if n_clusters is None else recenter_cache[n_clusters](avg, beta)
            rm_logits = rec @ text_proto.t() * 100
            rm_prob = F.softmax(rm_logits, dim=-1)
            uncertainty, style_prob = style_probabilities(
                rec, avg, rm_logits, text_bank, style_proto, style_class_proto
            )
            for style_lambda in style_lambdas:
                final_prob = (1 - style_lambda * uncertainty[:, None]) * rm_prob
                final_prob += style_lambda * uncertainty[:, None] * style_prob
                score = accuracy(final_prob, labels)
                candidates.append((score, n_clusters, beta, style_lambda))
                print(
                    f"score={score:.4f} K={n_clusters} beta={beta:.4f} lambda={style_lambda:.4f}",
                    flush=True,
                )
            del rec, rm_logits, rm_prob, uncertainty, style_prob
            torch.cuda.empty_cache()

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    print("TOP_RESULTS", flush=True)
    for score, clusters, beta, style_lambda in candidates[:10]:
        print(
            f"score={score:.4f} K={clusters} beta={beta:.4f} lambda={style_lambda:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
