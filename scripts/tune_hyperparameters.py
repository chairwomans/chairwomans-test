"""Tune recentering settings on a dataset validation split and emit JSON results."""

import argparse
import json
from pathlib import Path
import sys

import clip
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset_setup import load_evaluation_dataset
from utils.final_evaluation import (_confidence_prior, _dataset_templates,
                                    _fit_recenter, _load_cached_avg,
                                    _load_cached_raw, _text_proto)


def _parse_clusters(values):
    return tuple(None if value.strip().lower() == "none" else int(value) for value in values.split(","))


def _parse_floats(values):
    return tuple(float(value) for value in values.split(","))


@torch.no_grad()
def tune(model, data_root, config, clusters, betas, device):
    dataset_id, dataset, _ = load_evaluation_dataset(data_root, config)
    if not getattr(dataset, "val", None):
        raise ValueError(f"{Path(config).stem} has no validation split; refusing to tune on test data")
    items = dataset.val
    label_to_name = {}
    for item in sorted(items, key=lambda item: item.label):
        label_to_name.setdefault(item.label, item.classname)
    label_to_index = {label: index for index, label in enumerate(label_to_name)}
    labels = np.asarray([label_to_index[item.label] for item in items])
    text_proto = _text_proto(model, list(label_to_name.values()), _dataset_templates(dataset_id), device)
    avg = _load_cached_avg(data_root, "ViT-B/16", config, items, device, "val")
    raw = _load_cached_raw(data_root, "ViT-B/16", config, items, device, "val")
    candidates = []
    for k in clusters:
        recenter = (lambda values, beta: values) if k is None else _fit_recenter(avg, seed=42, n_clusters=k)
        for beta in ((0.0,) if k is None else betas):
            logits = recenter(avg, beta) @ text_proto.t() * 100
            prior_logits = recenter(raw, beta) @ text_proto.t() * 100
            _, correction = _confidence_prior(prior_logits)
            score = float(((logits + correction).argmax(-1).cpu().numpy() == labels).mean() * 100)
            candidates.append({"score": score, "K": k, "beta": beta})
    candidates.sort(key=lambda row: (-row["score"], row["K"] is not None, row["K"] or 0, row["beta"]))
    return {"dataset": Path(config).stem, "validation_examples": len(items), "candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--clusters", required=True, help="Comma-separated K values; use none for no recentering")
    parser.add_argument("--betas", required=True, help="Comma-separated beta values")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Hyperparameter tuning requires CUDA")
    model, _ = clip.load("ViT-B/16", device="cuda")
    model.eval()
    result = tune(model, "data", args.config, _parse_clusters(args.clusters), _parse_floats(args.betas), "cuda")
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
