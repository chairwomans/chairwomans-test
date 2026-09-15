"""Mirror + GMM recenter + style-conditioned CLIP comparison pipeline."""


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

import clip
from data.local_datasets import class_names, load_dataset

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.",
    "a photo of the hard to see {}.", "a low resolution photo of the {}.",
    "a rendering of a {}.", "graffiti of a {}.", "a bad photo of the {}.",
    "a cropped photo of the {}.", "a tattoo of a {}.", "the embroidered {}.",
    "a photo of a hard to see {}.", "a bright photo of a {}.", "a photo of a clean {}.",
    "a photo of a dirty {}.", "a dark photo of the {}.", "a drawing of a {}.",
    "a photo of my {}.", "the plastic {}.", "a photo of the cool {}.",
    "a close-up photo of a {}.", "a black and white photo of the {}.",
    "a painting of the {}.", "a painting of a {}.", "a pixelated photo of the {}.",
    "a sculpture of the {}.", "a bright photo of the {}.", "a cropped photo of a {}.",
    "a plastic {}.", "a photo of the dirty {}.", "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.", "a photo of the {}.", "a good photo of the {}.",
    "a rendering of the {}.", "a {} in a video game.", "a photo of one {}.",
    "a doodle of a {}.", "a close-up photo of the {}.", "a photo of a {}.",
    "the origami {}.", "the {} in a video game.", "a sketch of a {}.",
    "a doodle of the {}.", "a origami {}.", "a low resolution photo of a {}.",
    "the toy {}.", "a rendition of the {}.", "a photo of the clean {}.",
    "a photo of a large {}.", "a rendition of a {}.", "a photo of a nice {}.",
    "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.",
    "art of a {}.", "a sketch of the {}.", "a embroidered {}.",
    "a pixelated photo of a {}.", "itap of the {}.", "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.", "a plushie {}.", "a photo of the nice {}.",
    "a photo of the small {}.", "a photo of the weird {}.", "the cartoon {}.",
    "art of the {}.", "a drawing of the {}.", "a photo of the large {}.",
    "a black and white photo of a {}.", "the plushie {}.", "a dark photo of a {}.",
    "itap of a {}.", "graffiti of the {}.", "a toy {}.", "itap of my {}.",
    "a photo of a cool {}.", "a photo of a small {}.", "a tattoo of the {}.",
]
STYLE_SPECS = [("photo", "a real photograph", "a photo of a {}"),
               ("painting", "a painting", "a painting of a {}"),
               ("cartoon", "a cartoon illustration", "a cartoon of a {}"),
               ("sketch", "a sketch", "a sketch of a {}"),
               ("line_drawing", "a line drawing", "a line drawing of a {}"),
               ("digital_illustration", "a digital illustration", "a digital illustration of a {}"),
               ("3d_render", "a 3d rendered image", "a 3d rendering of a {}"),
               ("sculpture", "a sculpture", "a sculpture of a {}")]


def _encode(model, preprocess, images, device):
    batch = torch.stack([preprocess(image.convert("RGB")) for image in images]).to(device)
    with torch.no_grad():
        emb = model.encode_image(batch).float()
    return F.normalize(emb, dim=-1)


def _text_proto(model, class_names, device):
    rows = []
    for name in tqdm(class_names, desc="text prototypes"):
        tokens = clip.tokenize([t.format(name.replace("_", " ")) for t in IMAGENET_TEMPLATES], truncate=True).to(device)
        with torch.no_grad():
            rows.append(F.normalize(model.encode_text(tokens).float().mean(0), dim=0))
    return torch.stack(rows)


def _style_banks(model, class_names, device):
    text_bank = []
    for name in tqdm(class_names, desc="template bank"):
        tokens = clip.tokenize([t.format(name.replace("_", " ")) for t in IMAGENET_TEMPLATES], truncate=True).to(device)
        with torch.no_grad():
            text_bank.append(F.normalize(model.encode_text(tokens).float(), dim=-1))
    text_bank = torch.stack(text_bank)
    style_tokens = clip.tokenize([row[1] for row in STYLE_SPECS], truncate=True).to(device)
    with torch.no_grad():
        style_proto = F.normalize(model.encode_text(style_tokens).float(), dim=-1)
    style_class_proto = []
    for _, _, template in STYLE_SPECS:
        tokens = clip.tokenize([template.format(name.replace("_", " ")) for name in class_names], truncate=True).to(device)
        with torch.no_grad():
            style_class_proto.append(F.normalize(model.encode_text(tokens).float(), dim=-1))
    return text_bank, style_proto, torch.stack(style_class_proto)


def _fit_recenter(embeds, seed, n_clusters):
    mean_np = embeds.cpu().numpy().mean(0)
    pca = PCA(n_components=min(16, embeds.shape[1], len(embeds)), random_state=seed).fit(embeds.cpu().numpy() - mean_np)
    reduced = pca.transform(embeds.cpu().numpy() - mean_np)
    gmm = GaussianMixture(n_components=min(n_clusters, len(embeds)), covariance_type="diag", random_state=seed, n_init=5).fit(reduced)
    hard = gmm.predict(reduced)
    means = torch.stack([embeds[hard == k].mean(0) if (hard == k).any() else embeds.mean(0) for k in range(gmm.n_components)])
    global_mean = torch.tensor(mean_np, device=embeds.device, dtype=embeds.dtype)

    def recenter(values, beta):
        probs = torch.tensor(gmm.predict_proba(pca.transform(values.detach().cpu().numpy() - mean_np)), device=values.device, dtype=values.dtype)
        return F.normalize(values - beta * (probs @ means), dim=-1)
    return recenter


def run_dataset(model, preprocess, dataset_id, data_root, device, seed=42, beta=0.35, batch_size=64, max_test_samples=None, **dataset_kwargs):
    dataset = load_dataset(dataset_id, data_root, **dataset_kwargs)
    names = class_names(dataset)
    n = len(dataset)
    test_idx = list(range(n)) if max_test_samples is None else list(range(min(max_test_samples, n)))
    def images_at(ids): return [dataset[i][0] for i in ids]
    text_proto = _text_proto(model, names, device)
    text_bank, style_proto, style_class_proto = _style_banks(model, names, device)
    # Fit the unsupervised recentering model on the complete test set.
    # There is no random train/validation split and no test-label tuning.
    fit_embeds = torch.cat([_encode(model, preprocess, images_at(test_idx[i:i+batch_size]), device) for i in range(0, len(test_idx), batch_size)])
    recenter = _fit_recenter(fit_embeds, seed, 6)
    raw, mirrored, labels = [], [], []
    for start in tqdm(range(0, len(test_idx), batch_size), desc=f"{dataset_id} test", leave=False):
        batch = images_at(test_idx[start:start+batch_size])
        raw.append(_encode(model, preprocess, batch, device))
        mirrored.append(_encode(model, preprocess, [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in batch], device))
        labels.extend(dataset[i][1] for i in test_idx[start:start+batch_size])
    raw, mirrored = torch.cat(raw), torch.cat(mirrored)
    avg = F.normalize((raw + mirrored) / 2, dim=-1)
    rec = recenter(avg, best_beta)
    clip_prob = F.softmax(raw @ text_proto.t() * 100, dim=-1)
    rm_logits = rec @ text_proto.t() * 100
    rm_prob = F.softmax(rm_logits, dim=-1)
    top2 = rm_logits.topk(2, dim=-1).indices
    support = []
    for start in range(0, len(rec), batch_size):
        emb = rec[start:start + batch_size]
        t1, t2 = text_bank[top2[start:start + batch_size, 0]], text_bank[top2[start:start + batch_size, 1]]
        s1 = torch.einsum("bd,bmd->bm", emb, t1)
        s2 = torch.einsum("bd,bmd->bm", emb, t2)
        support.append((s1 > s2).float().mean(-1))
    uncertainty = 1 - torch.cat(support)
    style_weights = F.softmax(avg @ style_proto.t() * 100, dim=-1)
    style_prob = []
    for start in range(0, len(rec), batch_size):
        emb = rec[start:start + batch_size]
        logits = torch.einsum("bd,skd->bsk", emb, style_class_proto) * 100
        per_style = F.softmax(logits, dim=-1)
        style_prob.append((per_style * style_weights[start:start + batch_size].unsqueeze(-1)).sum(1))
    style_prob = torch.cat(style_prob)
    final_prob = (1 - uncertainty[:, None]) * rm_prob + uncertainty[:, None] * style_prob
    clip_pred = clip_prob.argmax(-1).cpu().numpy()
    final_pred = final_prob.argmax(-1).cpu().numpy()
    labels = np.array(labels)
    return {"dataset": dataset_id, "samples": len(labels), "beta": beta,
            "clip": float((clip_pred == labels).mean() * 100),
            "final": float((final_pred == labels).mean() * 100)}


def run_comparison(model_name, data_root, dataset_id: str, seed=42, beta=0.35, max_test_samples=None, **dataset_kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    results = [run_dataset(model, preprocess, dataset_id, data_root, device, seed, beta=beta, max_test_samples=max_test_samples, **dataset_kwargs)]
    print("\n" + "=" * 64)
    print(f"{'dataset':<18}{'samples':>10}{'CLIP':>12}{'FINAL':>12}{'gain':>12}")
    for row in results:
        print(f"{row['dataset']:<18}{row['samples']:>10}{row['clip']:>11.2f}%{row['final']:>11.2f}%{row['final']-row['clip']:>+11.2f}pp")
    return results


def calibrate_beta(model_name, data_root, dataset_id, device=None, seed=42,
                   beta_grid=(0.0, 0.2, 0.35, 0.5), max_samples=None,
                   **dataset_kwargs):
    """Select beta on a separate labeled calibration split and return metrics."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    dataset = load_dataset(dataset_id, data_root, **dataset_kwargs)
    names = class_names(dataset)
    indices = list(range(len(dataset)))
    if max_samples is not None:
        indices = indices[:max_samples]
    text_proto = _text_proto(model, names, device)
    embeds = []
    for start in tqdm(range(0, len(indices), 64), desc=f"{dataset_id} calibration"):
        embeds.append(_encode(model, preprocess, [dataset[i][0] for i in indices[start:start + 64]], device))
    embeds = torch.cat(embeds)
    labels = np.array([dataset[i][1] for i in indices])
    recenter = _fit_recenter(embeds, seed, 6)
    scores = {}
    for beta in beta_grid:
        pred = (recenter(embeds, beta) @ text_proto.t() * 100).argmax(-1).cpu().numpy()
        scores[str(beta)] = float((pred == labels).mean() * 100)
    best_beta = max(scores, key=scores.get)
    return {"dataset": dataset_id, "backbone": model_name, "samples": len(indices),
            "beta": float(best_beta), "grid": scores}
