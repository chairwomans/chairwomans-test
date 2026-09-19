"""Final CLIP evaluation pipeline on each CoOp dataset's fixed test split."""

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

STYLE_SPECS = [("photo", "a real photograph", "a photo of a {}"),
               ("painting", "a painting", "a painting of a {}"),
               ("cartoon", "a cartoon illustration", "a cartoon of a {}"),
               ("sketch", "a sketch", "a sketch of a {}"),
               ("line_drawing", "a line drawing", "a line drawing of a {}"),
               ("digital_illustration", "a digital illustration", "a digital illustration of a {}"),
               ("3d_render", "a 3d rendered image", "a 3d rendering of a {}"),
               ("sculpture", "a sculpture", "a sculpture of a {}")]

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


def _dataset_templates(dataset_name):
    try:
        return _TEMPLATES_BY_DATASET[dataset_name]
    except KeyError as error:
        raise KeyError(f"No prompt templates registered for CoOp dataset: {dataset_name}") from error


def _text_proto(model, class_names, prompt_templates, device):
    rows = []
    for name in tqdm(class_names, desc="text prototypes"):
        tokens = clip.tokenize([t.format(name.replace("_", " ")) for t in prompt_templates], truncate=True).to(device)
        with torch.no_grad():
            rows.append(F.normalize(model.encode_text(tokens).float().mean(0), dim=0))
    return torch.stack(rows)


def _style_banks(model, class_names, prompt_templates, device):
    text_bank = []
    for name in tqdm(class_names, desc="template bank"):
        tokens = clip.tokenize([t.format(name.replace("_", " ")) for t in prompt_templates], truncate=True).to(device)
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
    def recenter(values, beta):
        probs = torch.tensor(gmm.predict_proba(pca.transform(values.detach().cpu().numpy() - mean_np)), device=values.device, dtype=values.dtype)
        return F.normalize(values - beta * (probs @ means), dim=-1)
    return recenter


def run_dataset(model, preprocess, data_root, dataset_config, device, batch_size=64):
    dataset_id, dataset, settings = load_evaluation_dataset(data_root, dataset_config)
    items = dataset.test
    if not items:
        raise ValueError(f"CoOp dataset has no usable test split: {dataset_id}")
    names = [item.classname for item in sorted(items, key=lambda item: item.label)]
    names = list(dict.fromkeys(names))
    prompt_templates = _dataset_templates(dataset_id)
    test_idx = list(range(len(items)))
    def images_at(ids):
        return _open_images(items, ids)
    text_proto = _text_proto(model, names, prompt_templates, device)
    if settings.style_lambda > 0:
        text_bank, style_proto, style_class_proto = _style_banks(model, names, prompt_templates, device)
    original, mirrored, labels = [], [], []
    for start in tqdm(range(0, len(test_idx), batch_size), desc=f"{dataset_id} test", leave=False):
        batch = images_at(test_idx[start:start+batch_size])
        original.append(_encode(model, preprocess, batch, device))
        mirrored.append(_encode(model, preprocess, [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in batch], device))
        labels.extend(items[i].label for i in test_idx[start:start+batch_size])
    original, mirrored = torch.cat(original), torch.cat(mirrored)
    avg = F.normalize((original + mirrored) / 2, dim=-1)
    if settings.n_clusters is None:
        rec = avg
    else:
        recenter = _fit_recenter(avg, seed=42, n_clusters=settings.n_clusters)
        rec = recenter(avg, settings.beta)
    rm_logits = rec @ text_proto.t() * 100
    rm_prob = F.softmax(rm_logits, dim=-1)
    if settings.style_lambda == 0:
        final_prob = rm_prob
    else:
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
        weight = settings.style_lambda * uncertainty[:, None]
        final_prob = (1 - weight) * rm_prob + weight * style_prob
    final_pred = final_prob.argmax(-1).cpu().numpy()
    labels = np.array(labels)
    return {"dataset": dataset_id, "final": float((final_pred == labels).mean() * 100)}


def run_final_evaluation(model_name, data_root, dataset_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    results = [run_dataset(
        model, preprocess, data_root, dataset_config, device,
    )]
    print("\n" + "=" * 30)
    print(f"{'dataset':<18}{'FINAL':>12}")
    for row in results:
        print(f"{row['dataset']:<18}{row['final']:>11.2f}%")
    return results
