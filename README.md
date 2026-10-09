# chairwomans-test

Zero-shot CLIP classification over cached image features, using domain-conditioned
prompt routing combined with each dataset's own CLIP prompt templates.

For every test image, the model estimates which visual "domain" (photo, cartoon,
sketch, sculpture, ...) it most resembles from a fixed set of hand-written domain
phrases, then blends per-domain class text prototypes by that estimate. Each
prototype is built from `(domain phrase, dataset template)` pairs, e.g.
`"a cartoon image, a photo of a {}, a type of aircraft."`. No training or
fine-tuning is involved — it's zero-shot CLIP with input-conditional prompt routing.

Every one of a dataset's own templates is combined with every domain phrase.
ImageNet-family datasets use CLIP's 80-template ensemble, so that's 1600
prompts per class -- currently only run with `--datasets imagenet-only` or
`all`, since it's much slower than the no-imagenet datasets.

## Setup

```bash
pip install -r requirements.txt
```

Datasets and their official splits must already be prepared under a data root
(see `DATASETS.md`), and CLIP image feature caches must already exist under
`<data-root>/cache/<backbone>/<dataset>/raw.pt` (built with `scripts/cache_features.py`
or the `scripts/cache_all_features.ps1` wrapper).

## Run

```bash
python run_final_evaluation.py --data-root ./data --backbone ViT-B/16 --datasets all
```

Options:
- `--datasets` — comma-separated dataset names, or a preset: `all` (default),
  `no-imagenet`, `imagenet-only`
- `--backbone` — `ViT-B/16` (default) / `ViT-B/32` / `ViT-L/14`
- `--dry-run` — just validate the caches exist, without loading CLIP

Or via the wrapper scripts: `scripts/run_final_evaluation.sh` / `.ps1`.

## Layout

- `run_final_evaluation.py` — CLI entry point
- `utils/final_evaluation.py` — the pipeline itself (cache loading, domain
  routing, prototype building, evaluation)
- `utils/domain_prompts.py` — hand-written domain-style phrases
- `utils/templates.py` — per-dataset CLIP prompt templates
- `utils/dataset_setup.py`, `datasets/`, `configs/*.yaml`, `scripts/cache_features.py`
  — dataset preparation and feature-cache generation (upstream of this pipeline)
