# Improving Zero-Shot Image Classification of Vision-Language Models via Domain-Weighted Prompt Ensembling

Zero-shot CLIP classification over image features, using domain-conditioned
prompt routing combined with each dataset's own CLIP prompt templates.

For every test image, the model estimates which visual "domain" (photo, cartoon,
sketch, sculpture, ...) it most resembles from a fixed set of hand-written domain
phrases, then blends per-domain class text prototypes by that estimate. Each
prototype is built from `(domain phrase, dataset template)` pairs, e.g.
`"a cartoon image, a photo of a {}, a type of aircraft."`. No training or
fine-tuning is involved — it's zero-shot CLIP with input-conditional prompt routing.

## Setup

```bash
pip install -r requirements.txt
```

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
