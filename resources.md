# Resources

The resource-finding stage was skipped for this workspace, so everything below was gathered by the
experiment runner during the session. Nothing was pre-downloaded.

## Pre-training corpora (downloaded by `src/prepare_data.py`)

| Domain | Hugging Face dataset | Files used | Notes |
|---|---|---|---|
| web | `allenai/c4` (en) | `en/c4-train.00000-of-01024.json.gz` | generic web crawl, documents capped at 30k chars |
| wiki | `Salesforce/wikitext` (`wikitext-103-raw-v1`) | train (2 parquet) + official validation | whole articles |
| code | `codeparrot/codeparrot-clean` | `file-000000000001.json.gz` | GitHub Python files, capped at 30k chars |
| arxiv | `ccdv/arxiv-summarization` (`document`) | train shards 0–1, `article` field | paper bodies, capped at 30k chars |

Each domain is cut to the same pool of 128,865,134 training tokens (shared 16,384-token byte-level BPE
trained on a balanced 40 MB-per-domain sample). Held-out pools: 0.27M (wiki) to 1.8M tokens.
Statistics and a text sample per domain: `results/data_stats.json`.

## Downstream evaluation data (downloaded by `src/tasks.py`)

| Task | Dataset | Use |
|---|---|---|
| SciQ | `allenai/sciq` | zero-shot MC (all 13,679 items); fine-tune on train, test on val+test (2,000) |
| ARC-Easy / ARC-Challenge | `allenai/ai2_arc` | zero-shot MC (5,197 / 2,590 items) |
| OpenBookQA | `allenai/openbookqa` (main) | zero-shot MC (5,957) |
| CommonsenseQA | `tau/commonsense_qa` | zero-shot MC (train+val, 10,962) |
| ProofWriter | `tasksource/proofwriter` | fine-tune 3,000 / test 1,500, class balanced |
| ListOps | generated (`src/tasks.py`, seed 0) | fine-tune 4,000 / test 2,000 |
| Arithmetic | generated (`src/tasks.py`, seed 0) | fine-tune 4,000 / test 2,000 |

## Papers
See `literature_review.md` (abstract-level; PDFs were not downloaded).

## Code
All code was written for this study (`src/`). No external repositories were cloned.

## Storage note
`/data` was 100% full (≈6 GB free), so the virtual environment, token pools, HF cache and checkpoints
live under `~/.cache/neurico-scratch/<workspace>/` and are symlinked as `.venv` and `scratch/`.
