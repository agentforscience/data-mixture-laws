# Data Mixture Laws: Predicting Downstream Performance from Pre-Training Corpus Composition

A controlled dose–response study of pre-training data mixtures. 81 small GPTs (17M parameters, 60M
tokens each) were trained from scratch on mixtures of web, Wikipedia, code and arXiv text, and evaluated
on per-domain loss, knowledge QA and reasoning probes, to test whether downstream performance is
log-linear in domain share, saturates at 20%, is driven by code (reasoning), or is hurt by >50% web.

Full write-up: **[REPORT.md](REPORT.md)**. Plan and deviation log: [planning.md](planning.md).

## Key findings (small-scale regime only)

- **Loss is log-linear in a domain's share** (R² 0.98–0.9996 vs 0.67–0.87 for linear), **with no
  saturation at 20%**: loss keeps falling 0.36–0.55 nats from 20% to 100% share. Share 0 is a cliff — the
  first 2% of a domain gives 43–67% of its total benefit.
- **Downstream accuracy is nearly flat across mixtures.** Variation across 53 mixtures is 1.0–2.4× the
  seed noise; no downstream score follows a log-linear dose–response.
- **Code is not a strong driver of reasoning scores**: r = −0.10 (CI −0.55 to 0.57) on random mixtures,
  0.01 on all mixtures, against a hypothesised r > 0.7. Only ListOps shows a small positive association
  with code (r = 0.39).
- **Web above 50% does not measurably hurt knowledge tasks**; 75% web equals 25–50% web. What hurts is a
  corpus with no natural language: pure code costs 7.4 points of fine-tuned SciQ accuracy.
- **A log-share mixing law predicts held-out mixtures' loss** (r = 0.94–0.9996) but downstream scores are
  not predictable from the mixture beyond noise.

Caveats: tiny models, weak downstream instruments (zero-shot QA 2–3 points above chance), 1–3 seeds per
mixture. The downstream nulls bound effects at about 1–3 points here; they do not transfer to LLM scale.

## Reproduce

```bash
uv venv && source .venv/bin/activate && uv sync      # or: uv pip install -r requirements.txt
export HF_HOME=$PWD/scratch/hf                        # any directory with ~3 GB free
mkdir -p scratch results figures logs                 # scratch/ holds corpora and checkpoints (~5 GB)

python src/prepare_data.py                 # download 4 corpora, train tokenizer, build token pools (~10 min)
python src/run_all.py --list               # show the 81-run design
python src/run_all.py --stage train        # pre-train every mixture (resumable; ~11 GPU-hours on an A6000)
touch logs/TRAIN_FINISHED
python src/run_all.py --stage eval         # downstream evaluation of every checkpoint (~4 GPU-hours)
python src/evaluate.py --name scratch_s0 --random_init   # no-pre-training control
python src/analyze.py                      # tables -> results/analysis.md, figures -> figures/
```

`run_all.py` schedules one job per GPU that has free memory and re-queues jobs that fail, so it can share
a machine. Seeds are fixed (`--seed` for pre-training, `FT_SEED` for fine-tuning); evaluation of a given
checkpoint is exactly reproducible, pre-training is reproducible up to GPU non-determinism.

## Files

| Path | Content |
|---|---|
| `REPORT.md` | full report with results, figures and limitations |
| `planning.md` | pre-registered plan, hypotheses, deviation log |
| `literature_review.md`, `resources.md` | prior work and data sources (compiled in-session) |
| `src/prepare_data.py` | corpora, tokenizer, token pools |
| `src/model.py`, `src/train.py` | GPT and fixed-budget mixture pre-training |
| `src/tasks.py`, `src/evaluate.py` | downstream tasks; zero-shot MC scoring and fine-tuning probes |
| `src/run_all.py` | mixture design and GPU scheduler |
| `src/analyze.py` | statistics, mixing-law fits, figures |
| `results/runs.csv` | one row per run with every metric |
| `results/analysis.md` | all tables and tests |
| `results/pretrain/`, `results/evals/` | raw per-run outputs |
| `figures/` | fig1–fig5 |
