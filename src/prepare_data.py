"""Build the four-domain pre-training corpus.

Downloads raw text for each domain, trains one shared byte-level BPE tokenizer on a
balanced sample, and writes equal-sized token pools (uint16 memmaps) plus held-out
validation pools to DATA_DIR.

Domains
  web   : allenai/c4 (en)              -- generic web crawl
  wiki  : Salesforce/wikitext-103-raw  -- encyclopedic text
  code  : codeparrot/codeparrot-clean  -- GitHub Python files
  arxiv : ccdv/arxiv-summarization     -- scientific articles (body text)
"""
import gzip
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer

from common import DATA_DIR, DOMAINS, VOCAB_SIZE, EOS

RAW_CHARS = 620_000_000      # raw characters kept per domain (before tokenisation)
VAL_CHARS = 6_000_000        # held-out characters per domain
DOC_CAP = 30_000             # max characters per document (keeps document diversity)
TOK_SAMPLE_CHARS = 40_000_000  # per-domain sample used to train the tokenizer
SEED = 0


def dl(repo, fname):
    return hf_hub_download(repo, fname, repo_type="dataset")


def iter_web():
    for i in range(4):
        with gzip.open(dl("allenai/c4", f"en/c4-train.{i:05d}-of-01024.json.gz"), "rt") as f:
            for line in f:
                yield json.loads(line)["text"]


def iter_wiki_split(split):
    files = {"train": ["train-00000-of-00002", "train-00001-of-00002"],
             "validation": ["validation-00000-of-00001"]}[split]
    for fn in files:
        tab = pq.read_table(dl("Salesforce/wikitext", f"wikitext-103-raw-v1/{fn}.parquet"))
        doc = []
        for line in tab.column("text").to_pylist():
            # article boundaries are level-1 headings: " = Title = "
            s = line.strip()
            if s.startswith("= ") and not s.startswith("= =") and doc:
                yield "".join(doc)
                doc = []
            if s:
                doc.append(line)
        if doc:
            yield "".join(doc)


def iter_code():
    for i in range(1, 5):
        with gzip.open(dl("codeparrot/codeparrot-clean", f"file-{i:012d}.json.gz"), "rt") as f:
            for line in f:
                yield json.loads(line)["content"]


def iter_arxiv():
    for i in range(4):
        pf = pq.ParquetFile(dl("ccdv/arxiv-summarization", f"document/train-{i:05d}-of-00015.parquet"))
        for batch in pf.iter_batches(batch_size=512, columns=["article"]):
            for t in batch.column("article").to_pylist():
                yield t


def collect(it, n_chars, cap=DOC_CAP, long_docs=False):
    """Collect documents until n_chars characters are gathered."""
    docs, total = [], 0
    for t in it:
        if not t or len(t) < 200:
            continue
        if not long_docs:
            t = t[:cap]
        docs.append(t)
        total += len(t)
        if total >= n_chars:
            break
    return docs, total


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = random.Random(SEED)
    raw = {}
    for name, it in [("web", iter_web()), ("code", iter_code()), ("arxiv", iter_arxiv())]:
        docs, total = collect(it, RAW_CHARS + VAL_CHARS)
        rng.shuffle(docs)
        # hold out whole documents for validation
        val, n = [], 0
        while n < VAL_CHARS:
            d = docs.pop()
            val.append(d)
            n += len(d)
        raw[name] = (docs, val)
        print(name, "train docs", len(docs), "chars", total - n, "val docs", len(val), flush=True)
    # wiki: WikiText-103 has an official validation split; articles are long, keep them whole
    wtrain, tot = collect(iter_wiki_split("train"), RAW_CHARS, long_docs=True)
    wval, _ = collect(iter_wiki_split("validation"), VAL_CHARS, long_docs=True)
    rng.shuffle(wtrain)
    raw["wiki"] = (wtrain, wval)
    print("wiki train docs", len(wtrain), "chars", tot, "val docs", len(wval), flush=True)

    # ---- shared tokenizer on a balanced sample ----
    def sample_iter():
        for name in DOMAINS:
            n = 0
            for d in raw[name][0]:
                yield d
                n += len(d)
                if n > TOK_SAMPLE_CHARS:
                    break
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(sample_iter(), vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[EOS])
    tok.save(os.path.join(DATA_DIR, "tokenizer.json"))
    eos = tok.token_to_id(EOS)
    assert eos == 0 and tok.get_vocab_size() <= 65535

    def encode(docs):
        out = []
        for i in range(0, len(docs), 2000):
            for e in tok.encode_batch(docs[i:i + 2000]):
                out.append(np.asarray([eos] + e.ids, dtype=np.uint16))
        return np.concatenate(out)

    stats = {}
    pools = {}
    for name in DOMAINS:
        tr, va = raw[name]
        chars = sum(map(len, tr))
        pools[name] = encode(tr)
        v = encode(va)
        np.save(os.path.join(DATA_DIR, f"{name}_val.npy"), v)
        stats[name] = {"train_docs": len(tr), "train_chars": chars, "train_tokens_raw": int(len(pools[name])),
                       "chars_per_token": chars / len(pools[name]), "val_tokens": int(len(v)),
                       "example": tr[0][:400]}
        print(name, {k: v for k, v in stats[name].items() if k != "example"}, flush=True)
    # equalise pool sizes so every domain offers the same number of tokens
    n = min(len(p) for p in pools.values())
    for name in DOMAINS:
        np.save(os.path.join(DATA_DIR, f"{name}_train.npy"), pools[name][:n])
        stats[name]["train_tokens"] = int(n)
    json.dump(stats, open("results/data_stats.json", "w"), indent=2)
    print("pool tokens per domain:", n)


if __name__ == "__main__":
    sys.exit(main())
