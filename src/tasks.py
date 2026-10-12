"""Downstream task data.

Knowledge-intensive tasks (zero-shot multiple choice): SciQ, ARC-Easy, ARC-Challenge, OpenBookQA,
CommonsenseQA.  All official splits are pooled because nothing is trained on them.

Reasoning tasks (fixed-budget fine-tuning): ProofWriter (real data), ListOps and multi-digit
arithmetic (procedurally generated with fixed seeds, as in the original benchmarks).
"""
import random

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def _pq(repo, fname):
    return pq.read_table(hf_hub_download(repo, fname, repo_type="dataset")).to_pandas()


# ----------------------------------------------------------------------------- knowledge (MC)
def _choices_fmt(df, qcol):
    items = []
    for r in df.itertuples():
        ch = getattr(r, "choices")
        labels, texts = list(ch["label"]), list(ch["text"])
        if r.answerKey not in labels:
            continue
        items.append((getattr(r, qcol), texts, labels.index(r.answerKey)))
    return items


def load_knowledge():
    """Returns {task: [(question, [choices], gold_index), ...]}."""
    rng = random.Random(0)
    out = {}
    sciq = []
    for sp in ["train", "validation", "test"]:
        for r in _pq("allenai/sciq", f"data/{sp}-00000-of-00001.parquet").itertuples():
            ch = [r.correct_answer, r.distractor1, r.distractor2, r.distractor3]
            order = list(range(4))
            rng.shuffle(order)
            sciq.append((r.question, [ch[i] for i in order], order.index(0)))
    out["sciq"] = sciq
    for cfg, key in [("ARC-Easy", "arc_easy"), ("ARC-Challenge", "arc_challenge")]:
        out[key] = sum((_choices_fmt(_pq("allenai/ai2_arc", f"{cfg}/{sp}-00000-of-00001.parquet"), "question")
                        for sp in ["train", "validation", "test"]), [])
    out["openbookqa"] = sum((_choices_fmt(_pq("allenai/openbookqa", f"main/{sp}-00000-of-00001.parquet"), "question_stem")
                             for sp in ["train", "validation", "test"]), [])
    # CommonsenseQA test labels are hidden -> train + validation
    out["commonsenseqa"] = sum((_choices_fmt(_pq("tau/commonsense_qa", f"data/{sp}-00000-of-00001.parquet"), "question")
                                for sp in ["train", "validation"]), [])
    return out


def sciq_finetune():
    """Closed-book QA fine-tuning data: SciQ train as (prompt, answer) pairs; validation+test as MC items."""
    rng = random.Random(1)
    tr = [(mc_prompt(r.question), " " + r.correct_answer.strip())
          for r in _pq("allenai/sciq", "data/train-00000-of-00001.parquet").itertuples()]
    te = []
    for sp in ["validation", "test"]:
        for r in _pq("allenai/sciq", f"data/{sp}-00000-of-00001.parquet").itertuples():
            ch = [r.correct_answer, r.distractor1, r.distractor2, r.distractor3]
            order = list(range(4))
            rng.shuffle(order)
            te.append((r.question, [ch[i] for i in order], order.index(0)))
    return tr, te


def mc_prompt(q):
    return f"Question: {q.strip()}\nAnswer:"


# ----------------------------------------------------------------------------- reasoning (fine-tune)
def proofwriter(n_train, n_test, max_chars=700, seed=0):
    """Deductive reasoning over natural-language rules; 3-way True / False / Unknown, class balanced.
    Only questions that need at least one inference step (QDep >= 1) or are Unknown are kept so
    that the task cannot be solved by string lookup alone."""
    def prep(df, n, rng):
        df = df[(df.theory.str.len() <= max_chars) & ~df.config.str.contains("NatLang")]
        df = df[(df.QDep >= 1) | (df.answer == "Unknown")]
        per = n // 3
        ex = []
        for a in ["True", "False", "Unknown"]:
            sub = df[df.answer == a]
            idx = rng.sample(range(len(sub)), per)
            ex += [(f"{r.theory}\nQuestion: {r.question}\nAnswer:", " " + r.answer) for r in sub.iloc[idx].itertuples()]
        rng.shuffle(ex)
        return ex
    rng = random.Random(seed)
    tr = _pq("tasksource/proofwriter", "data/train-00000-of-00002-6176cf3d78f48858.parquet")
    te = _pq("tasksource/proofwriter", "data/test-00000-of-00001-3e27b013c60e12d8.parquet")
    return prep(tr, n_train, rng), prep(te, n_test, rng)


def _listops_expr(rng, depth, max_args=4):
    if depth == 0 or rng.random() < 0.25:
        v = rng.randrange(10)
        return str(v), v
    op = rng.choice(["MAX", "MIN", "MED", "SM"])
    args = [_listops_expr(rng, depth - 1, max_args) for _ in range(rng.randint(2, max_args))]
    vals = [a[1] for a in args]
    v = {"MAX": max(vals), "MIN": min(vals), "MED": sorted(vals)[len(vals) // 2], "SM": sum(vals) % 10}[op]
    return "[ " + op + " " + " ".join(a[0] for a in args) + " ]", v


def listops(n_train, n_test, seed=0):
    """ListOps (Nangia & Bowman 2018): nested MAX/MIN/MED/SUM-mod-10 expressions, 10-way answer."""
    rng = random.Random(seed)
    seen, ex = set(), []
    while len(ex) < n_train + n_test:
        s, v = _listops_expr(rng, 3)
        if s.startswith("[") and s not in seen and len(s) < 160:
            seen.add(s)
            ex.append((s + " =", " " + str(v)))
    return ex[:n_train], ex[n_train:]


def arithmetic(n_train, n_test, seed=0):
    """3-digit addition / subtraction, digits space-separated; exact match on the full answer."""
    rng = random.Random(seed)
    seen, ex = set(), []
    while len(ex) < n_train + n_test:
        a, b, op = rng.randrange(100, 1000), rng.randrange(100, 1000), rng.choice("+-")
        if op == "-" and b > a:
            a, b = b, a
        if (a, b, op) in seen:
            continue
        seen.add((a, b, op))
        r = a + b if op == "+" else a - b
        sp = lambda n: " ".join(str(n))
        ex.append((f"{sp(a)} {op} {sp(b)} =", " " + sp(r)))
    return ex[:n_train], ex[n_train:]


# fixed fine-tuning budgets (calibrated once on the uniform-mixture pilot and a random-init control)
REASONING = {
    "proofwriter": dict(fn=proofwriter, n_train=3000, n_test=1500, steps=500, bs=32, lr=3e-4),
    "listops": dict(fn=listops, n_train=4000, n_test=2000, steps=500, bs=64, lr=3e-4),
    "arithmetic": dict(fn=arithmetic, n_train=4000, n_test=2000, steps=1500, bs=64, lr=3e-4),
}
SCIQ_FT = dict(steps=400, bs=64, lr=3e-4)
