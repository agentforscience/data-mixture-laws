"""Downstream evaluation of one pre-trained checkpoint.

  * knowledge: zero-shot multiple choice, scored by length-normalised answer log-likelihood
  * reasoning: fine-tune for a fixed, small budget and report exact-match accuracy on held-out items

Usage: python src/evaluate.py --name web_p0.50          (checkpoint from CKPT_DIR)
       python src/evaluate.py --name scratch --random_init --seed 0   (no-pre-training control)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from common import CKPT_DIR, DATA_DIR, RESULTS_DIR, retry_oom
from model import build
from tasks import REASONING, SCIQ_FT, load_knowledge, mc_prompt, sciq_finetune

FT_SEED = 1234  # fine-tuning data order / dropout-free; identical for every checkpoint


def load_model(name, random_init=False, seed=0):
    torch.manual_seed(seed)
    model = build("cuda")
    if not random_init:
        sd = torch.load(os.path.join(CKPT_DIR, name + ".pt"), map_location="cuda")
        model.load_state_dict({k: v.float() for k, v in sd.items()})
    return model


def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.cuda()


# ----------------------------------------------------------------------------- knowledge
@torch.no_grad()
def eval_mc(model, tok, items, bs=512):
    """Return per-item correctness (bool array). Score = mean log p(answer tokens | prompt)."""
    flat = []  # (item, choice, ids, n_prompt)
    for i, (q, choices, _) in enumerate(items):
        p = [0] + tok.encode(mc_prompt(q)).ids
        for j, c in enumerate(choices):
            a = tok.encode(" " + c.strip()).ids
            ids = (p + a)[-model.ctx:]
            flat.append((i, j, ids, len(ids) - len(a)))
    order = sorted(range(len(flat)), key=lambda k: len(flat[k][2]))  # length-bucketed batches
    scores = [dict() for _ in items]
    for b in range(0, len(order), bs):
        chunk = [flat[k] for k in order[b:b + bs]]
        def score():
            x = pad_batch([c[2] for c in chunk])
            mask = torch.zeros_like(x[:, 1:], dtype=torch.bool)
            for r, (_, _, ids, npr) in enumerate(chunk):
                mask[r, npr - 1:len(ids) - 1] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.head(model(x[:, :-1], hidden=True)[mask])  # answer positions only
            lp = -F.cross_entropy(logits.float(), x[:, 1:][mask], reduction="none")
            tot = torch.zeros(len(chunk), device=lp.device).index_add_(0, mask.nonzero()[:, 0], lp)
            return (tot / mask.sum(1)).tolist()
        for (i, j, _, _), v in zip(chunk, retry_oom(score)):
            scores[i][j] = v
    return np.array([max(s, key=s.get) == it[2] for s, it in zip(scores, items)])


# ----------------------------------------------------------------------------- reasoning
def encode_ex(tok, ex):
    out = []
    for prompt, ans in ex:
        p = [0] + tok.encode(prompt).ids
        a = tok.encode(ans).ids
        out.append((p + a, len(p)))
    return out


def finetune(model, tr, cfg):
    """Fine-tune in place for a fixed budget (loss on answer tokens only). Returns the last loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(FT_SEED)
    model.train()
    warm = cfg["steps"] // 10
    for step in range(cfg["steps"]):
        idx = torch.randint(0, len(tr), (cfg["bs"],), generator=g).tolist()
        for gr in opt.param_groups:  # linear warm-up then linear decay
            gr["lr"] = cfg["lr"] * min((step + 1) / warm, (cfg["steps"] - step) / (cfg["steps"] - warm))

        def step_fn():
            opt.zero_grad(set_to_none=True)
            logits, y, mask = answer_logits(model, [tr[i] for i in idx])
            loss = F.cross_entropy(logits, y)
            loss.backward()
            return loss
        loss = retry_oom(step_fn)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return float(loss.item())


def answer_logits(model, batch):
    """batch: [(ids, answer_start)] -> logits at answer positions (N, vocab), their targets (N,), mask (B, T)."""
    x = pad_batch([s for s, _ in batch])
    mask = torch.zeros_like(x[:, 1:], dtype=torch.bool)
    for i, (s, st) in enumerate(batch):
        mask[i, st - 1:len(s) - 1] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.head(model(x[:, :-1], hidden=True)[mask]).float()
    return logits, x[:, 1:][mask], mask


def finetune_eval(model, tok, cfg):
    """Fine-tune on a reasoning task, return (exact-match accuracy, answer-token accuracy, last loss).
    The pre-trained weights are restored afterwards."""
    train, test = cfg["fn"](cfg["n_train"], cfg["n_test"])
    tr, te = encode_ex(tok, train), encode_ex(tok, test)
    assert max(len(s) for s, _ in tr + te) <= model.ctx, max(len(s) for s, _ in tr + te)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    last = finetune(model, tr, cfg)
    em, tokacc = [], []
    with torch.no_grad():
        for b in range(0, len(te), 500):
            logits, y, mask = retry_oom(lambda: answer_logits(model, te[b:b + 500]))
            hit = logits.argmax(-1) == y
            full = torch.ones_like(mask)
            full[mask] = hit
            em += full.all(1).tolist()
            tokacc += hit.tolist()
    model.load_state_dict(init)
    return float(np.mean(em)), float(np.mean(tokacc)), last


def sciq_ft_eval(model, tok):
    """Closed-book QA: fine-tune on SciQ train answers, then multiple-choice accuracy on val+test."""
    train, items = sciq_finetune()
    init = {k: v.clone() for k, v in model.state_dict().items()}
    finetune(model, encode_ex(tok, train), SCIQ_FT)
    c = eval_mc(model, tok, items)
    model.load_state_dict(init)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="knowledge,reasoning")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))
    model = load_model(args.name, args.random_init, args.seed).eval()
    out_dir = os.path.join(RESULTS_DIR, "evals")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, args.name + ".json")
    res = json.load(open(path)) if os.path.exists(path) else {"name": args.name}
    t0 = time.time()
    if "knowledge" in args.only:
        items_correct = {}
        for task, items in load_knowledge().items():
            c = eval_mc(model, tok, items)
            items_correct[task] = c
            res[f"know/{task}"] = float(c.mean())
            res[f"know_n/{task}"] = len(c)
            res[f"know_chance/{task}"] = float(np.mean([1 / len(it[1]) for it in items]))
        c = sciq_ft_eval(model, tok)
        items_correct["sciq_ft"] = c
        res["know_ft/sciq"] = float(c.mean())
        np.savez_compressed(os.path.join(out_dir, args.name + "_items.npz"), **items_correct)
    if "reasoning" in args.only:
        for task, cfg in REASONING.items():
            acc, tokacc, last = finetune_eval(model, tok, cfg)
            res[f"reason/{task}"] = acc
            res[f"reason_tok/{task}"] = tokacc
            res[f"reason_ftloss/{task}"] = last
    res["eval_time_s"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
