"""Pre-train one small GPT on a given domain mixture with a fixed token budget.

Usage: python src/train.py --name web_p0.50 --mix 0.5,0.1667,0.1667,0.1667 --seed 0
Mixture weights follow the order in common.DOMAINS and are *token* proportions:
every training sequence is drawn from domain d with probability mix[d].
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CKPT_DIR, DATA_DIR, DOMAINS, RESULTS_DIR, retry_oom
from model import CONFIG, build, n_params

HP = dict(batch=64, lr=1e-3, wd=0.1, warmup_frac=0.03, final_lr_frac=0.1, betas=(0.9, 0.95), clip=1.0)


def load_pool(name, split, device):
    a = np.load(os.path.join(DATA_DIR, f"{name}_{split}.npy"), mmap_mode="r")
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.int16)).to(device)  # vocab < 32768


@torch.no_grad()
def val_loss(model, pool, ctx, max_tokens=1_000_000, bs=64):
    """Mean next-token cross-entropy (nats) on non-overlapping windows of a held-out pool."""
    n = min((len(pool) - 1) // ctx, max_tokens // ctx)
    tot, cnt = 0.0, 0
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=pool.device)[:, None] * ctx + torch.arange(ctx + 1, device=pool.device)
        x = pool[idx].long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        tot += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item()
        cnt += x[:, 1:].numel()
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mix", required=True, help="comma-separated weights, order: " + ",".join(DOMAINS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=float, default=60e6, help="total pre-training token budget")
    ap.add_argument("--no_save", action="store_true")
    args = ap.parse_args()

    mix = np.array([float(x) for x in args.mix.split(",")], dtype=np.float64)
    assert len(mix) == len(DOMAINS) and mix.min() >= 0
    mix = mix / mix.sum()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    ctx, B = CONFIG["ctx"], HP["batch"]
    pools = [load_pool(d, "train", dev) for d in DOMAINS]
    vals = [load_pool(d, "val", dev) for d in DOMAINS]

    model = build(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": HP["wd"]}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=HP["lr"], betas=HP["betas"], fused=True)
    cmodel = torch.compile(model)
    steps = int(args.tokens // (B * ctx))
    warm = max(1, int(HP["warmup_frac"] * steps))

    def lr_at(s):
        if s < warm:
            return HP["lr"] * (s + 1) / warm
        prog = (s - warm) / max(1, steps - warm)
        return HP["lr"] * (HP["final_lr_frac"] + (1 - HP["final_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))

    g = torch.Generator(device=dev).manual_seed(args.seed)
    probs = torch.tensor(mix, device=dev, dtype=torch.float32)
    offs = torch.arange(ctx + 1, device=dev)
    seen = np.zeros(len(DOMAINS), dtype=np.int64)
    t0, log = time.time(), []
    for s in range(steps):
        # sample the domain of each sequence, then a random window from that domain's pool
        dom = torch.multinomial(probs, B, replacement=True, generator=g)
        x = torch.empty(B, ctx + 1, dtype=torch.long, device=dev)
        for d in range(len(DOMAINS)):
            m = dom == d
            k = int(m.sum())
            if k:
                st = torch.randint(0, len(pools[d]) - ctx - 1, (k,), device=dev, generator=g)
                x[m] = pools[d][st[:, None] + offs].long()
                seen[d] += k * ctx
        for gr in opt.param_groups:
            gr["lr"] = lr_at(s)

        def step_fn():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = cmodel(x[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            loss.backward()
            return loss
        loss = retry_oom(step_fn)  # shared GPUs: wait out a neighbour's memory spike instead of dying
        torch.nn.utils.clip_grad_norm_(model.parameters(), HP["clip"])
        opt.step()
        if s % 200 == 0 or s == steps - 1:
            log.append((s, loss.item()))
            print(f"{args.name} step {s}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    train_time = time.time() - t0

    model.eval()
    vl = {d: retry_oom(lambda: val_loss(model, v, ctx)) for d, v in zip(DOMAINS, vals)}
    print(args.name, "val", vl, flush=True)
    out = dict(name=args.name, mix=dict(zip(DOMAINS, mix.tolist())), seed=args.seed, tokens=steps * B * ctx,
               tokens_seen=dict(zip(DOMAINS, seen.tolist())), steps=steps, val_loss=vl, train_log=log,
               train_time_s=train_time, n_params=n_params(model), hp={**HP, **CONFIG})
    if not args.no_save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(CKPT_DIR, args.name + ".pt"))

    os.makedirs(os.path.join(RESULTS_DIR, "pretrain"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "pretrain", args.name + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
