"""Small decoder-only transformer (GPT-2 style, pre-LN, tied embeddings)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(n_layer=6, n_head=6, d_model=384, ctx=256, vocab=16384)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, n_layer, n_head, d_model, ctx, vocab):
        super().__init__()
        self.ctx = ctx
        self.wte = nn.Embedding(vocab, d_model)
        self.wpe = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init)
        for b in self.blocks:  # scaled init of residual projections
            for m in (b.proj, b.out):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, hidden=False):
        """idx: (B, T) long -> logits (B, T, vocab); with hidden=True return final hidden states
        instead, so that callers can apply `self.head` only at the positions they need."""
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        return x if hidden else self.head(x)


def build(device="cuda"):
    return GPT(**CONFIG).to(device)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
