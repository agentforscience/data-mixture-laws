"""Shared constants and paths."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Large artefacts (token pools, checkpoints) live on a scratch disk symlinked at ./scratch
SCRATCH = os.environ.get("SCRATCH_DIR", os.path.join(ROOT, "scratch"))
DATA_DIR = os.path.join(SCRATCH, "data")
CKPT_DIR = os.path.join(SCRATCH, "ckpt")
RESULTS_DIR = os.path.join(ROOT, "results")

DOMAINS = ["web", "wiki", "code", "arxiv"]
VOCAB_SIZE = 16384
EOS = "<|endoftext|>"


def retry_oom(fn, wait=20, max_wait=600):
    """Run fn(); on CUDA OOM (a neighbour on the shared GPU grew) free the cache, wait and retry.

    fn must be safe to re-execute from scratch (no optimiser step may have happened)."""
    import time
    import torch
    waited = 0
    while True:
        try:
            return fn()
        except torch.OutOfMemoryError:
            if waited >= max_wait:
                raise
            torch.cuda.empty_cache()
            print(f"[retry_oom] CUDA OOM, waiting {wait}s (total {waited}s)", flush=True)
            time.sleep(wait)
            waited += wait
