"""Experiment design + a small GPU scheduler for a *shared* machine.

  python src/run_all.py --stage train    # pre-train every mixture in the design
  python src/run_all.py --stage eval     # downstream evaluation of every checkpoint
  python src/run_all.py --list           # print the design

The scheduler launches at most one of our jobs per GPU, and only on GPUs that currently have
MIN_FREE_MB of free memory (other users' jobs are never touched). Jobs that crash (e.g. OOM
because a neighbour grew) are re-queued. Finished jobs are skipped, so the script is resumable.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from common import CKPT_DIR, DOMAINS, RESULTS_DIR, ROOT

# Priority tiers: the machine is shared, so the design is ordered such that a truncated run is
# still analysable (coarse sweeps -> noise floor -> random mixtures + web replicates -> fine grid).
TIER_A = [1.0, 0.0, 0.5, 0.2]
TIER_B = [0.05, 0.75]
TIER_D = [0.1, 0.35, 0.02]
SWEEP_P = TIER_A + TIER_B + TIER_D
N_RANDOM = 16
REPL_SEEDS = [1, 2]
WEB_REPL_P = [0.5, 1.0, 0.75]
MIN_FREE_MB = 7000
SECOND_JOB_FREE_MB = 16000
MAX_TRIES = 4


def sweep_mix(dom, p):
    """Domain `dom` gets share p; the other three share the remainder equally."""
    m = np.full(len(DOMAINS), (1 - p) / (len(DOMAINS) - 1))
    m[DOMAINS.index(dom)] = p
    return m


def design():
    """Ordered list of runs: dict(name, mix, seed, group). Earlier = higher priority."""
    def sw(d, p, s=0):
        return dict(name=f"{d}_p{p:.2f}_s{s}", mix=sweep_mix(d, p), seed=s, group=f"sweep_{d}")
    sweeps = lambda ps, doms=DOMAINS: [sw(d, p) for p in ps for d in doms]
    uni = lambda s: dict(name=f"uniform_s{s}", mix=np.full(4, 0.25), seed=s, group="uniform")
    rand = np.random.default_rng(2026).dirichlet(np.ones(4), size=N_RANDOM)
    rnd = [dict(name=f"rand{i:02d}_s0", mix=rand[i], seed=0, group="random") for i in range(N_RANDOM)]
    others = [d for d in DOMAINS if d != "web"]
    runs = [uni(0)] + sweeps(TIER_A)                                        # A: coarse sweeps
    runs += [uni(1), uni(2), sw("web", 0.75)] + sweeps([0.05])              # B: noise floor, low end
    runs += [sw("web", p, s) for s in REPL_SEEDS for p in WEB_REPL_P]       #    web-sweep replicates (H4)
    runs += rnd[:8]                                                         # C: random mixtures (H3)
    runs += sweeps([0.75], others) + rnd[8:] + sweeps(TIER_D)               # D: finer grid
    runs += [sw("web", 0.2, s) for s in REPL_SEEDS]
    # E: extra seed replicates of the extremes (added once it was clear the machine had freed up;
    #    chosen before looking at the full results): every leave-one-out and pure-domain model,
    #    plus two interior points of the code sweep (H3).
    runs += [sw(d, 0.0, s) for s in REPL_SEEDS for d in DOMAINS]
    runs += [sw(d, 1.0, s) for s in REPL_SEEDS for d in others]
    runs += [sw("code", p, s) for s in REPL_SEEDS for p in (0.2, 0.5)]
    return runs


def my_gpus(script):
    """{gpu index: number of *our* processes running `script` on it}."""
    q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader"], capture_output=True, text=True).stdout
    uuid2idx = {l.split(", ")[1]: int(l.split(", ")[0]) for l in q("--query-gpu=index,uuid").strip().splitlines()}
    out = {}
    for l in q("--query-compute-apps=gpu_uuid,pid").strip().splitlines():
        u, pid = l.split(", ")
        try:
            if script in open(f"/proc/{pid}/cmdline").read().replace("\0", " "):
                out[uuid2idx[u]] = out.get(uuid2idx[u], 0) + 1
        except OSError:
            pass
    return out


def is_running(script, name):
    return subprocess.run(["pgrep", "-f", f"{script} --name {name}( |$)"], capture_output=True).returncode == 0


def free_gpus(counts):
    """GPUs we may start a job on: none of our jobs there and MIN_FREE_MB free, or one of our
    jobs there and still SECOND_JOB_FREE_MB free (an otherwise lightly used GPU)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    res = []
    for line in out.strip().splitlines():
        i, tot, used = [int(x) for x in line.split(",")]
        need = {0: MIN_FREE_MB, 1: SECOND_JOB_FREE_MB}.get(counts.get(i, 0))
        if need is not None and tot - used >= need:
            res.append((tot - used, i))
    return [i for _, i in sorted(res, reverse=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval"], default="train")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N runs of the design")
    args = ap.parse_args()
    runs = design()[: args.limit]
    if args.list:
        for r in runs:
            print(r["name"], np.round(r["mix"], 3).tolist(), r["group"])
        print(len(runs), "runs")
        return
    json.dump([{**r, "mix": r["mix"].tolist()} for r in design()], open(os.path.join(RESULTS_DIR, "design.json"), "w"), indent=1)
    py, src = sys.executable, os.path.join(ROOT, "src")
    queue, known = [], set()

    def refill():
        """(Re)scan for work. The eval stage is re-scanned continuously so that it follows a
        concurrently running train stage; it stops once logs/TRAIN_FINISHED exists."""
        for r in runs:
            if r["name"] in known:
                continue
            if args.stage == "train":
                done = os.path.exists(os.path.join(RESULTS_DIR, "pretrain", r["name"] + ".json"))
                cmd = [py, f"{src}/train.py", "--name", r["name"], "--mix", ",".join(f"{x:.6f}" for x in r["mix"]), "--seed", str(r["seed"])]
            else:
                if not os.path.exists(os.path.join(RESULTS_DIR, "pretrain", r["name"] + ".json")):
                    continue  # not trained yet (the checkpoint is written right after this file)
                path = os.path.join(RESULTS_DIR, "evals", r["name"] + ".json")
                done = os.path.exists(path) and "reason/arithmetic" in json.load(open(path))
                cmd = [py, f"{src}/evaluate.py", "--name", r["name"]]
            known.add(r["name"])
            if not done and is_running(cmd[1], r["name"]):
                print("already running, skipped:", r["name"], flush=True)
                continue
            if not done:
                queue.append((r["name"], cmd, 0))

    refill()
    flag = os.path.join(ROOT, "logs", "TRAIN_FINISHED")
    following = lambda: args.stage == "eval" and not os.path.exists(flag)
    print(f"{len(queue)} jobs queued", flush=True)
    log_dir = os.path.join(ROOT, "logs", "runs")
    os.makedirs(log_dir, exist_ok=True)
    running = {}  # name -> (gpu, cmd, tries, proc, t0)
    env0 = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    script = f"{src}/{'train' if args.stage == 'train' else 'evaluate'}.py"
    while queue or running or following():
        if args.stage == "eval":
            refill()
        for name, (gpu, cmd, tries, proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[name]
            print(f"[{time.strftime('%H:%M:%S')}] {name} on gpu{gpu} rc={rc} {time.time()-t0:.0f}s; {len(queue)} left", flush=True)
            if rc != 0 and tries + 1 < MAX_TRIES:
                queue.append((name, cmd, tries + 1))
        if queue:
            counts = my_gpus(script)
            for gpu, *_ in running.values():  # jobs we just launched may not be visible to nvidia-smi yet
                counts[gpu] = max(counts.get(gpu, 0), sum(1 for g, *_ in running.values() if g == gpu))
            cand = free_gpus(counts)
            if cand:  # one launch per tick, on the emptiest eligible GPU
                gpu = cand[0]
                name, cmd, tries = queue.pop(0)
                f = open(os.path.join(log_dir, f"{args.stage}_{name}.log"), "a")
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**env0, "CUDA_VISIBLE_DEVICES": str(gpu)})
                running[name] = (gpu, cmd, tries, proc, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] start {name} on gpu{gpu}", flush=True)
                time.sleep(25)  # let the job allocate before re-reading free memory
        time.sleep(10)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
