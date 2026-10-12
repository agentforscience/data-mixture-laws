# Planning: Data Mixture Laws

*Written before any model was trained. Deviations are logged at the bottom.*

## Motivation & Novelty Assessment

### Why This Research Matters
The composition of a pre-training corpus (how much web text, code, encyclopedic text, papers…) is chosen
mostly by heuristics, yet it is one of the few levers that changes what a model can do at a fixed compute
budget. If downstream performance followed a simple, predictable law in the domain proportions, mixtures
could be chosen from a handful of small runs instead of trial and error at scale.

### Gap in Existing Work
- *Data Mixing Laws* (Ye et al., ICLR 2025) and *Scaling Laws for Optimal Data Mixtures* (Shukor et al.,
  NeurIPS 2025) fit functional forms to **validation loss**, not to downstream task accuracy.
- *RegMix* (Liu et al., ICLR 2025) regresses a loss target on mixtures and reports which domains correlate
  with downstream performance (web, not Wikipedia), but does not characterise the *shape* of the
  proportion→performance curve (log-linear? where does it saturate?).
- *To Code, or Not To Code?* (Aryabumi et al., ICLR 2025) shows code helps reasoning/knowledge tasks, but
  with a handful of mixtures, not a dose–response curve.
Nobody has (to our knowledge) jointly tested the specific shape claims in our hypothesis: log-linearity,
a ~20% saturation point, a code→reasoning correlation above 0.7, and a web>50% penalty on knowledge tasks.

### Our Novel Contribution
A controlled dose–response study: ~60 small GPTs trained from scratch at a fixed token budget on
systematically varied 4-domain mixtures, each evaluated on (a) per-domain held-out loss, (b) zero-shot
knowledge QA, and (c) fine-tuning transfer to reasoning tasks, with seed replicates to quantify noise and
pre-specified statistical tests for each of the hypothesis' four claims.

### Experiment Justification
- **E1 – single-domain sweeps** (each domain's share p ∈ {0,…,1}, others split the rest equally): the only
  way to see the *shape* of performance vs. proportion and to locate a saturation point (H1, H2, H4).
- **E2 – random simplex mixtures** (Dirichlet): an unbiased design for correlations and for fitting /
  cross-validating a mixture regression, so that domain effects are not confounded with the sweep
  structure (H3, and predictive "law" check).
- **E3 – seed replicates**: without a noise floor no curve-shape comparison or "degradation" claim is
  interpretable.
- **E4 – no-pre-training control** for the fine-tuned reasoning probes: establishes whether there is any
  transfer at all to measure.

## Research Question
At fixed model size and token budget, how does downstream performance depend on the proportion of each
domain in the pre-training mixture — is it log-linear with saturation near 20%, is code the strongest
driver of reasoning performance, and does >50% web text hurt knowledge-intensive tasks?

## Hypothesis Decomposition
- **H1 (log-linear):** along a single-domain sweep, performance is better described by `a + b·log(p)` than
  by `a + b·p`. Test: leave-one-out CV error and AICc of linear vs log-linear vs exponential
  (Ye et al. form) vs log-linear-with-plateau fits, for p > 0.
- **H2 (saturation at ~20%):** most of the gain from a domain is obtained by p = 0.2. Test: fraction of the
  total p=0→p=best improvement achieved at p=0.2; breakpoint of a fitted hinge model with bootstrap CI;
  whether performance beyond 0.2 differs from the 0.2 level by more than seed noise.
- **H3 (code → reasoning, r > 0.7):** across the random mixtures, Pearson/Spearman correlation between
  code share and the reasoning score exceeds 0.7 and exceeds that of every other domain. Also reported:
  mixture-regression (Scheffé) coefficients, since shares on a simplex are mutually negatively correlated.
- **H4 (web > 50% hurts knowledge):** in the web sweep, knowledge score at p ∈ {0.75, 1.0} is lower than at
  p ≤ 0.5 by more than seed noise (Welch t-test on seed replicates; monotone-trend test).

Independent variable: mixture vector over {web, wiki, code, arxiv}. Dependent variables below.
Controlled: architecture, tokenizer, token budget, optimiser, schedule, eval protocol.

## Proposed Methodology

### Approach
Training real language models from scratch is the only direct test; the price is scale
(34M-parameter GPT, 100M tokens/run). All conclusions are therefore about this small-scale regime and are
stated as such. No LLM APIs are needed.

- **Domains:** web = C4-en; wiki = WikiText-103; code = GitHub Python (codeparrot-clean);
  arxiv = scientific article bodies (ccdv/arxiv-summarization). Equal-size token pools; shared 16k BPE
  trained on a balanced sample.
- **Model:** 8-layer, 512-wide GPT, context 512; AdamW, cosine schedule, bf16 autocast.
- **Mixtures:** (E1) for each domain p ∈ {0, .02, .05, .1, .2, .35, .5, .75, 1}, remainder split equally;
  (E2) ~20 Dirichlet(1,1,1,1) draws; (E3) 3 seeds for the uniform mix and for the web-sweep points
  {.2, .5, .75, 1} (needed for H4), plus extra seeds on the code sweep if budget allows.

### Evaluation Metrics
1. **Per-domain held-out loss** (nats/token) — the quantity prior mixing laws model.
2. **Knowledge-intensive tasks, zero-shot multiple choice** scored by length-normalised answer
   log-likelihood: SciQ, ARC-Easy, ARC-Challenge, OpenBookQA, CommonsenseQA. Tasks on which the *uniform*
   pilot model is not above chance are reported but excluded from the composite (decided on the pilot
   only, before any mixture comparison).
3. **Reasoning tasks, fixed-budget fine-tuning transfer** (tiny models cannot do zero-shot reasoning):
   ProofWriter (deductive, natural language), ListOps (hierarchical symbolic), multi-digit arithmetic.
   Accuracy after an identical, small fine-tuning budget. Control: randomly initialised model.
4. **Composites:** mean accuracy over tasks within a family (and z-scored mean as a robustness check).

### Baselines
Uniform mixture (0.25 each); single-domain models (p = 1); leave-one-domain-out models (p = 0);
no-pre-training control for reasoning probes; chance level for MC tasks. For curve shape: linear fit as
the null against log-linear.

### Statistical Analysis Plan
α = 0.05. Curve-shape: LOO-CV RMSE and AICc per (domain, metric). Correlations: Pearson and Spearman with
bootstrap 95% CIs over runs; Holm correction across the 4 domains. Degradation tests: Welch t-test across
seeds, effect size as difference in accuracy points with CI; item-level paired bootstrap as a secondary
check. Noise floor: SD across seeds for every metric.

## Expected Outcomes
Support: log fits beat linear fits on most (domain, metric) pairs; ≥ ~80% of gains realised by p = 0.2;
code share has the largest positive correlation with reasoning (> 0.7); knowledge score drops beyond 50%
web. Refutation: any of these failing beyond seed noise. Partial outcomes are reported per claim.

## Timeline
Data + tokenizer 20 min · pilot + eval harness 40 min · ~60 pre-training runs on 3–4 GPUs ≈ 2 h ·
evaluation ≈ 40 min · analysis 40 min · report 30 min.

## Potential Challenges
- Zero-shot knowledge accuracy near chance at this scale → composite restricted to tasks above chance;
  if none, fall back to fine-tuned closed-book QA and say so.
- Fine-tuning probes at floor/ceiling → calibrate budget on the uniform pilot + random-init control only.
- Shared GPUs / nearly full data disk → large artefacts on a scratch disk; scheduler skips busy GPUs.
- Simplex confound (raising one share lowers the others) → report regression coefficients, not only
  marginal correlations.

## Success Criteria
Each of H1–H4 receives a clear supported / not supported / inconclusive verdict backed by a test and a
noise estimate, with all runs, raw outputs and code saved.

## Deviations from the plan (logged during execution)

1. **No pre-gathered resources.** The resource-finder stage was skipped, so the literature review,
   corpora and evaluation data were assembled in-session (`literature_review.md`, `resources.md`).
2. **Smaller model and budget.** The 4 GPUs turned out to be shared with other sessions' concurrent jobs (all at 100% utilisation, 40–47 GB used). The planned 34M-parameter / 100M-token / context-512
   configuration was replaced by **17M parameters (6 layers × 384), 60M tokens, context 256** before any
   run that is analysed was started. The single pilot run in the old configuration was discarded.
3. **Evaluation probes fixed on the pilot.** Using only the uniform-mixture model and a random-init
   control: (a) zero-shot knowledge composite = SciQ + ARC-Easy (the only tasks above chance);
   (b) the planned fallback, **fine-tuned closed-book SciQ**, was added because zero-shot accuracy is only
   2–3 points above chance; (c) reasoning composite = ListOps + ProofWriter; (d) **arithmetic is reported
   but excluded from the composite** because the pilot showed negative transfer (random init 76%,
   pre-trained 8%) — it later proved bimodal across seeds.
4. **Design order and size.** Runs were queued in priority tiers in case compute ran out. It did not:
   all 63 planned runs finished, and 18 **extra seed replicates** (every p=0 and p=1 model, code sweep at
   0.2 and 0.5) were added once the machine freed up. The choice of which points to replicate was made
   before the full results were analysed, but after 9 runs had been inspected while debugging the
   analysis script.
5. **Law fitting excludes zero-share mixtures.** After seeing that p=0 is a cliff that no smooth law
   fits, multi-domain laws were fitted on interior mixtures only. An "effective-share" law tried in
   between fitted poorly and was dropped; this is reported in REPORT.md.
6. **Curve-shape models.** With 9 non-zero points per sweep the 3-parameter models are only loosely
   identified; leave-one-out error is the primary criterion.
