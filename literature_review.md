# Literature Review (compiled by the experiment runner)

The automated resource-finding stage was **skipped** for this workspace (`.neurico/pipeline_state.json`:
`"skipped": true`), so no papers, datasets or code were pre-gathered. This short review was assembled at
the start of the session. The first four entries were checked by web search (abstract-level only — the
PDFs were not downloaded or read in full); the last group is from the author's background knowledge and
was not re-verified.

## Verified (abstract level)

**Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance** — Ye, Liu, Sun,
Zhan, Zhou, Qiu. ICLR 2025, arXiv:2403.16952. Claims that model performance (validation loss) is a
predictable function of mixture proportions; fits functional forms ("data mixing laws", exponential in
the proportions) on a few sampled mixtures and nests them with step/size scaling laws to choose a
mixture for a 1B model on RedPajama. *Relevance:* closest prior work; its target is validation loss, not
downstream accuracy, and its functional form is exponential-in-proportion rather than log-linear.

**RegMix: Data Mixture as Regression for Language Model Pre-training** — Liu et al. ICLR 2025,
arXiv:2407.01492. Trains many tiny models (512 × 1M parameters, 1B tokens) on random mixtures, fits a
regression from mixture to a target loss and transfers the best mixture to larger models. Reports that
web corpora, rather than "high-quality" sources such as Wikipedia, correlate most positively with
downstream performance. *Relevance:* validates the small-proxy-model methodology used here, and its web
finding runs against our hypothesis' claim that heavy web text hurts.

**To Code, or Not To Code? Exploring Impact of Code in Pre-training** — Aryabumi et al. (Cohere).
ICLR 2025, arXiv:2408.10914. Ablations on 470M–2.8B models; adding code to pre-training gives relative
gains of up to 8.2% on natural-language reasoning and 4.2% on world knowledge versus text-only
pre-training. *Relevance:* the direct motivation for the "code → reasoning" claim; it uses a few
mixtures rather than a dose–response curve.

**Scaling Laws for Optimal Data Mixtures** — Shukor et al. (Sorbonne/Apple). NeurIPS 2025,
arXiv:2507.09404. Extends scaling laws to predict loss as a function of model size, token count and
domain weights, fitted on small runs. *Relevance:* again a loss-level law.

## Background (not re-verified in this session)
- DoReMi (Xie et al., 2023) and DoGE (Fan et al., 2023): learn domain weights with a proxy model.
- Chinchilla / Kaplan-style scaling laws: loss is a power law in data and parameters, which is why a
  logarithmic dependence on the amount of in-domain data is the natural prior for H1.
- BabyLM-style work: models trained on ≤100M words are evaluated mostly by likelihood-based and
  fine-tuning protocols because zero-shot benchmark accuracy is near chance at that scale. This is why
  our reasoning tasks are measured by fine-tuning transfer.

## Implications for the design
1. Small proxy models on many mixtures are the accepted methodology (RegMix, Data Mixing Laws).
2. Prior laws target loss; the open question is whether *downstream* metrics follow similar shapes.
3. Proportions live on a simplex: raising one share lowers the others, so marginal correlations must be
   complemented by mixture-regression coefficients.
4. Published evidence disagrees on web text (RegMix: most useful) vs. the hypothesis (harmful >50%).
