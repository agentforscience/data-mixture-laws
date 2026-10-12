"""Analysis of the mixture experiments: tables, statistical tests and figures.

Reads results/design.json, results/pretrain/*.json, results/evals/*.json and writes
results/runs.csv, results/analysis.json, results/analysis.md and figures/*.png.
Works on whatever subset of the design has finished.
"""
import glob
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

from common import DOMAINS, RESULTS_DIR, ROOT

FIG = os.path.join(ROOT, "figures")
COL = dict(web="#2a78d6", wiki="#eb6834", code="#1baf7a", arxiv="#eda100")  # fixed categorical order
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
                     "lines.markersize": 6, "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight"})

# Composite definitions were fixed after the uniform-mixture pilot + random-init control only.
K_ZS = ["know/sciq", "know/arc_easy"]            # zero-shot tasks on which the pilot beat chance
R_TASKS = ["reason/listops", "reason/proofwriter"]  # fine-tuned tasks with positive transfer in the pilot
DOWN = {"K_zs": "Knowledge, zero-shot (SciQ + ARC-Easy)", "K_ft": "Knowledge, fine-tuned (SciQ)",
        "R": "Reasoning (ListOps + ProofWriter)", "reason/arithmetic": "Arithmetic (fine-tuned)"}


# ----------------------------------------------------------------------------- data
def load():
    design = {r["name"]: r for r in json.load(open(os.path.join(RESULTS_DIR, "design.json")))}
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "pretrain", "*.json"))):
        p = json.load(open(f))
        e = os.path.join(RESULTS_DIR, "evals", p["name"] + ".json")
        if p["name"] not in design or not os.path.exists(e):
            continue
        e = json.load(open(e))
        if "reason/arithmetic" not in e:
            continue
        row = dict(name=p["name"], group=design[p["name"]]["group"], seed=p["seed"])
        row.update({d: p["mix"][d] for d in DOMAINS})
        row.update({f"loss/{d}": p["val_loss"][d] for d in DOMAINS})
        row.update({k: v for k, v in e.items() if k.startswith(("know/", "know_ft/", "reason/", "reason_tok/"))})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["K_zs"] = df[K_ZS].mean(1)
    df["K_ft"] = df["know_ft/sciq"]
    df["R"] = df[R_TASKS].mean(1)
    df["mixkey"] = df[DOMAINS].round(4).astype(str).agg(",".join, axis=1)
    return df


def seed_sd(df, metric):
    """Pooled within-mixture SD across seeds (noise floor) and its degrees of freedom."""
    ss, dof = 0.0, 0
    for _, g in df.groupby("mixkey"):
        if g.seed.nunique() > 1:
            ss += ((g[metric] - g[metric].mean()) ** 2).sum()
            dof += len(g) - 1
    return (np.sqrt(ss / dof) if dof else np.nan), dof


def sweep(df, dom, metric):
    """Mean of `metric` at each share p of `dom` along its sweep line (uniform = p 0.25)."""
    g = df[(df.group == f"sweep_{dom}") | (df.group == "uniform")]
    s = g.groupby(g[dom].round(4))[metric].agg(["mean", "count"])
    return s.index.values.astype(float), s["mean"].values, s["count"].values


# ----------------------------------------------------------------------------- curve shapes (H1/H2)
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, X @ beta


def fit_predict(kind, p, y, pnew):
    """Fit a one-dimensional law y(p) on (p, y) with p > 0 and predict at pnew."""
    one = np.ones_like(p)
    if kind == "linear":
        b, _ = _ols(np.c_[one, p], y)
        return b[0] + b[1] * pnew, 2
    if kind == "log":
        b, _ = _ols(np.c_[one, np.log(p)], y)
        return b[0] + b[1] * np.log(pnew), 2
    if kind == "exp":      # Ye et al. data-mixing-law form: c + k * exp(t * p)
        best = None
        for t in -np.logspace(-1, 2, 60):
            b, yh = _ols(np.c_[one, np.exp(t * p)], y)
            r = ((y - yh) ** 2).sum()
            if best is None or r < best[0]:
                best = (r, t, b)
        _, t, b = best
        return b[0] + b[1] * np.exp(t * pnew), 3
    if kind == "loghinge":  # log-linear up to a saturation share p*, flat afterwards
        best = None
        for ps in np.logspace(np.log10(0.06), 0, 40):
            b, yh = _ols(np.c_[one, np.log(np.minimum(p, ps))], y)
            r = ((y - yh) ** 2).sum()
            if best is None or r < best[0] - 1e-12:
                best = (r, ps, b)
        _, ps, b = best
        return b[0] + b[1] * np.log(np.minimum(pnew, ps)), 3, ps
    raise ValueError(kind)


def shape_table(df):
    """LOO-CV RMSE and AICc of candidate laws for every (domain sweep, metric)."""
    out = []
    metrics = ["in-domain loss", "K_zs", "K_ft", "R"]
    for dom in DOMAINS:
        for m in metrics:
            col = f"loss/{dom}" if m == "in-domain loss" else m
            p, y, _ = sweep(df, dom, col)
            keep = p > 0
            p, y = p[keep], y[keep]
            n = len(p)
            if n < 5:
                continue
            rec = dict(domain=dom, metric=m, n=n, noise_sd=seed_sd(df[df[dom] > 0], col)[0], range=float(y.max() - y.min()))
            for kind in ["linear", "log", "exp", "loghinge"]:
                res = fit_predict(kind, p, y, p)
                k = res[1]
                rss = float(((y - res[0]) ** 2).sum())
                loo = []
                for i in range(n):
                    m_ = np.arange(n) != i
                    loo.append(float(fit_predict(kind, p[m_], y[m_], p[i:i + 1])[0][0] - y[i]))
                rec[f"loo_{kind}"] = float(np.sqrt(np.mean(np.square(loo))))
                rec[f"r2_{kind}"] = 1 - rss / float(((y - y.mean()) ** 2).sum() + 1e-30)
                rec[f"aicc_{kind}"] = (n * np.log(rss / n + 1e-30) + 2 * k + 2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else np.nan
                if kind == "loghinge":
                    rec["hinge_pstar"] = float(res[2])
            rec["best_loo"] = min(["linear", "log", "exp", "loghinge"], key=lambda k: rec[f"loo_{k}"])
            out.append(rec)
    return pd.DataFrame(out)


def saturation_table(df):
    """How much of the total effect of a domain (p: 0 -> best) is already realised at p = 0.2 / 0.25 / 0.5."""
    out = []
    for dom in DOMAINS:
        for m in ["in-domain loss", "K_zs", "K_ft", "R"]:
            col = f"loss/{dom}" if m == "in-domain loss" else m
            sign = -1 if m == "in-domain loss" else 1
            p, y, _ = sweep(df, dom, col)
            y = sign * y
            if 0.0 not in p or 0.2 not in p:
                continue
            y0, ybest, pbest = y[p == 0][0], y.max(), p[y.argmax()]
            sd = seed_sd(df[df[dom] > 0], col)[0]  # noise among runs that contain the domain
            rec = dict(domain=dom, metric=m, y_p0=sign * y0, y_best=sign * ybest, p_best=pbest,
                       total_gain=ybest - y0, noise_sd=sd, gain_over_noise=(ybest - y0) / sd if sd else np.nan)
            for q in [0.05, 0.2, 0.5]:
                if q in p:
                    rec[f"frac_at_{q}"] = (y[p == q][0] - y0) / (ybest - y0) if ybest > y0 else np.nan
            for q in [0.5, 1.0]:
                if q in p:
                    rec[f"delta_0.2_to_{q}"] = y[p == q][0] - y[p == 0.2][0]
            out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- correlations (H3)
def boot_ci(x, y, fn, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    vals = np.array([fn(x[i], y[i]) for i in idx if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0])
    return np.percentile(vals, [2.5, 97.5])


def corr_table(df, subset, label):
    out = []
    g = df if subset is None else df[df.group == subset]
    g = g.groupby("mixkey").mean(numeric_only=True)  # average seed replicates
    if len(g) < 6:
        return pd.DataFrame()
    for m in ["R", "reason/listops", "reason/proofwriter", "reason/arithmetic", "K_zs", "K_ft"]:
        for dom in DOMAINS:
            x, y = g[dom].values, g[m].values
            r, pr = stats.pearsonr(x, y)
            rho, ps = stats.spearmanr(x, y)
            lo, hi = boot_ci(x, y, lambda a, b: stats.pearsonr(a, b)[0])
            out.append(dict(runs=label, n=len(g), metric=m, domain=dom, pearson=r, p=pr, ci_lo=lo, ci_hi=hi, spearman=rho, p_s=ps))
    out = pd.DataFrame(out)
    out["p_holm"] = np.nan
    for _, idx in out.groupby("metric").groups.items():  # Holm across the four domains
        ps_ = out.loc[idx, "p"].sort_values()
        adj = np.maximum.accumulate([min(1, (len(ps_) - i) * v) for i, v in enumerate(ps_)])
        out.loc[ps_.index, "p_holm"] = adj
    return out


def scheffe(df, metrics):
    """Linear mixture (Scheffe) model y = sum_d beta_d * p_d: beta_d is the predicted score of a
    pure-d corpus, so coefficients are comparable across domains despite the simplex constraint."""
    g = df.groupby("mixkey").mean(numeric_only=True)
    X = g[DOMAINS].values
    out = []
    for m in metrics:
        y = g[m].values
        beta, yh = _ols(X, y)
        rng = np.random.default_rng(0)
        bs = np.array([_ols(X[i], y[i])[0] for i in rng.integers(0, len(y), (2000, len(y)))])
        rec = dict(metric=m, n=len(y), r2=1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        for j, d in enumerate(DOMAINS):
            rec[f"beta_{d}"] = beta[j]
            rec[f"se_{d}"] = bs[:, j].std()
        out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- web > 50% (H4)
def web_table(df):
    g = df[(df.group == "sweep_web") | (df.group == "uniform")]
    rows = []
    for m in ["K_zs", "K_ft", "know/sciq", "know/arc_easy", "R", "loss/wiki", "loss/arxiv"]:
        by = {round(p, 2): v[m].values for p, v in g.groupby(g.web.round(4))}
        for p, v in sorted(by.items()):
            rows.append(dict(metric=m, web_share=p, n_seeds=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan))
    tests = []
    for m in ["K_zs", "K_ft"]:
        by = {round(p, 2): v[m].values for p, v in g.groupby(g.web.round(4))}
        for a, b in [(0.5, 0.75), (0.5, 1.0), (0.25, 0.5), (0.25, 0.75), (0.25, 1.0)]:
            if a in by and b in by and len(by[a]) > 1 and len(by[b]) > 1:
                t, pv = stats.ttest_ind(by[b], by[a], equal_var=False)
                tests.append(dict(metric=m, contrast=f"web {b} vs {a}", diff_pts=100 * (by[b].mean() - by[a].mean()),
                                  t=t, p=pv, n=f"{len(by[b])} vs {len(by[a])}"))
        hi = g[g.web.round(2) >= 0.5]
        if hi.web.nunique() >= 3:
            sl = stats.linregress(hi.web, hi[m])
            tests.append(dict(metric=m, contrast="slope for web>=0.5 (pts per +10% web)", diff_pts=10 * sl.slope, t=sl.slope / sl.stderr, p=sl.pvalue, n=str(len(hi))))
    return pd.DataFrame(rows), pd.DataFrame(tests)


# ----------------------------------------------------------------------------- mixture-law prediction
def _nls(resid, inits, bounds):
    best = None
    for x0 in inits:
        try:
            r = least_squares(resid, x0, bounds=bounds)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def loss_laws(dom):
    """Candidate laws for the held-out loss of domain `dom` as a function of the full mixture P (n, 4),
    all shares > 0.  name -> (n_params, fit(P, y) -> predict(P))."""
    j = DOMAINS.index(dom)

    def scheffe_(P, y):
        beta, _ = _ols(P, y)
        return lambda Q: Q @ beta

    def own_log(P, y):     # a + b log(p_own): ignores how the rest of the mixture is composed
        beta, _ = _ols(np.c_[np.ones(len(P)), np.log(P[:, j])], y)
        return lambda Q: beta[0] + beta[1] * np.log(Q[:, j])

    def add_log(P, y):     # a + sum_j b_j log(p_j): log-linear in every domain's share
        beta, _ = _ols(np.c_[np.ones(len(P)), np.log(P)], y)
        pred = lambda Q: np.c_[np.ones(len(Q)), np.log(Q)] @ beta
        pred.params = dict(a=beta[0], **{f"b_{d}": beta[1 + k] for k, d in enumerate(DOMAINS)})
        return pred

    def ye_exp(P, y):      # Ye et al. (2024) data mixing law: c + k exp(sum_j t_j p_j)
        f = lambda x, Q: x[0] + x[1] * np.exp(Q @ x[2:])
        rng = np.random.default_rng(0)
        inits = [np.r_[y.min() - rng.uniform(0, 1), 1.0, rng.normal(0, 3, 4)] for _ in range(40)]
        x = _nls(lambda x: f(x, P) - y, inits, ([-50, 0] + [-80] * 4, [50, 100] + [80] * 4))
        return lambda Q: f(x, Q)

    return {"linear (Scheffe)": (4, scheffe_), "log of own share": (2, own_log),
            "additive log-share": (5, add_log), "Ye et al. exp": (6, ye_exp)}


def law_prediction(df):
    """Fit laws on the structured interior runs (sweeps + uniform with every share > 0) and predict the
    held-out random mixtures.  Mixtures with a zero share are excluded from law fitting: p = 0 is a
    qualitatively different regime (see the saturation table)."""
    g = df.groupby("mixkey").mean(numeric_only=True).join(df.groupby("mixkey").group.first())
    g = g[(g[DOMAINS] > 0).all(axis=1)]
    tr, te = g[g.group != "random"], g[g.group == "random"]
    if len(te) < 4:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    Ptr, Pte = tr[DOMAINS].values, te[DOMAINS].values
    out, preds, transfer = [], {}, []
    for d in DOMAINS:
        m = f"loss/{d}"
        ytr, yte = tr[m].values, te[m].values
        for name, (k, fit) in loss_laws(d).items():
            pred = fit(Ptr, ytr)
            ph = pred(Pte)
            out.append(dict(metric=m, law=name, n_params=k, n_train=len(tr), n_test=len(te),
                            rmse_train=float(np.sqrt(((pred(Ptr) - ytr) ** 2).mean())),
                            rmse_test=float(np.sqrt(((ph - yte) ** 2).mean())),
                            max_abs_err_test=float(np.abs(ph - yte).max()),
                            r_test=float(stats.pearsonr(ph, yte)[0]),
                            rmse_mean_baseline=float(np.sqrt(((yte - ytr.mean()) ** 2).mean()))))
            preds[(m, name)] = (yte, ph)
        # transfer coefficients from the additive log-share law, fitted on all interior mixtures
        P, y = g[DOMAINS].values, g[m].values
        par = loss_laws(d)["additive log-share"][1](P, y).params
        rng = np.random.default_rng(0)
        bs = pd.DataFrame([loss_laws(d)["additive log-share"][1](P[i], y[i]).params for i in rng.integers(0, len(y), (1000, len(y)))])
        transfer.append(dict(target_loss=d, n=len(y), **{k: v for k, v in par.items()}, **{k.replace("b_", "se_"): bs[k].std() for k in par if k != "a"}))
    feats = {"linear (Scheffe)": lambda d: d[DOMAINS].values,
             "additive log-share": lambda d: np.c_[np.ones(len(d)), np.log(d[DOMAINS].values)]}
    down = []
    for m in ["K_zs", "K_ft", "R", "reason/listops", "reason/proofwriter"]:
        rec = dict(metric=m, n_train=len(tr), n_test=len(te), noise_sd=seed_sd(df, m)[0],
                   rmse_mean_baseline=float(np.sqrt(((te[m] - tr[m].mean()) ** 2).mean())))
        for name, f in feats.items():
            beta, _ = _ols(f(tr), tr[m].values)
            ph = f(te) @ beta
            rec[f"rmse_{name}"] = float(np.sqrt(((te[m].values - ph) ** 2).mean()))
            rec[f"r_{name}"] = float(stats.pearsonr(ph, te[m].values)[0])
        down.append(rec)
    return pd.DataFrame(out), pd.DataFrame(down), pd.DataFrame(transfer), preds


# ----------------------------------------------------------------------------- ablation contrasts
def ablation_table(df):
    """Seed-replicated contrasts against the uniform mixture: domain removed (p=0), pure domain (p=1),
    and any other replicated sweep point. Welch t-tests; differences in accuracy points."""
    uni = df[df.group == "uniform"]
    rows = []
    for dom in DOMAINS:
        g = df[df.group == f"sweep_{dom}"]
        for p, v in g.groupby(g[dom].round(2)):
            if len(v) < 2:
                continue
            for m in ["K_zs", "K_ft", "reason/listops", "reason/proofwriter", "R", "reason/arithmetic"]:
                t, pv = stats.ttest_ind(v[m], uni[m], equal_var=False)
                rows.append(dict(domain=dom, share=p, metric=m, n=len(v), mean=100 * v[m].mean(), sd=100 * v[m].std(ddof=1),
                                 uniform_mean=100 * uni[m].mean(), diff_pts=100 * (v[m].mean() - uni[m].mean()), t=t, p=pv))
    return pd.DataFrame(rows)


def loss_downstream_corr(df):
    g = df.groupby("mixkey").mean(numeric_only=True)
    g["loss/natural-language mean"] = g[["loss/web", "loss/wiki", "loss/arxiv"]].mean(axis=1)
    rows = []
    for m in ["K_zs", "K_ft", "reason/listops", "reason/proofwriter", "R"]:
        rec = dict(metric=m, n=len(g))
        for l in [f"loss/{d}" for d in DOMAINS] + ["loss/natural-language mean"]:
            rec["spearman vs " + l] = stats.spearmanr(g[l], g[m])[0]
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig_indomain(df, shapes):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for dom in DOMAINS:
        p, y, _ = sweep(df, dom, f"loss/{dom}")
        ax[0].plot(p, y, "-o", color=COL[dom], label=dom, markeredgecolor="white", markeredgewidth=1.5)
        k = p > 0
        ax[1].plot(p[k], y[k], "o", color=COL[dom], label=dom, markeredgecolor="white", markeredgewidth=1.5)
        if k.sum() >= 3:
            xs = np.logspace(np.log10(p[k].min()), 0, 50)
            ax[1].plot(xs, fit_predict("log", p[k], y[k], xs)[0], "-", color=COL[dom], lw=1.2, alpha=0.8)
    ax[0].set(xlabel="share of the domain in the pre-training mixture", ylabel="held-out loss on that domain (nats/token)",
              title="In-domain loss vs. domain share")
    ax[1].set(xscale="log", xlabel="share of the domain (log scale)", title="Same data, log axis; lines = log-linear fits")
    ax[0].legend(title="domain")
    fig.savefig(os.path.join(FIG, "fig1_indomain_loss.png"))
    plt.close(fig)


def fig_downstream(df, scratch):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    for ax, (m, title) in zip(axes.ravel(), DOWN.items()):
        sd = seed_sd(df, m)[0]
        u = df[df.group == "uniform"][m].mean()
        if np.isfinite(sd):
            ax.axhspan(100 * max(u - 2 * sd, 0), 100 * min(u + 2 * sd, 1), color="#efeee8", zorder=0, label="uniform mix ± 2 seed SD")
        for dom in DOMAINS:
            p, y, _ = sweep(df, dom, m)
            ax.plot(p, 100 * y, "-o", color=COL[dom], label=dom, markeredgecolor="white", markeredgewidth=1.5)
        sk = {"K_zs": np.mean([scratch[k] for k in K_ZS]), "K_ft": scratch["know_ft/sciq"],
              "R": np.mean([scratch[k] for k in R_TASKS])}.get(m, scratch.get(m))
        if sk is not None and m != "reason/arithmetic":
            ax.axhline(100 * sk, color=MUTED, ls=":", lw=1.5, label="no pre-training")
        ax.set(title=title, ylabel="accuracy (%)")
    for ax in axes[1]:
        ax.set_xlabel("share of the swept domain (others split the rest equally)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(os.path.join(FIG, "fig2_downstream_sweeps.png"))
    plt.close(fig)


def fig_web(df, scratch):
    g = df[(df.group == "sweep_web") | (df.group == "uniform")]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for a, m in zip(ax, ["K_zs", "K_ft"]):
        a.plot(g.web, 100 * g[m], "o", color=COL["web"], alpha=0.45, markersize=5, label="individual seeds")
        mu = g.groupby(g.web.round(4))[m].mean()
        a.plot(mu.index, 100 * mu.values, "-", color=COL["web"], label="mean")
        a.axvline(0.5, color=MUTED, ls="--", lw=1)
        a.text(0.51, a.get_ylim()[0], " hypothesised\n threshold", color=MUTED, fontsize=8, va="bottom")
        a.set(title=DOWN[m], xlabel="web share of the pre-training mixture", ylabel="accuracy (%)")
    ax[0].legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "fig3_web_knowledge.png"))
    plt.close(fig)


def fig_corr(ct):
    ms = ["R", "reason/listops", "K_zs", "K_ft"]
    labels = [l for l in ["random mixtures", "all mixtures"] if (ct.runs == l).any()]
    fig, axes = plt.subplots(1, len(labels), figsize=(6.6 * len(labels), 4), sharey=True, squeeze=False)
    w = 0.19
    for ax, label in zip(axes[0], labels):
        sub = ct[ct.runs == label]
        for j, dom in enumerate(DOMAINS):
            s_ = sub[sub.domain == dom].set_index("metric").loc[ms]
            x = np.arange(len(ms)) + (j - 1.5) * (w + 0.01)
            ax.bar(x, s_.pearson, w, color=COL[dom], label=dom)
            ax.errorbar(x, s_.pearson, yerr=[s_.pearson - s_.ci_lo, s_.ci_hi - s_.pearson], fmt="none", ecolor=INK, lw=1)
        ax.axhline(0.7, color=MUTED, ls="--", lw=1)
        ax.text(-0.45, 0.73, "hypothesised r = 0.7 (code, reasoning)", color=MUTED, fontsize=8)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(np.arange(len(ms)), ["Reasoning\ncomposite", "ListOps", "Knowledge\nzero-shot", "Knowledge\nfine-tuned"])
        ax.set(ylim=(-1, 1), title=f"{label} (n = {int(sub.n.iloc[0])})")
    axes[0, 0].set_ylabel("Pearson r between domain share and score\n(bars: bootstrap 95% CI)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, title="domain whose share is correlated", ncol=4, loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.14))
    fig.savefig(os.path.join(FIG, "fig4_correlations.png"))
    plt.close(fig)


def fig_law(preds):
    laws = [("linear (Scheffe)", "o", "#2a78d6"), ("log of own share", "s", "#eb6834"),
            ("additive log-share", "D", "#1baf7a"), ("Ye et al. exp", "^", "#eda100")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, d in zip(axes, DOMAINS):
        for name, mk, c in laws:
            y, ph = preds[(f"loss/{d}", name)]
            ax.plot(y, ph, mk, color=c, label=name, markeredgecolor="white", markeredgewidth=1, markersize=6.5)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=0)
        ax.set(title=f"held-out {d} loss", xlabel="measured (nats/token)", ylabel="predicted", ylim=(lo - 0.3, hi + 0.3))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.2), title="law fitted on sweep runs (all shares > 0), tested on 16 random mixtures")
    fig.savefig(os.path.join(FIG, "fig5_law_prediction.png"))
    plt.close(fig)


def md(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt) if len(df) else "_not enough runs_"


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load()
    df.drop(columns="mixkey").to_csv(os.path.join(RESULTS_DIR, "runs.csv"), index=False)
    scratch = json.load(open(os.path.join(RESULTS_DIR, "evals", "scratch_s0.json")))
    S = ["# Analysis output (auto-generated by src/analyze.py)\n", f"Runs analysed: {len(df)} "
         f"({df.groupby('group').size().to_dict()}); distinct mixtures: {df.mixkey.nunique()}\n"]

    metrics = [f"loss/{d}" for d in DOMAINS] + ["know/sciq", "know/arc_easy", "know/arc_challenge", "know/openbookqa",
               "know/commonsenseqa", "K_zs", "K_ft", "reason/listops", "reason/proofwriter", "R", "reason/arithmetic"]
    desc = pd.DataFrame([dict(metric=m, mean=df[m].mean(), sd_across_mixtures=df.groupby("mixkey")[m].mean().std(),
                              min=df[m].min(), max=df[m].max(), seed_sd=seed_sd(df, m)[0], seed_dof=seed_sd(df, m)[1],
                              scratch=scratch.get(m, np.nan)) for m in metrics])
    desc["signal_to_noise"] = desc.sd_across_mixtures / desc.seed_sd
    S += ["## Descriptive statistics and noise floor\n", md(desc, ".4f"), ""]

    rep = df[df.duplicated("mixkey", keep=False)].sort_values(["mixkey", "seed"])
    S += ["## Seed replicates\n", md(rep[["name"] + [f"loss/{d}" for d in DOMAINS] + ["K_zs", "K_ft", "R", "reason/arithmetic"]], ".4f"), ""]

    key = ["name"] + DOMAINS + [f"loss/{d}" for d in DOMAINS] + ["K_zs", "K_ft", "reason/listops", "reason/proofwriter", "R", "reason/arithmetic"]
    S += ["## All runs\n", md(df.sort_values(["group", "name"])[key], ".3f"), ""]

    shapes = shape_table(df)
    S += ["## H1 - curve shape along single-domain sweeps (p > 0)\n",
          md(shapes[["domain", "metric", "n", "range", "noise_sd", "loo_linear", "loo_log", "loo_exp", "loo_loghinge", "r2_linear", "r2_log", "hinge_pstar", "best_loo"]], ".4f") if len(shapes) else "_n/a_", ""]
    sat = saturation_table(df)
    S += ["## H2 - saturation\n", md(sat, ".3f"), ""]

    ct = pd.concat([corr_table(df, None, "all mixtures"), corr_table(df, "random", "random mixtures")])
    S += ["## H3 - correlations of domain share with downstream scores\n", md(ct, ".3f"), ""]
    sch = scheffe(df, ["R", "reason/listops", "reason/proofwriter", "reason/arithmetic", "K_zs", "K_ft"])
    S += ["### Scheffe mixture regression (beta_d = predicted score of a pure-d corpus)\n", md(sch, ".3f"), ""]

    wt, tests = web_table(df)
    S += ["## H4 - web sweep\n", md(wt, ".4f"), "", md(tests, ".3f"), ""]

    abl = ablation_table(df)
    S += ["## Seed-replicated contrasts against the uniform mixture (accuracy points)\n", md(abl, ".2f"), ""]
    ldc = loss_downstream_corr(df)
    S += ["## Rank correlation of downstream scores with held-out losses across mixtures\n", md(ldc, ".2f"), ""]
    law, law_down, transfer, preds = law_prediction(df)
    S += ["## Mixture-law prediction of held-out random mixtures: per-domain loss\n", md(law, ".4f"), "",
          "### Additive log-share law L_d = a + sum_j b_j log(p_j), fitted on all interior mixtures (bootstrap SE)\n", md(transfer, ".3f"), "",
          "### Downstream scores\n", md(law_down, ".4f"), ""]

    fig_indomain(df, shapes)
    fig_downstream(df, scratch)
    fig_web(df, scratch)
    if len(ct):
        fig_corr(ct)
    if preds:
        fig_law(preds)
    open(os.path.join(RESULTS_DIR, "analysis.md"), "w").write("\n".join(S))
    json.dump(dict(desc=desc.to_dict("records"), shapes=shapes.to_dict("records"), saturation=sat.to_dict("records"),
                   correlations=ct.to_dict("records"), scheffe=sch.to_dict("records"), web=wt.to_dict("records"),
                   web_tests=tests.to_dict("records"), law=law.to_dict("records"), law_downstream=law_down.to_dict("records"),
                   transfer=transfer.to_dict("records"), ablation=abl.to_dict("records"), loss_downstream=ldc.to_dict("records")),
              open(os.path.join(RESULTS_DIR, "analysis.json"), "w"), indent=1, default=float)
    print("\n".join(S))


if __name__ == "__main__":
    main()
