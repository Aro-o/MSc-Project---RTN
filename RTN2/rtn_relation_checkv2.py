"""Relation-source check: which source names the classes the model really confuses?

RTN pushes harder on negatives that hold a related class, so a class's related
classes should be the ones the model actually confuses it with. This check
scores candidate relation sources against the confusions of the trained
MulSupCon model on the dev set, before any RTN training. It reads the saved
MulSupCon runs and the encoders' training features, trains nothing, never uses
test labels, and uses rtn_skeleton.py read-only.

Relation sources (each gives a similarity between every two classes; each
class's top 5% most similar classes count as related, as in the skeleton):

    GloVe + WordNet              the skeleton's current source (words)
    image: own appearance        each class's own share of the training
                                 features, separated from the classes it
                                 appears with (ridge regression on the
                                 labels): the candidate for version 3
    image: class averages        the plain average training feature of each
                                 class (reference: it mixes in the classes a
                                 class appears with)
    CLIP text (reference only)   optional, off by default; never used in RTN

How a class's confusions are measured. For class c, every dev image without c
gets the percentile of its c-score among all dev images without c (averaged
over the 3 MulSupCon seeds), minus the image's own level (its average
percentile over all the classes it is a negative of), so that busy images
that score high for every class do not count as confusing them all. The
confusion of another class k is how much higher the dev images holding k sit
than an average negative (percentile points), shrunk towards zero when only
a few dev images hold k. It needs only negatives, so it works for tail
classes with few dev positives; a class k needs at least MIN_IMAGES such dev
images to be measured.

Check version 1 (18 Nov) had neither correction: the same few classes (busy
images) topped almost every class's list of confusers, which is how the flaw
showed up.
"""
import contextlib
import os
import warnings

import numpy as np
import pandas as pd

import rtn_skeleton as sk

CHECK_VERSION = 2         # 2: each image's own level removed, small counts shrunk
COHORTS = ("tail", "medium", "head")
MIN_IMAGES = 3            # dev images holding k (and not c) needed to measure k's confusion
SHRINK = 3                # pseudo-images at zero confusion added to every estimate
RIDGE = 1.0               # ridge penalty of the "own appearance" profiles (fixed, not tuned)
TOP = 4                   # the 4 most confused classes (the skeleton relates 4 of 79)
GLOVE = "GloVe + WordNet (now)"
AVERAGES = "image: class averages (reference)"
OWN = "image: own appearance"
CLIP = "CLIP text (reference only)"
RANDOM = "random classes (chance)"


@contextlib.contextmanager
def quiet():
    """Hide the skeleton's own tables while loading."""
    show = sk._show
    sk._show = lambda obj: None
    try:
        yield
    finally:
        sk._show = show


# ================================================================= components
def class_averages(X, Y):
    """C x d: the average (standardised) training feature of each class."""
    Yb = np.asarray(Y) > 0.5
    return np.stack([X[Yb[:, c]].mean(0) for c in range(Yb.shape[1])])


def own_appearance(X, Y, lam=RIDGE):
    """C x d: each class's own contribution to the training features, separated
    from the classes it appears with: the ridge solution of X ~ Y W."""
    Yf = (np.asarray(Y) > 0.5).astype(np.float64)
    A = Yf.T @ Yf + lam * np.eye(Yf.shape[1])
    return np.linalg.solve(A, Yf.T @ (X - X.mean(0)))


def text_similarity(path, L):
    """Cosine similarity of saved class-name embeddings (for the reference)."""
    E = np.load(path)
    if E.ndim != 2 or E.shape[0] != L:
        raise SystemExit(f"{path} has shape {E.shape}; expected {L} rows, one per class.")
    return sk.cosine_similarity(E)


def dev_percentiles(scores, Y_dev, centre=True):
    """N x C: for every class and dev image without it, the image's score
    percentile among all dev images without the class, averaged over seeds
    (NaN for the class's positives). centre=True removes each image's own
    level, its average percentile over the classes it is a negative of, so an
    image that scores high for every class (e.g. a busy scene) does not count
    as confusing them all; the same idea as the skeleton's Control 2."""
    Yd = np.asarray(Y_dev) > 0.5
    P = np.zeros(Yd.shape)
    for S in scores:
        for c in range(Yd.shape[1]):
            neg = ~Yd[:, c]
            P[neg, c] += 100 * pd.Series(S[neg, c]).rank(pct=True).to_numpy()
    P /= len(scores)
    P[Yd] = np.nan
    if centre:
        P = P - np.nanmean(P, axis=1, keepdims=True)
    return P


def confusion_matrix(P, Y_dev, min_images=MIN_IMAGES, shrink=SHRINK):
    """C x C: how much higher the dev images holding k (without c) sit in c's
    ranking than an average negative of c, beyond their own level (percentile
    points; NaN when fewer than min_images such images), shrunk towards 0 by
    n / (n + shrink) so that classes with few dev images do not top the lists
    by luck; and the number n of those images."""
    Yd = np.asarray(Y_dev) > 0.5
    L = Yd.shape[1]
    conf, n = np.full((L, L), np.nan), np.zeros((L, L), dtype=int)
    for c in range(L):
        neg = ~Yd[:, c]
        base = np.nanmean(P[neg, c])
        for k in range(L):
            if k == c:
                continue
            m = neg & Yd[:, k]
            n[c, k] = int(m.sum())
            if n[c, k] >= min_images:
                conf[c, k] = (P[m, c].mean() - base) * n[c, k] / (n[c, k] + shrink)
    return conf, n


# ===================================================================== check
class Check:
    """One encoder: its MulSupCon dev confusions and the candidate sources."""

    def __init__(self, dataset, encoder, quick_test=False, colour=None):
        with quiet():
            exp = sk.Experiment(dataset, encoder, quick_test=quick_test, allow_cpu=True)
            exp.relations()
            exp.prepare()
        self.exp, self.label = exp, encoder["label"]
        self.colour = colour or sk.COLOURS[0]
        D = exp.D
        scores = []
        for seed in exp.seeds:
            p = exp._spec("mulsupcon", "uniform", {}, seed)["path"]
            if not os.path.exists(p):
                raise SystemExit(f"{self.label}: the MulSupCon run for seed {seed} is missing; "
                                 "run this encoder's notebook first.")
            r = exp.rcr.load_run(p)
            if not exp._matches(r):
                raise SystemExit(f"{p} does not match this split; rerun the encoder notebook.")
            scores.append(np.asarray(r["dev_logits"], dtype=np.float64))
        self.names, self.cohort, self.L = exp.names, np.asarray(exp.cohort), exp.L
        self.cls = {g: np.flatnonzero(self.cohort == g) for g in COHORTS}
        self.Y_fit = D["Y_fit"]
        X = D["Xf"].detach().cpu().numpy().astype(np.float64)
        self.sims = {GLOVE: exp.sim,
                     OWN: sk.cosine_similarity(own_appearance(X, self.Y_fit)),
                     AVERAGES: sk.cosine_similarity(class_averages(X, self.Y_fit))}
        self.related = {s: sk.related_mask(m) for s, m in self.sims.items()}
        self.conf, self.n = confusion_matrix(dev_percentiles(scores, D["Y_dev"]), D["Y_dev"])
        self._per = {}

    def add_source(self, name, sim):
        self.sims[name] = sim
        self.related[name] = sk.related_mask(sim)
        self._per.pop(name, None)

    def top(self, c):
        """The TOP most confused measurable classes of class c (bool mask)."""
        valid = np.isfinite(self.conf[c])
        out = np.zeros(self.L, dtype=bool)
        if valid.sum() >= TOP:
            out[np.argsort(-np.where(valid, self.conf[c], -np.inf), kind="stable")[:TOP]] = True
        return out

    def per_class(self, source):
        """Per class: mean confusion of its measurable related classes, the share
        of them among its 4 most confused classes, and the share measurable."""
        if source in self._per:
            return self._per[source]
        rows = []
        for c in range(self.L):
            valid = np.isfinite(self.conf[c])
            if source == RANDOM:          # expected values for 4 random classes
                rows.append(dict(confusion=np.nanmean(self.conf[c]) if valid.any() else np.nan,
                                 in_top=100 * TOP / valid.sum() if valid.sum() >= TOP else np.nan,
                                 measurable=np.nan))
                continue
            rel = self.related[source][c]
            r = rel & valid
            rows.append(dict(
                confusion=float(self.conf[c, r].mean()) if r.any() else np.nan,
                in_top=100 * float((self.top(c) & r).sum() / r.sum())
                if r.any() and valid.sum() >= TOP else np.nan,
                measurable=100 * float(r.sum() / rel.sum())))
        self._per[source] = pd.DataFrame(rows)
        return self._per[source]


# ===================================================================== tables
def load(dataset, encoders, quick_test=False, clip_text_file=None):
    """One Check per encoder. clip_text_file: saved CLIP class-name embeddings,
    added as a reference source only (None leaves it out)."""
    checks = []
    for i, enc in enumerate(encoders):
        ch = Check(dataset, enc, quick_test=quick_test, colour=sk.COLOURS[i % len(sk.COLOURS)])
        if clip_text_file:
            ch.add_source(CLIP, text_similarity(clip_text_file, ch.L))
        checks.append(ch)
    rows = [{"encoder": ch.label, "MulSupCon runs (dev scores)": len(ch.exp.seeds),
             "dev images": int(len(ch.exp.D["dev"])),
             "class pairs measurable (%)": round(100 * float(np.isfinite(ch.conf).sum())
                                                 / (ch.L * (ch.L - 1)), 1)} for ch in checks]
    sk._show(pd.DataFrame(rows))
    return checks


def scores_table(checks):
    """Per encoder, cohort and source: how confused the model is with each
    source's related classes, against random classes."""
    rows = []
    for ch in checks:
        for coh in COHORTS:
            idx = ch.cls[coh]
            for s in list(ch.sims) + [RANDOM]:
                sub = ch.per_class(s).iloc[idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rows.append({"encoder": ch.label, "cohort": coh, "relation source": s,
                                 "confusion of the related classes":
                                     round(float(np.nanmean(sub["confusion"])), 2),
                                 "related among the 4 most confused (%)":
                                     round(float(np.nanmean(sub["in_top"])), 1),
                                 "related classes measurable (%)":
                                     "-" if s == RANDOM else
                                     round(float(np.nanmean(sub["measurable"])), 1)})
    return pd.DataFrame(rows)


def against_glove(checks, seed=0):
    """Per cohort: each image source minus GloVe + WordNet in the confusion of
    the related classes, per class averaged over the encoders, with a 95%
    bootstrap interval over classes; and the decision for version 3."""
    rng = np.random.default_rng(seed)
    rows, tail = [], {}
    for coh in COHORTS:
        idx = checks[0].cls[coh]
        for s in [s for s in checks[0].sims if s != GLOVE]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d = np.nanmean([ch.per_class(s)["confusion"].to_numpy()[idx]
                                - ch.per_class(GLOVE)["confusion"].to_numpy()[idx]
                                for ch in checks], axis=0)
            r = sk._boot(rng, d, 10000)
            rows.append({"cohort": coh, "relation source": s, "classes": r[3],
                         "difference from GloVe + WordNet": sk._ci(r)})
            if coh == "tail":
                tail[s] = r
    table = pd.DataFrame(rows)
    own, avg = tail.get(OWN), tail.get(AVERAGES)
    # Only "own appearance" is a candidate: class averages mix in the classes a
    # class appears with, which the skeleton's relation types already handle.
    if own is not None and np.isfinite(own[1]) and own[1] > 0:
        decision = (f"Version 3 changes the relation source to '{OWN}': on the tail classes its "
                    f"related classes are more confused than GloVe + WordNet's by "
                    f"{sk._ci(own)} percentile points (interval above zero).")
    else:
        decision = (f"'{OWN}' does not beat GloVe + WordNet clearly on the tail classes, so the "
                    "relation source is not the bottleneck: version 3 keeps GloVe + WordNet and "
                    "moves RTN into fine-tuning instead.")
        if avg is not None and np.isfinite(avg[1]) and avg[1] > 0:
            decision += (f" (Class averages do name more confusers, {sk._ci(avg)}: mostly classes "
                         "that appear together, which points at the companions' relation value "
                         "rather than at the source.)")
    return table, decision


def examples(check, cohort="tail"):
    """Related classes of each class from every source, next to the classes the
    model confuses it with most on the dev set."""
    rows = []
    nm = check.names
    for c in check.cls[cohort]:
        row = {"class": nm[c]}
        for s in check.sims:
            row[s] = ", ".join(nm[k] for k in np.flatnonzero(check.related[s][c]))
        row["most confused on dev (MulSupCon)"] = ", ".join(
            nm[k] for k in np.argsort(-np.where(check.top(c), check.conf[c], -np.inf))[:TOP]
            if check.top(c)[k])
        rows.append(row)
    return pd.DataFrame(rows)


def types_table(checks):
    """How each source would change the skeleton's relation types (pairs in the
    training part), per cohort."""
    rows = []
    for ch in checks:
        for s, sim in ch.sims.items():
            if s == GLOVE and ch is not checks[0]:
                continue                       # the same for every encoder
            _, m = sk.relation_types(ch.Y_fit, sim)
            for coh in COHORTS:
                idx = ch.cls[coh]
                rows.append({"relation source": s,
                             "encoder": "all" if s in (GLOVE, CLIP) else ch.label,
                             "cohort": coh,
                             "related, seen together": int(m["related_seen"][idx].sum()),
                             "related, never together": int(m["related_never"][idx].sum()),
                             "informative companion": int(m["informative"][idx].sum())})
    return pd.DataFrame(rows).drop_duplicates()


# ==================================================================== figure
def figure(checks, path, cohort="tail"):
    import matplotlib.pyplot as plt
    sources = [s for s in checks[0].sims] + [RANDOM]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=sk.SURFACE)
    w = 0.8 / len(checks)
    x = np.arange(len(sources))
    for i, ch in enumerate(checks):
        idx = ch.cls[cohort]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = [float(np.nanmean(ch.per_class(s)["confusion"].iloc[idx])) for s in sources]
        ax.bar(x - 0.4 + w * (i + 0.5), v, width=w * 0.92, color=ch.colour, label=ch.label)
    ax.axhline(0, color=sk.MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" (", "\n(").replace(": ", ":\n") for s in sources], fontsize=8.5)
    ax.set_ylabel("How confused the model is with\nthe related classes (percentile points)")
    ax.set_title(f"{cohort.capitalize()} classes: does each relation source name the classes "
                 "the model confuses? (higher = better)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    sk._style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=sk.SURFACE)
    plt.close(fig)
