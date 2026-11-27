"""Diagnostics for the RTN skeleton: why does less confusion not give more AP?

Reads the runs saved by the encoder notebooks and trains nothing. It uses
rtn_skeleton.py read-only (the skeleton's fingerprint does not change).

For every class and every test image without that class (a negative), the
negative is put in one group, by the strongest relation it has to the class
(the same relation types as the skeleton's confusion measure):

    related, seen together   it holds a related class seen with the class
    related, never together  it holds a related class never seen with it
    informative companion    it holds a companion (2+ shared images, lift >= 2)
    generic companion        it holds a frequent companion with low lift
    seen together once       it holds a class seen with it only once
    unrelated                none of these

"AP lost to a group" is how much a class's AP rises when that group's
negatives are taken out of the ranking: exactly the AP those negatives cost.
"""
import contextlib
import json
import os
import warnings

import numpy as np
import pandas as pd

import rtn_skeleton as sk

GROUP_ORDER = (0, 1, 2, 3, -1, 4)       # priority when an image holds several labels
GROUP_NAMES = {0: "related, seen together", 1: "related, never together",
               2: "informative companion", 3: "generic companion",
               -1: "seen together once", 4: "unrelated"}
TARGETS = (0, 1, 2)                     # the negatives RTN gives extra weight
COHORTS = ("tail", "medium", "head")
OLD_RUNS = {"clip": ("coco_mlt/contrastive/runs", "ours_alpha4", "round 6, alpha 4"),
            "resnet50": ("coco_mlt/round7/relation_contrastive_resnet50/runs", "ours_alpha1",
                         "round 7, alpha 1")}
COLOURS = sk.COLOURS
SURFACE, MUTED = sk.SURFACE, sk.MUTED


@contextlib.contextmanager
def quiet():
    """Hide the skeleton's own tables while loading."""
    show = sk._show
    sk._show = lambda obj: None
    try:
        yield
    finally:
        sk._show = show


def _boot(x, rng, n_boot=10000):
    return sk._boot(rng, x, n_boot)


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, int(ok.sum())
    return float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman")), int(ok.sum())


class Diag:
    """The saved runs of one encoder, with the relation groups of every test image."""

    def __init__(self, dataset, encoder, quick_test=False, colour=None):
        with quiet():
            exp = sk.Experiment(dataset, encoder, quick_test=quick_test, allow_cpu=True)
            exp.relations()
            exp.prepare()
        self.exp, self.key, self.label = exp, encoder["key"], encoder["label"]
        self.colour = colour or COLOURS[0]
        path = os.path.join(exp.out, "result.json")
        self.saved_fingerprint = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.saved_fingerprint = json.load(f).get("fingerprint")
        self.runs, missing = {key: {} for key in sk.SHORT}, []
        for key, _, kind, setting in sk.VARIANTS:
            for seed in exp.seeds:
                p = exp._spec(key, kind, setting, seed)["path"]
                if not os.path.exists(p):
                    missing.append(os.path.basename(p))
                    continue
                r = exp.rcr.load_run(p)
                if not exp._matches(r):
                    raise SystemExit(f"{p} does not match this split; rerun the notebook.")
                self.runs[key][seed] = r
        if missing:
            raise SystemExit(f"{self.label}: missing runs ({', '.join(missing)}); run its "
                             "notebook first.")
        exp.runs = self.runs
        with quiet():
            exp.results()
            exp.confusion()
        self.L, self.names, self.cohort = exp.L, exp.names, exp.cohort
        self.Y = exp.Y_test > 0.5
        self.cls = {g: np.flatnonzero(self.cohort == g) for g in COHORTS}
        self.scores = {key: [np.asarray(self.runs[key][s]["test_logits"], dtype=np.float64)
                             for s in exp.seeds] for key in sk.SHORT}
        self.G = self._group_matrix()
        self.neg_group = self._negative_groups()
        self.old, self.old_note = self._old_runs()
        self._lost = {}

    # ------------------------------------------------------------------
    def _group_matrix(self):
        """Relation group of every class pair, as in the skeleton's confusion measure."""
        Y_all = self.exp.Y_all
        shared = Y_all.T.astype(np.int64) @ Y_all.astype(np.int64)
        n = np.diag(shared).copy()
        base = n / len(Y_all)
        G = np.full((self.L, self.L), -9)
        for t in range(self.L):
            for c in range(self.L):
                if c == t:
                    continue
                k = shared[t, c]
                lift = (k / n[t]) / base[c] if n[t] and base[c] else 0.0
                if self.exp.related[t, c]:
                    G[t, c] = 0 if k >= 1 else 1
                elif k == 0:
                    G[t, c] = 4
                elif k >= sk.MIN_SHARED:
                    G[t, c] = 2 if lift >= sk.LIFT_MIN else 3
                else:
                    G[t, c] = -1
        return G

    def _negative_groups(self):
        """L x N: group of each test image for each class (99 = a positive)."""
        N = len(self.Y)
        out = np.full((self.L, N), 4, dtype=np.int16)
        for t in range(self.L):
            done = np.zeros(N, bool)
            for code in GROUP_ORDER:
                cls = np.flatnonzero(self.G[t] == code)
                if cls.size:
                    has = self.Y[:, cls].any(1) & ~done
                    out[t, has] = code
                    done |= has
            out[t, self.Y[:, t]] = 99
        return out

    def _old_runs(self):
        """Earlier RTN runs with CLIP-text relation types, if saved (reference only)."""
        if self.key not in OLD_RUNS:
            return None, None
        folder, prefix, note = OLD_RUNS[self.key]
        runs = []
        for seed in self.exp.seeds:
            p = os.path.join(folder, f"{prefix}_{seed}_{self.exp.cfg_tag}.npz")
            if not os.path.exists(p):
                return None, None
            r = self.exp.rcr.load_run(p)
            if not self.exp._matches(r):
                return None, None
            runs.append(np.asarray(r["test_logits"], dtype=np.float64))
        return runs, note

    # ------------------------------------------------------------------
    def class_ap(self, S):
        return np.array([self.exp.crc.average_precision(S[:, t], self.Y[:, t])
                         if self.Y[:, t].any() else np.nan for t in range(self.L)])

    def lost(self, key):
        """Per seed and class: AP, the AP lost to each group, and to RTN's targets
        together (all in AP points)."""
        if key in self._lost:
            return self._lost[key]
        ap_fn = self.exp.crc.average_precision
        n_s = len(self.scores[key])
        ap = np.full((n_s, self.L), np.nan)
        lost = {code: np.full((n_s, self.L), np.nan) for code in GROUP_ORDER}
        lost_targets = np.full((n_s, self.L), np.nan)
        for si, S in enumerate(self.scores[key]):
            for t in range(self.L):
                y = self.Y[:, t]
                if not y.any():
                    continue
                st, g = S[:, t], self.neg_group[t]
                full = ap_fn(st, y)
                ap[si, t] = full
                for code in GROUP_ORDER:
                    keep = g != code
                    lost[code][si, t] = ap_fn(st[keep], y[keep]) - full
                keep = ~np.isin(g, TARGETS)
                lost_targets[si, t] = ap_fn(st[keep], y[keep]) - full
        out = dict(ap=100 * ap, lost={c: 100 * v for c, v in lost.items()},
                   targets=100 * lost_targets)
        self._lost[key] = out
        return out

    def counts(self):
        """Average number of test negatives per class in each group, per cohort."""
        rows = []
        for coh in COHORTS:
            idx = self.cls[coh]
            row = {"encoder": self.label, "cohort": coh,
                   "positives": round(float(self.Y[:, idx].sum(0).mean()), 1)}
            for code in GROUP_ORDER:
                row[GROUP_NAMES[code]] = round(float((self.neg_group[idx] == code).sum(1).mean()), 1)
            rows.append(row)
        return rows


# ====================================================================== tables
def load(dataset, encoders, quick_test=False):
    diags = []
    for i, enc in enumerate(encoders):
        d = Diag(dataset, enc, quick_test=quick_test, colour=COLOURS[i % len(COLOURS)])
        diags.append(d)
    rows = [dict(encoder=d.label, fingerprint=d.exp.fingerprint,
                 matches_saved_results=d.saved_fingerprint == d.exp.fingerprint,
                 earlier_clip_text_runs=d.old_note or "-") for d in diags]
    table = pd.DataFrame(rows)
    sk._show(table)
    if not table["matches_saved_results"].all():
        raise SystemExit("These runs are not the ones behind the saved results; rerun the "
                         "encoder notebooks with the current rtn_skeleton.py first.")
    return diags


def where_ap_is_lost(diags):
    """Per encoder and cohort: AP, and the AP lost to each group of negatives, for
    MulSupCon and the change with RTN (AP points, averaged over classes and seeds)."""
    rows = []
    for d in diags:
        for coh in COHORTS:
            idx = d.cls[coh]
            for key in sk.SHORT:
                L = d.lost(key)
                row = {"encoder": d.label, "cohort": coh, "variant": sk.SHORT[key],
                       "AP": round(float(np.nanmean(L["ap"][:, idx])), 2)}
                for code in GROUP_ORDER:
                    row[GROUP_NAMES[code]] = round(float(np.nanmean(L["lost"][code][:, idx])), 2)
                row["all RTN targets together"] = round(float(np.nanmean(L["targets"][:, idx])), 2)
                rows.append(row)
    return pd.DataFrame(rows)


def counts_table(diags):
    return pd.DataFrame([r for d in diags for r in d.counts()])


def class_by_class(diags):
    """Tail classes: does a larger confusion drop go with a larger AP gain?"""
    per, summary = [], []
    for d in diags:
        exp, tail = d.exp, d.cls["tail"]
        dgap = exp.rg["rtn"][0, tail] - exp.rg["mulsupcon"][0, tail]
        dap = exp.ap["rtn"][tail] - exp.ap["mulsupcon"][tail]
        lm, lr = d.lost("mulsupcon")["targets"], d.lost("rtn")["targets"]
        dtarget = np.nanmean(lr, axis=0)[tail] - np.nanmean(lm, axis=0)[tail]
        for j, t in enumerate(tail):
            per.append({"encoder": d.label, "class": d.names[t],
                        "test positives": int(d.Y[:, t].sum()),
                        "AP change": round(float(dap[j]), 2),
                        "gap change, C1": round(float(dgap[j]), 2),
                        "change in AP lost to RTN targets": round(float(dtarget[j]), 2)})
        r1, n1 = _spearman(dgap, dap)
        r2, n2 = _spearman(dtarget, dap)
        summary.append({"encoder": d.label, "tail classes": n1,
                        "Spearman: gap change vs AP change": round(r1, 2),
                        "Spearman: change in AP lost to targets vs AP change": round(r2, 2)})
    per = pd.DataFrame(per)
    a, b, c = (per["gap change, C1"].to_numpy(), per["AP change"].to_numpy(),
               per["change in AP lost to RTN targets"].to_numpy())
    r1, n1 = _spearman(a, b)
    r2, _ = _spearman(c, b)
    summary.append({"encoder": "all three", "tail classes": n1,
                    "Spearman: gap change vs AP change": round(r1, 2),
                    "Spearman: change in AP lost to targets vs AP change": round(r2, 2)})
    return pd.DataFrame(summary), per


def positives(diags, seed=0):
    """Did RTN push down true positives? Change (RTN - MulSupCon) in the share of
    negatives a positive outranks, for positives that also hold a related class seen
    together with the class, and for the other positives (percentage points)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in diags:
        for coh in COHORTS:
            change = {"with a look-alike": [], "without": []}
            for t in d.cls[coh]:
                y = d.Y[:, t]
                look = np.flatnonzero(d.G[t] == 0)
                with_l = y & (d.Y[:, look].any(1) if look.size else np.zeros_like(y))
                parts = {"with a look-alike": with_l, "without": y & ~with_l}
                beat = {key: {p: [] for p in parts} for key in sk.SHORT}
                for key in sk.SHORT:
                    for S in d.scores[key]:
                        b = d.exp.crc.beat_rates(S[y, t], S[~y, t])   # one value per positive
                        for p, m in parts.items():
                            mp = m[y]
                            beat[key][p].append(float(b[mp].mean()) if mp.any() else np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # no such positives
                    for p in parts:
                        change[p].append(100 * (np.nanmean(beat["rtn"][p])
                                                - np.nanmean(beat["mulsupcon"][p])))
            row = {"encoder": d.label, "cohort": coh}
            for p, v in change.items():
                v = np.asarray(v, float)
                r = _boot(v, rng)
                row[f"positives {p}: classes"] = r[3]
                row[f"positives {p}: change"] = sk._ci(r)
            rows.append(row)
    return pd.DataFrame(rows)


def earlier_relations(diags):
    """Reference only: earlier RTN runs with relation types from CLIP's class-name text,
    against the CLIP-free skeleton, class by class (same MulSupCon runs)."""
    summary, per = [], []
    for d in diags:
        if d.old is None:
            continue
        tail = d.cls["tail"]
        base = np.nanmean([d.class_ap(S) for S in d.scores["mulsupcon"]], axis=0)
        new = np.nanmean([d.class_ap(S) for S in d.scores["rtn"]], axis=0)
        old = np.nanmean([d.class_ap(S) for S in d.old], axis=0)
        d_old, d_new = 100 * (old - base)[tail], 100 * (new - base)[tail]
        r, n = _spearman(d_old, d_new)
        summary.append({"encoder": d.label, "earlier runs": d.old_note,
                        "tail AP change, earlier relation types": round(float(np.nanmean(d_old)), 2),
                        "tail AP change, GloVe + WordNet": round(float(np.nanmean(d_new)), 2),
                        "Spearman over tail classes": round(r, 2)})
        for j in np.argsort(-(d_old - d_new))[:6]:
            t = tail[j]
            per.append({"encoder": d.label, "class": d.names[t],
                        "AP change, earlier": round(float(d_old[j]), 2),
                        "AP change, GloVe + WordNet": round(float(d_new[j]), 2),
                        "GloVe + WordNet related classes": ", ".join(
                            d.names[c] for c in np.flatnonzero(d.exp.related[t]))})
    return pd.DataFrame(summary), pd.DataFrame(per)


# ===================================================================== figures
def figure_lost(diags, path):
    import matplotlib.pyplot as plt
    codes = list(GROUP_ORDER)
    fig, axes = plt.subplots(1, len(diags), figsize=(5.2 * len(diags), 4.3), facecolor=SURFACE,
                             sharey=True, squeeze=False)
    for ax, d in zip(axes[0], diags):
        tail = d.cls["tail"]
        y = np.arange(len(codes))[::-1]
        for key, col, off in (("mulsupcon", MUTED, 0.18), ("rtn", d.colour, -0.18)):
            L = d.lost(key)
            v = [float(np.nanmean(L["lost"][c][:, tail])) for c in codes]
            ax.barh(y + off, v, height=0.34, color=col, label=sk.SHORT[key])
        ax.set_yticks(y)
        ax.set_yticklabels([GROUP_NAMES[c] for c in codes])
        ax.set_xlabel("Tail AP lost to these negatives (AP points)")
        ax.set_title(d.label, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        sk._style(ax)
    fig.suptitle("Where tail AP is lost: the AP each type of negative costs (higher = costs more)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def figure_classes(per, diags, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    for d in diags:
        sub = per[per["encoder"] == d.label]
        ax.scatter(sub["gap change, C1"], sub["AP change"], s=36, color=d.colour,
                   edgecolor=SURFACE, linewidth=0.8, label=d.label)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Change in related-class confusion, Control 1 (AUC points; left = less)")
    ax.set_ylabel("Change in AP (AP points; up = better)")
    ax.set_title("Tail classes, RTN vs MulSupCon: does less confusion go with more AP?",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    sk._style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
