#!/usr/bin/env python3
"""
Context-reliance check for tail labels
======================================

First experiment for companion-controlled contrastive learning. It uses the
annotations and saved validation predictions you already have, so no model is
trained. It asks two questions:

  1. Positives: are tail labels recognised less well when their usual
     companions are absent from the image?
  2. Negatives: are images that contain a tail label's usual companions, but
     where the tail label is verified absent, given falsely high scores?

If you pass predictions from more than one condition (for example self-only
and exact graph), it also asks:

  3. Where does the difference between conditions come from: images where a
     usual companion is present, or images where it is absent?

Definitions
-----------
* Usual companions of a tail label t: the --top-k labels that co-occur with t
  most often in TRAINING annotations (equivalently, highest P(c | t)), with at
  least --min-cooc shared training images. Validation annotations are never
  used to choose companions. By default the counts come from --train-pos; a
  larger training co-occurrence matrix can be supplied with --cooc.
* In context: at least one usual companion is verified present in the image.
  Out of context: no usual companion is verified present (with --strict: every
  usual companion is verified absent).
* Everything uses verified entries only; unknown annotations are ignored.
* Per-label AUC gaps (AUC = probability a positive is ranked above a
  negative, so group sizes do not matter):
    gap_pos = AUC(in-context positives vs all negatives)
            - AUC(out-of-context positives vs all negatives)
    gap_neg = AUC(all positives vs out-of-context negatives)
            - AUC(all positives vs in-context negatives)
  A positive gap means the model does better when the usual context matches.
* Busy-image confound: images that contain a companion also contain more
  objects, and busier images may score higher for any label. Two independent
  controls address this, and a verdict of CLEAR needs both:
    Control 1 compares in-context and out-of-context images of the same tail
      label that have the same number of other verified-present labels
      (0, 1, 2, 3, 4 or more). It cannot count objects nobody annotated, so
      it can under-correct.
    Control 2 subtracts each image's median score over all labels before
      comparing, which removes an image-wide score boost whether or not the
      objects behind it are annotated.
  Intervals come from resampling whole labels, so images within a label are
  not assumed to be independent.
* Matched control labels (secondary): the same gaps using labels that never
  co-occur with t in training but have similar training frequency to its
  companions. This needs many verified annotations per label and is only
  reported when at least 10 labels qualify.
* Companion type (--companions, needs --label-embeddings): co-occurring
  labels can be scene context (snow for snowboard) or near-identical
  categories (a sibling breed, a parent such as "Bird"), and the two mean
  different things. A label counts as semantically close to a tail label if
  it is among the tail label's top --related-pct % most similar labels by
  text embedding. "context" keeps only companions that are not close;
  "related" keeps only those that are. A context-reliance claim should rest
  on the "context" run. The embeddings' label order is checked using label
  names that share a rare word.

Verdicts are only given when at least 10 labels contribute. With several
seeds per condition, each label's values are averaged over seeds first.

Outputs (in --out)
------------------
  summary.txt                the printed report
  summary.csv                one row per condition and measure
  per_label.csv              one row per condition and tail label
  comparison.csv             differences from the reference condition
  context_reliance.png       figure for questions 1 and 2
  change_by_context.png      figure for question 3 (only with 2+ conditions)

Example
-------
  python context_reliance_check.py \\
    --train-pos training_positives.npz \\
    --val-pos validation_plan_v1/positives.npz \\
    --val-neg validation_plan_v1/negatives.npz \\
    --labels label_frequencies.csv \\
    --pred self_only=preds/self_s42.npy,preds/self_s43.npy,preds/self_s44.npy \\
    --pred exact=preds/exact_s42.npy,preds/exact_s43.npy,preds/exact_s44.npy \\
    --out context_check

(In Windows PowerShell put the whole command on one line and wrap each --pred
value in double quotes, because PowerShell splits unquoted text at commas.)

Prediction files hold validation scores of shape (validation images, labels),
in the same image and label order as the validation annotation matrices.
Logits or probabilities both work; higher must mean "more likely present".
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import sparse, stats

NPZ_KEY_GUESSES = ("scores", "logits", "probs", "probabilities", "predictions",
                   "preds", "pred", "positives", "negatives", "counts",
                   "y", "labels", "arr_0")
NAME_COLUMN_GUESSES = ("display_name", "DisplayName", "label_name", "name",
                       "label")
MIN_LABELS = 10      # fewest contributing labels before a verdict is given
OBJECT_CAP = 4       # object counts used in the adjustment: 0, 1, 2, 3, 4+

# Chart colours (reference palette; blue/orange validated for colour-blind
# separation, grey used deliberately for the de-emphasised series).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check whether tail labels depend on their usual "
                    "co-occurring companions, using saved predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Example", 1)[1] if "Example" in __doc__ else "")
    data = p.add_argument_group("annotations")
    data.add_argument("--train-pos", required=True,
                      help="Training images x labels matrix of verified "
                           "positives (.npz scipy-sparse or dense, .npy, .pt).")
    data.add_argument("--val-pos", required=True,
                      help="Validation images x labels verified positives.")
    data.add_argument("--val-neg", required=True,
                      help="Validation images x labels verified negatives.")
    data.add_argument("--key",
                      help="Array name inside .npz/.pt files that hold "
                           "several arrays.")
    data.add_argument("--cooc",
                      help="Optional labels x labels TRAINING co-occurrence "
                           "count matrix in the same label order, used to "
                           "choose companions (for example counts from a "
                           "larger training manifest). Its label order is "
                           "checked against --train-pos.")
    data.add_argument("--label-embeddings",
                      help="Optional labels x dimensions text embeddings of "
                           "the label names (for example CLIP), in the same "
                           "label order. Needed for --companions context or "
                           "related.")

    preds = p.add_argument_group("predictions")
    preds.add_argument("--pred", action="append", required=True,
                       metavar="NAME=PATH[,PATH...]",
                       help="Validation scores for one condition. Repeat for "
                            "each condition; separate seed files with commas.")
    preds.add_argument("--reference",
                       help="Condition the others are compared with "
                            "(default: the first --pred).")

    tail = p.add_argument_group("which labels are tail labels")
    tail.add_argument("--labels",
                      help="CSV with one row per label, in matrix column "
                           "order.")
    tail.add_argument("--tail-column",
                      help="Column in --labels marking the cohort "
                           "(auto-detected if omitted).")
    tail.add_argument("--tail-value", default="tail",
                      help="Value marking tail labels (case-insensitive; "
                           "default: tail).")
    tail.add_argument("--tail-count-column",
                      help="Alternative: column with ORIGINAL training "
                           "positive counts; tail = counts within "
                           "--tail-range.")
    tail.add_argument("--tail-range", nargs=2, type=int, default=[1, 99],
                      metavar=("MIN", "MAX"))
    tail.add_argument("--tail-indices",
                      help="Alternative: text file with one tail label "
                           "column index per line.")

    opts = p.add_argument_group("analysis settings")
    opts.add_argument("--top-k", type=int, default=3,
                      help="Usual companions per tail label (default 3).")
    opts.add_argument("--min-cooc", type=int, default=2,
                      help="Minimum shared training images for a companion "
                           "(default 2).")
    opts.add_argument("--companions", choices=("all", "context", "related"),
                      default="all",
                      help="all: any co-occurring label (default). context: "
                           "only labels that are NOT semantically close to "
                           "the tail label. related: only labels that are "
                           "(siblings, parents). Uses --label-embeddings.")
    opts.add_argument("--related-pct", type=float, default=5.0,
                      help="A label counts as semantically close to a tail "
                           "label if it is among the tail label's top PCT%% "
                           "most similar labels by text embedding "
                           "(default 5).")
    opts.add_argument("--strict", action="store_true",
                      help="Out of context only if every companion is "
                           "verified absent.")
    opts.add_argument("--control-draws", type=int, default=50)
    opts.add_argument("--control-pool", type=int, default=5,
                      help="Draw each control label from this many "
                           "non-companions nearest in training frequency.")
    opts.add_argument("--min-group", type=int, default=1,
                      help="Minimum images in every group being compared "
                           "(default 1).")
    opts.add_argument("--bootstrap", type=int, default=5000)
    opts.add_argument("--seed", type=int, default=0)
    opts.add_argument("--out", default="context_check")
    opts.add_argument("--no-plots", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _pick_key(keys, path, key):
    keys = list(keys)
    if key and key in keys:
        return key
    if len(keys) == 1:
        return keys[0]
    for k in NPZ_KEY_GUESSES:
        if k in keys:
            return k
    raise SystemExit(f"{path} holds several arrays {keys}; pass --key to "
                     f"choose one.")


def load_matrix(path, key=None):
    """Load a 2-D matrix from .npy, .npz (dense or scipy sparse) or .pt."""
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.keys())
            is_sparse = {"data", "indices", "indptr", "shape"} <= set(keys)
            arr = None if is_sparse else z[_pick_key(keys, path, key)]
        if is_sparse:
            arr = sparse.load_npz(path)
    elif ext in (".pt", ".pth"):
        try:
            import torch
        except ImportError:
            raise SystemExit(f"{path} is a PyTorch file but torch is not "
                             f"installed; save it with numpy.save instead.")
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            obj = obj[_pick_key(obj.keys(), path, key)]
        arr = (obj.detach().cpu().numpy() if hasattr(obj, "detach")
               else np.asarray(obj))
    else:
        raise SystemExit(f"Unsupported file type: {path} (use .npy, .npz, "
                         f".pt or .pth).")
    if arr.ndim != 2:
        raise SystemExit(f"{path} should be a 2-D matrix; got shape "
                         f"{arr.shape}.")
    return arr


def to_bool(M, path):
    if sparse.issparse(M):
        if M.shape[0] * M.shape[1] > 400_000_000:
            raise SystemExit(f"{path} is too large to expand in memory "
                             f"({M.shape}).")
        M = M.toarray()
    M = np.asarray(M)
    if M.dtype.kind == "f":
        M = np.nan_to_num(M, nan=0.0)
    return M != 0


def load_scores(path, key, shape):
    S = load_matrix(path, key)
    if sparse.issparse(S):
        S = S.toarray()
    S = np.asarray(S, dtype=np.float64)
    if S.shape != shape:
        if S.shape == shape[::-1] and shape[0] != shape[1]:
            S = S.T
        else:
            raise SystemExit(f"{path} has shape {S.shape}, but the "
                             f"validation annotations are {shape} "
                             f"(images x labels).")
    if not np.all(np.isfinite(S)):
        raise SystemExit(f"{path} contains NaN or infinite scores.")
    return S


def parse_preds(items):
    preds = OrderedDict()
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--pred expects NAME=PATH[,PATH...], got: "
                             f"{item}")
        name, paths = item.split("=", 1)
        name = name.strip()
        paths = [x.strip() for x in paths.split(",") if x.strip()]
        if not name or not paths:
            raise SystemExit(f"--pred needs a name and at least one file: "
                             f"{item}")
        if name in preds:
            raise SystemExit(f"Condition named twice: {name}")
        preds[name] = paths
    return preds


def resolve_tail(args, L):
    names = [str(i) for i in range(L)]
    df = None
    if args.labels:
        df = pd.read_csv(args.labels)
        if len(df) != L:
            raise SystemExit(f"{args.labels} has {len(df)} rows, but the "
                             f"matrices have {L} label columns.")
        for c in NAME_COLUMN_GUESSES:
            if c in df.columns:
                names = df[c].astype(str).tolist()
                break
    if args.tail_indices:
        idx = np.atleast_1d(np.loadtxt(args.tail_indices, dtype=int))
        how = f"indices listed in {args.tail_indices}"
    elif df is not None and args.tail_count_column:
        if args.tail_count_column not in df.columns:
            raise SystemExit(f"Column '{args.tail_count_column}' not in "
                             f"{args.labels}. Columns: {list(df.columns)}")
        counts = pd.to_numeric(df[args.tail_count_column],
                               errors="coerce").to_numpy()
        lo, hi = args.tail_range
        idx = np.flatnonzero((counts >= lo) & (counts <= hi))
        how = f"'{args.tail_count_column}' between {lo} and {hi}"
    elif df is not None:
        target = args.tail_value.strip().lower()
        col = args.tail_column
        if col is None:
            cands = [c for c in df.columns
                     if df[c].astype(str).str.strip().str.lower()
                     .eq(target).any()]
            if len(cands) != 1:
                raise SystemExit(
                    f"Could not tell which column of {args.labels} marks the "
                    f"tail labels (candidates: {cands}). Columns: "
                    f"{list(df.columns)}. Pass --tail-column and "
                    f"--tail-value, or --tail-count-column.")
            col = cands[0]
        elif col not in df.columns:
            raise SystemExit(f"Column '{col}' not in {args.labels}. "
                             f"Columns: {list(df.columns)}")
        idx = np.flatnonzero(df[col].astype(str).str.strip().str.lower()
                             .eq(target).to_numpy())
        how = f"'{col}' == '{args.tail_value}'"
    else:
        raise SystemExit("Say which labels are tail labels: --labels CSV "
                         "(with a cohort column), --tail-count-column, or "
                         "--tail-indices FILE.")
    idx = np.unique(idx)
    if idx.size == 0:
        raise SystemExit(f"No tail labels found using {how}.")
    if idx.min() < 0 or idx.max() >= L:
        raise SystemExit("Tail label indices fall outside the label range.")
    return idx, names, how


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def beat_rates(a, b):
    """For each score in a: share of scores in b below it (ties count half)."""
    a = np.asarray(a, dtype=float)
    if b.size == 0:
        return np.full(a.size, np.nan)
    sb = np.sort(b)
    left = np.searchsorted(sb, a, side="left")
    right = np.searchsorted(sb, a, side="right")
    return (left + 0.5 * (right - left)) / sb.size


def auc(pos, neg):
    """Probability a positive outranks a negative; ties count one half."""
    if pos.size == 0 or neg.size == 0:
        return np.nan
    return float(beat_rates(pos, neg).mean())


def average_precision(scores, y):
    """Average precision, matching sklearn.metrics.average_precision_score."""
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order].astype(float)
    last = np.r_[np.flatnonzero(np.diff(s)), yy.size - 1]
    tps = np.cumsum(yy)[last]
    fps = (last + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def nanmean(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


EMPTY = dict(n_labels=0, mean=np.nan, ci_low=np.nan, ci_high=np.nan,
             median=np.nan, share_above_zero=np.nan, p_value=np.nan)


def summarize(x, n_boot, rng):
    """Mean over labels, bootstrap CI over labels, Wilcoxon signed-rank p."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    out = dict(EMPTY)
    if n == 0:
        return out
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = np.nan
    if n >= 6 and np.any(x != 0):
        try:
            p = float(stats.wilcoxon(x).pvalue)
        except ValueError:
            p = np.nan
    out.update(n_labels=n, mean=float(x.mean()), ci_low=float(lo),
               ci_high=float(hi), median=float(np.median(x)),
               share_above_zero=float(np.mean(x > 0)), p_value=p)
    return out


def within_label_effect(blocks, n_boot, rng):
    """In-context minus out-of-context beat rate, pooled within labels.

    blocks: list of (y, companion_present, other_objects) arrays, one per
    label, where y is each image's beat rate. Equivalent to a regression of y
    on the companion indicator with one intercept per label. Returns the
    estimate in points (x100), a 95% interval from resampling whole labels,
    and a two-sided bootstrap p-value.
    """
    sxx, sxy, n_rows = [], [], 0
    for y, comp, _ in blocks:
        if y.size < 2:
            continue
        x = comp.astype(float) - comp.mean()
        if np.any(x != 0):
            sxx.append(float(x @ x))
            sxy.append(float(x @ (y - y.mean())))
        n_rows += y.size
    out = dict(EMPTY)
    out["n_images"] = n_rows
    if not sxx:
        return out
    sxx, sxy = np.array(sxx), np.array(sxy)
    n = sxx.size
    est = sxy.sum() / sxx.sum()
    counts = np.array([np.bincount(rng.integers(0, n, n), minlength=n)
                       for _ in range(n_boot)], dtype=float)
    boots = (counts @ sxy) / (counts @ sxx)   # resample whole labels
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    out.update(n_labels=n, mean=100 * est, ci_low=100 * lo,
               ci_high=100 * hi, p_value=min(1.0, p))
    return out


def matched_effect(blocks, n_boot, rng):
    """Companion effect from exact matches on label and object count.

    Within each tail label, images are grouped by the number of other
    labelled objects (0, 1, 2, 3, 4 or more). Inside each group, the mean
    beat rate of in-context images minus that of out-of-context images is
    taken, and the differences are averaged with weights
    n_in * n_out / (n_in + n_out). Nothing is assumed about how object count
    affects scores, and nothing is shared across labels. The 95% interval
    resamples whole labels.
    """
    sums, weights, n_rows = [], [], 0
    for y, comp, objects in blocks:
        level = np.minimum(objects, OBJECT_CAP)
        s = w = 0.0
        for k in np.unique(level):
            in_k = comp & (level == k)
            out_k = ~comp & (level == k)
            a, b = int(in_k.sum()), int(out_k.sum())
            if a and b:
                wk = a * b / (a + b)
                s += wk * (y[in_k].mean() - y[out_k].mean())
                w += wk
                n_rows += a + b
        if w > 0:
            sums.append(s)
            weights.append(w)
    out = dict(EMPTY)
    out["n_images"] = n_rows
    if not sums:
        return out
    sums, weights = np.array(sums), np.array(weights)
    n = sums.size
    est = sums.sum() / weights.sum()
    counts = np.array([np.bincount(rng.integers(0, n, n), minlength=n)
                       for _ in range(n_boot)], dtype=float)
    boots = (counts @ sums) / (counts @ weights)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    out.update(n_labels=n, mean=100 * est, ci_low=100 * lo,
               ci_high=100 * hi, p_value=min(1.0, p))
    return out


# ---------------------------------------------------------------------------
# Companions, controls and gaps
# ---------------------------------------------------------------------------

def cooccurrence_counts(P):
    Ps = sparse.csr_matrix(P, dtype=np.int64)
    C = (Ps.T @ Ps).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    return C


def find_companions(C, t, k, min_cooc, allowed=None):
    """Top-k co-occurring labels of t (optionally only among `allowed`).

    Returns the companions and t's full co-occurrence row (unfiltered, so
    control labels can still be required never to co-occur with t).
    """
    row = np.asarray(C[t].toarray(), dtype=float).ravel()
    row[t] = 0.0
    ranked = row if allowed is None else np.where(allowed, row, 0.0)
    order = np.argsort(-ranked, kind="stable")  # ties: lower index first
    comp = order[:k]
    comp = comp[ranked[comp] >= min_cooc]
    return comp.astype(int), row


def text_similarity(path, key, L):
    """Cosine similarity between label-name embeddings (labels x labels)."""
    E = load_matrix(path, key)
    if sparse.issparse(E):
        E = E.toarray()
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != L:
        if E.shape[1] == L:
            E = E.T
        else:
            raise SystemExit(f"--label-embeddings has shape {E.shape}; "
                             f"expected {L} rows, one per label.")
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(E)):
        raise SystemExit("--label-embeddings has empty or invalid rows.")
    E = E / norms
    return E @ E.T


def related_labels(sim_row, t, pct):
    """Labels among t's top pct% most text-similar labels (excluding t)."""
    L = sim_row.size
    n = max(1, int(round(pct / 100.0 * (L - 1))))
    order = np.argsort(-sim_row, kind="stable")
    order = order[order != t][:n]
    mask = np.zeros(L, dtype=bool)
    mask[order] = True
    return mask


def embedding_order_check(sim, names):
    """Labels whose names share a rare word should be more similar.

    Only words used by at most 2% of labels (and at least 3) count, so common
    words such as "dog" or "car" don't dominate. Returns (z, pairs): how many
    standard deviations above the average similarity those pairs are. Near
    zero suggests the embeddings are listed in a different label order.
    None if too few names share a rare word.
    """
    import re
    from collections import Counter
    words = [set(w for w in re.findall(r"[a-z0-9]+", n.lower())
                 if len(w) >= 3) for n in names]
    df = Counter(w for ws in words for w in ws)
    limit = max(3, int(0.02 * len(names)))
    tokens = [set(w for w in ws if df[w] <= limit) for ws in words]
    rows, cols = np.triu_indices(len(names), k=1)
    sims = sim[rows, cols]
    shared = np.fromiter((bool(tokens[i] & tokens[j])
                          for i, j in zip(rows, cols)), dtype=bool,
                         count=rows.size)
    if shared.sum() < 20 or sims.std() == 0:
        return None
    z = (sims[shared].mean() - sims.mean()) / sims.std()
    return float(z), int(shared.sum())


def draw_controls(row, t, comp, freq, rng, pool):
    """Non-companion labels with training frequency similar to each companion."""
    cand = np.flatnonzero(row == 0)
    cand = cand[(cand != t) & ~np.isin(cand, comp)]
    chosen = []
    for c in comp:
        avail = cand[~np.isin(cand, chosen)]
        if avail.size == 0:
            return None
        dist = np.abs(np.log1p(freq[avail]) - np.log1p(freq[c]))
        nearest = avail[np.argsort(dist, kind="stable")[:pool]]
        chosen.append(int(rng.choice(nearest)))
    return np.array(chosen, dtype=int)


def context_masks(Vpos, Vneg, labels, strict):
    present = Vpos[:, labels].any(axis=1)
    absent = Vneg[:, labels].all(axis=1) if strict else ~present
    return present, absent


def gaps(s, ypos, yneg, present, absent, min_group):
    p_in, p_out = s[ypos & present], s[ypos & absent]
    n_in, n_out = s[yneg & present], s[yneg & absent]
    p_all, n_all = s[ypos], s[yneg]
    ok_pos = min(p_in.size, p_out.size, n_all.size) >= min_group
    ok_neg = min(n_in.size, n_out.size, p_all.size) >= min_group
    a_pi = auc(p_in, n_all) if ok_pos else np.nan
    a_po = auc(p_out, n_all) if ok_pos else np.nan
    a_ni = auc(p_all, n_in) if ok_neg else np.nan
    a_no = auc(p_all, n_out) if ok_neg else np.nan
    return dict(n_pos_in=p_in.size, n_pos_out=p_out.size,
                n_neg_in=n_in.size, n_neg_out=n_out.size,
                auc_pos_in=a_pi, auc_pos_out=a_po, gap_pos=a_pi - a_po,
                auc_neg_in=a_ni, auc_neg_out=a_no, gap_neg=a_no - a_ni)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def fmt(r, test="Wilcoxon"):
    if r["n_labels"] == 0 or not np.isfinite(r["mean"]):
        return "not enough data"
    p = r["p_value"]
    p_txt = f"{test} p = {p:.3g}" if np.isfinite(p) else "p not computed"
    extra = ""
    if np.isfinite(r.get("share_above_zero", np.nan)):
        extra = f", {100 * r['share_above_zero']:.0f}% above zero"
    return (f"{r['mean']:+6.2f}  [95% CI {r['ci_low']:+.2f} to "
            f"{r['ci_high']:+.2f}]  labels {r['n_labels']}{extra}, {p_txt}")


def verdict(r1, r2, positive_meaning):
    """Combine the two confound controls into one reading."""
    for r in (r1, r2):
        if r["n_labels"] < MIN_LABELS or not np.isfinite(r["mean"]):
            return (f"NOT ENOUGH DATA - only {r['n_labels']} labels "
                    f"contribute to a control (need at least {MIN_LABELS}).")
    above = [r["ci_low"] > 0 for r in (r1, r2)]
    below = [r["ci_high"] < 0 for r in (r1, r2)]
    if all(above):
        return f"CLEAR - {positive_meaning}"
    if all(below):
        return "REVERSED - the opposite pattern is clear under both controls."
    if any(above) or any(below):
        return ("MIXED - only one control shows a clear gap, so a "
                "busy-image explanation cannot be ruled out.")
    return "NOT CLEAR - both confidence intervals include zero."


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED, labelcolor=INK_2, length=0,
                   labelsize=9)


def _dot(ax, r, y, color):
    if not np.isfinite(r["mean"]):
        return
    ax.errorbar(r["mean"], y,
                xerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                fmt="o", color=color, ms=8, mec=SURFACE, mew=2,
                elinewidth=2, capsize=0, zorder=3)


def _xlim(ax, rows):
    vals = [0.0]
    for r in rows:
        if np.isfinite(r["mean"]):
            vals += [r["ci_low"], r["ci_high"]]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.08 * span, hi + 0.22 * span)


def _header(fig, height, title, subtitle, handles, x=0.012):
    fig.text(x, 0.975, title, ha="left", va="top", fontsize=13, color=INK)
    fig.text(x, 0.975 - 0.42 / height, subtitle, ha="left", va="top",
             fontsize=9, color=INK_2)
    fig.legend(handles=handles, loc="upper left", ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK_2,
               bbox_to_anchor=(x - 0.007, 0.975 - 0.72 / height))
    fig.tight_layout(rect=[0, 0, 1, 1 - 1.25 / height])


def _key(color, label):
    from matplotlib.lines import Line2D
    return Line2D([], [], marker="o", linestyle="", color=color, ms=8,
                  mec=SURFACE, mew=2, label=label)


def plot_reliance(summ, conds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("pos", "Positives: missed more when companions are absent"),
              ("neg", "Negatives: scored higher when companions are present")]
    height = 2.6 + 0.9 * len(conds)
    fig, axes = plt.subplots(1, 2, figsize=(11, height), sharey=True,
                             facecolor=SURFACE)
    for ax, (side, title) in zip(axes, panels):
        _style_axis(ax)
        ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)
        rows = []
        for i, c in enumerate(conds):
            series = [(summ[(c, f"matched_{side}")], i - 0.22, BLUE, True),
                      (summ[(c, f"centred_{side}")], i, ORANGE, True),
                      (summ[(c, f"unadjusted_{side}")], i + 0.22, MUTED,
                       False)]
            for r, y, color, label in series:
                rows.append(r)
                _dot(ax, r, y, color)
                if label and np.isfinite(r["mean"]):
                    ax.annotate(f"{r['mean']:+.1f}", xy=(r["ci_high"], y),
                                xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=9, color=INK_2)
        _xlim(ax, rows)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("Gap, in context minus out of context (AUC points)",
                      fontsize=9, color=INK_2)
    axes[0].set_yticks(range(len(conds)))
    axes[0].set_yticklabels(conds)
    axes[0].set_ylim(len(conds) - 0.5, -0.5)
    _header(fig, height, "Do tail labels depend on their usual companions?",
            "Within-label estimates with 95% intervals from resampling "
            "labels. Above zero means the model does better when the usual "
            "context matches.",
            [_key(BLUE, "Matched on number of objects"),
             _key(ORANGE, "Image-centred scores"),
             _key(MUTED, "No control")])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_change(comp_summ, others, ref, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = 2.4 + 0.7 * len(others)
    fig, ax = plt.subplots(figsize=(8, height), facecolor=SURFACE)
    _style_axis(ax)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)
    rows = []
    for i, c in enumerate(others):
        r_in = comp_summ[(c, "change_auc_pos_in")]
        r_out = comp_summ[(c, "change_auc_pos_out")]
        rows += [r_in, r_out]
        _dot(ax, r_in, i - 0.15, BLUE)
        _dot(ax, r_out, i + 0.15, ORANGE)
        for r, y in ((r_in, i - 0.15), (r_out, i + 0.15)):
            if np.isfinite(r["mean"]):
                ax.annotate(f"{r['mean']:+.1f}", xy=(r["ci_high"], y),
                            xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK_2)
    _xlim(ax, rows)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels(others)
    ax.set_ylim(len(others) - 0.5, -0.5)
    ax.set_xlabel(f"Change in tail-positive AUC versus {ref} (points)",
                  fontsize=9, color=INK_2)
    _header(fig, height, f"Where the change from {ref} comes from",
            "Mean over tail labels with 95% bootstrap CI.",
            [_key(BLUE, "Positives with a usual companion"),
             _key(ORANGE, "Positives without one")], x=0.015)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    say = Report()
    draw_rng = np.random.default_rng(args.seed)
    boot_rng = np.random.default_rng(args.seed + 1)

    # Annotations ----------------------------------------------------------
    P_train = to_bool(load_matrix(args.train_pos, args.key), args.train_pos)
    V_pos = to_bool(load_matrix(args.val_pos, args.key), args.val_pos)
    V_neg = to_bool(load_matrix(args.val_neg, args.key), args.val_neg)
    L = P_train.shape[1]
    if V_pos.shape != V_neg.shape:
        raise SystemExit(f"Validation positives {V_pos.shape} and negatives "
                         f"{V_neg.shape} differ in shape.")
    if V_pos.shape[1] != L:
        raise SystemExit(f"Training has {L} label columns but validation has "
                         f"{V_pos.shape[1]}.")
    conflicts = int((V_pos & V_neg).sum())
    if conflicts:
        V_pos, V_neg = V_pos & ~V_neg, V_neg & ~V_pos
    tail_idx, names, how = resolve_tail(args, L)
    freq = P_train.sum(axis=0).astype(float)
    objects_in_image = V_pos.sum(axis=1)

    C_train = cooccurrence_counts(P_train)
    if args.cooc:
        C = load_matrix(args.cooc, args.key)
        C = sparse.csr_matrix(C, dtype=np.float64)
        if C.shape != (L, L):
            raise SystemExit(f"--cooc has shape {C.shape}; expected "
                             f"({L}, {L}).")
        C.setdiag(0)
        C.eliminate_zeros()
        rows_nz, cols_nz = C_train.nonzero()
        shared = float(np.mean(np.asarray(C[rows_nz, cols_nz]).ravel() > 0)) \
            if rows_nz.size else np.nan
        density = C.nnz / float(L * (L - 1))
        if not np.isfinite(shared) or shared < 0.8:
            raise SystemExit(
                f"Label-order check failed: only {100 * shared:.1f}% of the "
                f"label pairs that co-occur in --train-pos also co-occur in "
                f"--cooc (a random order would give about "
                f"{100 * density:.1f}%). The two files probably list labels "
                f"in different orders, so --cooc cannot be used as it is.")
        cooc_source = args.cooc
        order_note = (f"Label-order check passed: {100 * shared:.1f}% of "
                      f"pairs co-occurring in --train-pos also co-occur in "
                      f"--cooc (random order: about {100 * density:.1f}%).")
    else:
        C = C_train
        cooc_source = "training positives (--train-pos)"
        order_note = None

    # Semantic closeness of labels (optional) ------------------------------
    sim, emb_note = None, None
    if args.label_embeddings:
        sim = text_similarity(args.label_embeddings, args.key, L)
        if names != [str(i) for i in range(L)]:
            check = embedding_order_check(sim, names)
            if check is None:
                emb_note = ("Embedding order check skipped: too few label "
                            "names share a word.")
            else:
                z, n_pairs = check
                if z < 0.5:
                    raise SystemExit(
                        f"Embedding label-order check failed: labels whose "
                        f"names share a word are only {z:.2f} SD more "
                        f"similar than average ({n_pairs} pairs). The "
                        f"embeddings are probably in a different label "
                        f"order from --labels.")
                emb_note = (f"Embedding order check passed: labels whose "
                            f"names share a word are {z:.1f} SD more similar "
                            f"than average ({n_pairs} pairs).")
        else:
            emb_note = ("Embedding order check skipped: --labels has no "
                        "name column.")
    elif args.companions != "all":
        raise SystemExit("--companions context or related needs "
                         "--label-embeddings.")

    # Companions, control draws and per-image design ----------------------
    labels_info = []
    close_counts = [0, 0]   # unfiltered companions that are close, total
    for t in tail_idx:
        allowed = None
        if sim is not None:
            close = related_labels(sim[t], t, args.related_pct)
            unfiltered, _ = find_companions(C, t, args.top_k, args.min_cooc)
            close_counts[0] += int(close[unfiltered].sum())
            close_counts[1] += int(unfiltered.size)
            if args.companions == "context":
                allowed = ~close
            elif args.companions == "related":
                allowed = close
        comp, row = find_companions(C, t, args.top_k, args.min_cooc, allowed)
        rec = dict(t=int(t), comp=comp, masks=None, ctrl=[])
        if comp.size:
            present, absent = context_masks(V_pos, V_neg, comp, args.strict)
            rec["masks"] = (present, absent)
            yp, yn = V_pos[:, t], V_neg[:, t]
            rec["pos_rows"] = np.flatnonzero(yp & (present | absent))
            rec["neg_rows"] = np.flatnonzero(yn & (present | absent))
            rec["pos_comp"] = present[rec["pos_rows"]]
            rec["neg_comp"] = present[rec["neg_rows"]]
            rec["pos_obj"] = np.minimum(
                objects_in_image[rec["pos_rows"]] - 1, OBJECT_CAP)
            rec["neg_obj"] = np.minimum(
                objects_in_image[rec["neg_rows"]], OBJECT_CAP)
            for _ in range(args.control_draws):
                ctrl = draw_controls(row, t, comp, freq, draw_rng,
                                     args.control_pool)
                if ctrl is not None:
                    rec["ctrl"].append(context_masks(V_pos, V_neg, ctrl,
                                                     args.strict))
        labels_info.append(rec)
    with_comp = [r for r in labels_info if r["comp"].size]

    say("Context-reliance check for tail labels")
    say("=" * 38)
    say(f"Tail labels: {tail_idx.size} ({how})")
    say(f"Companions from: {cooc_source}; top {args.top_k} with at least "
        f"{args.min_cooc} shared training images")
    if order_note:
        say(order_note)
    mode_text = {
        "all": "all co-occurring labels",
        "context": (f"context only (labels in the tail label's top "
                    f"{args.related_pct:g}% most text-similar excluded)"),
        "related": (f"related only (labels in the tail label's top "
                    f"{args.related_pct:g}% most text-similar)"),
    }
    say(f"Companion type: {mode_text[args.companions]}")
    if emb_note:
        say(emb_note)
    if sim is not None and close_counts[1]:
        say(f"Without filtering, {close_counts[0]} of {close_counts[1]} "
            f"usual companions ({100 * close_counts[0] / close_counts[1]:.0f}"
            f"%) are semantically close to their tail label (top "
            f"{args.related_pct:g}% by text similarity).")
    say(f"Tail labels with at least one companion: {len(with_comp)} of "
        f"{tail_idx.size}")
    say(f"Out of context means: "
        f"{'every companion verified absent' if args.strict else 'no companion verified present'}")
    say(f"Validation: {V_pos.shape[0]} images; verified positives "
        f"{int(V_pos.sum())}, verified negatives {int(V_neg.sum())}")
    if conflicts:
        say(f"Note: {conflicts} image-label pairs were marked both positive "
            f"and negative and were treated as unknown.")
    if not with_comp:
        say("No tail label has a companion at these settings. Try "
            "--min-cooc 1, or a larger training co-occurrence matrix via "
            "--cooc.")
        say.save(os.path.join(args.out, "summary.txt"))
        return 1
    if len(with_comp) < 0.5 * tail_idx.size:
        say("Fewer than half the tail labels have a companion. A larger "
            "training co-occurrence matrix via --cooc would help.")

    # Per-condition metrics -----------------------------------------------
    preds = parse_preds(args.pred)
    ref = args.reference or next(iter(preds))
    if ref not in preds:
        raise SystemExit(f"--reference {ref} is not one of the --pred names "
                         f"{list(preds)}.")
    meta = pd.DataFrame({
        "label_index": tail_idx,
        "label_name": [names[t] for t in tail_idx],
        "train_positives": freq[tail_idx].astype(int),
        "companion_type": args.companions,
        "companions": ["|".join(names[c] for c in r["comp"])
                       for r in labels_info],
        "control_draws_used": [len(r["ctrl"]) for r in labels_info],
    })

    per_label = OrderedDict()
    blocks = OrderedDict()     # condition -> side -> list of (y, comp, obj)
    blocks_c = OrderedDict()   # the same, from image-centred scores
    for cond, paths in preds.items():
        frames = []
        y_sum = {r["t"]: [np.zeros(r["pos_rows"].size),
                          np.zeros(r["neg_rows"].size)] for r in with_comp}
        y_sum_c = {r["t"]: [np.zeros(r["pos_rows"].size),
                            np.zeros(r["neg_rows"].size)] for r in with_comp}
        for path in paths:
            S = load_scores(path, args.key, V_pos.shape)
            # Each image's overall score level (median over all labels).
            S_c = S - np.median(S, axis=1, keepdims=True)
            rows = []
            for rec in labels_info:
                t = rec["t"]
                s, yp, yn = S[:, t], V_pos[:, t], V_neg[:, t]
                verified = yp | yn
                row = dict(label_index=t,
                           ap=average_precision(s[verified], yp[verified]))
                if rec["comp"].size:
                    present, absent = rec["masks"]
                    row.update(gaps(s, yp, yn, present, absent,
                                    args.min_group))
                    cp, cn = [], []
                    for c_present, c_absent in rec["ctrl"]:
                        g = gaps(s, yp, yn, c_present, c_absent,
                                 args.min_group)
                        cp.append(g["gap_pos"])
                        cn.append(g["gap_neg"])
                    row["ctrl_gap_pos"] = nanmean(cp)
                    row["ctrl_gap_neg"] = nanmean(cn)
                    y_sum[t][0] += beat_rates(s[rec["pos_rows"]], s[yn])
                    y_sum[t][1] += beat_rates(s[rec["neg_rows"]], s[yp])
                    sc = S_c[:, t]
                    y_sum_c[t][0] += beat_rates(sc[rec["pos_rows"]], sc[yn])
                    y_sum_c[t][1] += beat_rates(sc[rec["neg_rows"]], sc[yp])
                rows.append(row)
            frames.append(pd.DataFrame(rows))
        df = (pd.concat(frames)
              .groupby("label_index", sort=False).mean(numeric_only=True)
              .reset_index())
        for col in ("gap_pos", "gap_neg", "ctrl_gap_pos", "ctrl_gap_neg",
                    "auc_pos_in", "auc_pos_out", "auc_neg_in",
                    "auc_neg_out", "n_pos_in", "n_pos_out", "n_neg_in",
                    "n_neg_out"):
            if col not in df:
                df[col] = np.nan
        df["n_seeds"] = len(paths)
        per_label[cond] = meta.merge(df, on="label_index", how="left")
        n = len(paths)
        for store, sums in ((blocks, y_sum), (blocks_c, y_sum_c)):
            store[cond] = {
                "pos": [(sums[r["t"]][0] / n, r["pos_comp"], r["pos_obj"])
                        for r in with_comp if r["pos_rows"].size and
                        V_neg[:, r["t"]].any()],
                "neg": [(sums[r["t"]][1] / n, r["neg_comp"], r["neg_obj"])
                        for r in with_comp if r["neg_rows"].size and
                        V_pos[:, r["t"]].any()],
            }

    # Summaries -----------------------------------------------------------
    summ, summ_rows = {}, []

    def record(cond, measure, r):
        summ[(cond, measure)] = r
        summ_rows.append(dict(condition=cond, measure=measure, **r))

    for cond, df in per_label.items():
        for side in ("pos", "neg"):
            record(cond, f"per_label_gap_{side}",
                   summarize(100 * df[f"gap_{side}"].to_numpy(),
                             args.bootstrap, boot_rng))
            record(cond, f"unadjusted_{side}",
                   within_label_effect(blocks[cond][side], args.bootstrap,
                                       boot_rng))
            record(cond, f"matched_{side}",
                   matched_effect(blocks[cond][side], args.bootstrap,
                                  boot_rng))
            record(cond, f"centred_{side}",
                   within_label_effect(blocks_c[cond][side], args.bootstrap,
                                       boot_rng))
            specific = (df[f"gap_{side}"] - df[f"ctrl_gap_{side}"]).to_numpy()
            record(cond, f"control_label_specific_{side}",
                   summarize(100 * specific, args.bootstrap, boot_rng))
        tail_map = 100 * nanmean(df["ap"].to_numpy())
        summ[(cond, "tail_mAP")] = tail_map
        summ_rows.append(dict(condition=cond, measure="tail_mAP",
                              n_labels=int(np.isfinite(df["ap"]).sum()),
                              mean=tail_map))

    say()
    say("Sanity check: tail mAP from these predictions (should match the "
        "values you reported for the same runs)")
    for cond in per_label:
        say(f"  {cond:<16} {summ[(cond, 'tail_mAP')]:.4f}%  "
            f"({len(preds[cond])} file(s) averaged)")

    side_text = {"pos": "Positives", "neg": "Negatives"}
    for cond in per_label:
        say()
        say(f"Condition: {cond}   (gaps in AUC points; above zero = better "
            f"when the usual context matches)")
        for side in ("pos", "neg"):
            say(f"  {side_text[side]}")
            say(f"    Per-label gap, mean over labels      "
                f"{fmt(summ[(cond, f'per_label_gap_{side}')])}")
            say(f"    Pooled within label, no control      "
                f"{fmt(summ[(cond, f'unadjusted_{side}')], 'bootstrap')}")
            say(f"    Control 1: matched on no. of objects "
                f"{fmt(summ[(cond, f'matched_{side}')], 'bootstrap')}")
            say(f"    Control 2: image-centred scores      "
                f"{fmt(summ[(cond, f'centred_{side}')], 'bootstrap')}")
            r_ctrl = summ[(cond, f"control_label_specific_{side}")]
            if r_ctrl["n_labels"] >= MIN_LABELS:
                say(f"    Minus matched control labels         {fmt(r_ctrl)}")
            else:
                say(f"    Matched control labels: skipped, only "
                    f"{r_ctrl['n_labels']} labels have enough verified "
                    f"annotations")
        say("  Verdict, positives: " + verdict(
            summ[(cond, "matched_pos")], summ[(cond, "centred_pos")],
            "tail positives are missed more often when their usual "
            "companions are absent, under both controls."))
        say("  Verdict, negatives: " + verdict(
            summ[(cond, "matched_neg")], summ[(cond, "centred_neg")],
            "images with the usual companions but without the tail label "
            "get higher scores, under both controls."))

    # Comparison with the reference condition -----------------------------
    others = [c for c in per_label if c != ref]
    comp_summ, comp_rows = {}, []
    if others:
        base = per_label[ref].set_index("label_index")
        say()
        say(f"Change versus {ref} (AUC points for tail positives and "
            f"negatives, AP points overall)")
        pretty = OrderedDict([
            ("change_auc_pos_in", "Positives with a usual companion"),
            ("change_auc_pos_out", "Positives without one"),
            ("in_minus_out", "Difference (with minus without)"),
            ("change_auc_neg_in", "Negatives with a usual companion"),
            ("change_auc_neg_out", "Negatives without one"),
            ("change_ap", "Tail AP (all tail labels)"),
        ])
        for cond in others:
            cur = per_label[cond].set_index("label_index")
            diffs = {
                "change_auc_pos_in": cur["auc_pos_in"] - base["auc_pos_in"],
                "change_auc_pos_out": cur["auc_pos_out"] - base["auc_pos_out"],
                "change_auc_neg_in": cur["auc_neg_in"] - base["auc_neg_in"],
                "change_auc_neg_out": cur["auc_neg_out"] - base["auc_neg_out"],
                "change_ap": cur["ap"] - base["ap"],
            }
            diffs["in_minus_out"] = (diffs["change_auc_pos_in"]
                                     - diffs["change_auc_pos_out"])
            say(f"  {cond}:")
            for key, label in pretty.items():
                r = summarize(100 * diffs[key].to_numpy(), args.bootstrap,
                              boot_rng)
                comp_summ[(cond, key)] = r
                comp_rows.append(dict(condition=cond, reference=ref,
                                      measure=key, **r))
                say(f"    {label:<34} {fmt(r)}")
            r = comp_summ[(cond, "in_minus_out")]
            if r["n_labels"] < MIN_LABELS:
                say("    Reading: not enough labels to judge.")
            elif r["ci_low"] > 0:
                say(f"    Reading: the change from {ref} is concentrated in "
                    f"images where a usual companion is present.")
            elif r["ci_high"] < 0:
                say(f"    Reading: the change from {ref} is concentrated in "
                    f"images without a usual companion.")
            else:
                say("    Reading: no clear difference between images with "
                    "and without a usual companion.")

    # Files ---------------------------------------------------------------
    pd.concat([df.assign(condition=c) for c, df in per_label.items()]) \
        .to_csv(os.path.join(args.out, "per_label.csv"), index=False)
    pd.DataFrame(summ_rows).to_csv(os.path.join(args.out, "summary.csv"),
                                   index=False)
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(args.out, "comparison.csv"), index=False)

    if not args.no_plots:
        try:
            plot_reliance(summ, list(per_label), os.path.join(
                args.out, "context_reliance.png"))
            if others:
                plot_change(comp_summ, others, ref, os.path.join(
                    args.out, "change_by_context.png"))
        except ImportError:
            say("matplotlib is not installed; skipped the figures.")

    say()
    say("Settings: " + " ".join(f"--{k.replace('_', '-')} {v}"
                                for k, v in vars(args).items()
                                if k not in ("pred",)))
    say.save(os.path.join(args.out, "summary.txt"))
    print(f"\nSaved results to {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
