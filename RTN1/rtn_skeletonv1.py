"""The RTN skeleton: the fixed method and the fixed protocol, in one module.

Experiment notebooks only say which dataset and which encoder to use, and
call this module. Everything else lives here, so every experiment runs
exactly the same skeleton. To change the method, change a component here
and raise SKELETON_VERSION by one: the run names contain the version, so
no run of an older skeleton is ever reused, and every saved result carries
a fingerprint of this file, so results of different skeletons are never
compared with each other.

Components (the method; nothing here uses CLIP):
  1. Relation source. Each class name becomes the GloVe vectors of its name
     plus the GloVe vectors of its WordNet definition, weighted equally;
     class similarity is the cosine between these vectors.
  2. Relation types. Class c is related to class k if c is among k's top 5%
     most similar names. From the training labels:
         R = 1     related and seen together in training
             0.5   related but never seen together
             0.25  informative companion: not related, at least 2 shared
                   training images and at least twice as often as chance
             0     anything else
  3. Where RTN acts. The negatives of MulSupCon's contrastive loss, with
     weight w = 1 + ALPHA * R, ALPHA = 1 (not tuned).

Host: MulSupCon's official pipeline, as implemented and checked in round 6
(context_reliance_check.py, relation_contrastive_method.py,
relation_contrastive_runs.py), unchanged. With every R = 0, RTN is exactly
MulSupCon, which is also the baseline.

Protocol (how every experiment is run and judged):
  - the training images are split 80 / 20 into training and dev images
    (iterative stratification, seed 0); relation types come from the
    training part only;
  - seeds 42, 43, 44 for MulSupCon and for RTN;
  - test AP per class group, and the related-class gap on the skeleton's own
    related pairs (Control 1: matched number of objects; Control 2: centred
    scores), with 95% bootstrap intervals over classes;
  - decision rule (rounds 1-2): RTN works if tail AP improves (interval
    above zero and higher tail mAP in at least 2 of 3 seeds), the
    related-class gap shrinks under both controls, and head AP drops by at
    most 1 point. PARTLY: one of the first two, with the third.
"""
import filecmp
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

SKELETON_VERSION = 1

# ======================================================= component 1 ====
# Relation source: GloVe + WordNet (no CLIP)
GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz")
COCO_WORDNET_URL = ("https://raw.githubusercontent.com/lvis-dataset/lvis-api/master/data/"
                    "coco_to_synset.json")
STOPWORDS = frozenset("a an the of or and to for in on at by with from as is are be that "
                      "which its it into used usually often one more than consisting".split())


def cosine_similarity(E):
    """K x K cosine similarity between the rows of E (one per class)."""
    E = np.asarray(E, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(E)):
        raise ValueError("class embeddings have empty or invalid rows")
    E = E / norms
    return E @ E.T


def name_words(name):
    """Lower-case words of a class name ('traffic light' -> ['traffic', 'light'])."""
    return re.findall(r"[a-z0-9]+", name.lower())


def load_word_vectors(path, words):
    """Vectors of the given words from a GloVe text file (plain or .gz)."""
    want, found = set(words), {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            word, _, rest = line.rstrip("\n").partition(" ")
            if i == 0 and len(rest.split()) == 1:
                continue                      # word2vec header
            if word in want and word not in found:
                found[word] = np.array(rest.split(), dtype=np.float64)
                if len(found) == len(want):
                    break
    return found


def unit_average(words, vectors):
    """Unit-length average of the vectors of the words found; None if none."""
    have = [vectors[w] for w in words if w in vectors]
    if not have:
        return None
    v = np.mean(have, axis=0)
    return v / np.linalg.norm(v)


def text_embeddings(names, vectors, definitions, stopwords=STOPWORDS):
    """One vector per class: unit-length average of the name's words plus
    unit-length average of the definition's words (stopwords dropped).
    Returns (K x d array, {class name: [words not found]})."""
    rows, missing = [], {}
    for n, d in zip(names, definitions):
        ws = name_words(n)
        dws = [w for w in name_words(d) if w not in stopwords]
        lost = [w for w in ws + dws if w not in vectors]
        if lost:
            missing[n] = lost
        parts = [v for v in (unit_average(ws, vectors), unit_average(dws, vectors))
                 if v is not None]
        if not parts:
            raise ValueError(f"no word of '{n}' or its definition has a vector")
        rows.append(np.sum(parts, axis=0))
    return np.vstack(rows), missing


def glove_text_embeddings(names, glove_path, definitions):
    """text_embeddings with vectors read from a GloVe file."""
    words = {w for n in names for w in name_words(n)}
    words |= {w for d in definitions for w in name_words(d)}
    return text_embeddings(names, load_word_vectors(glove_path, words), definitions)


# ======================================================= component 2 ====
# Relation types
VALUES = (1.0, 0.5, 0.25)
RELATED_PCT = 5
MIN_SHARED = 2
LIFT_MIN = 2.0


def related_mask(sim, pct=RELATED_PCT):
    """K x K bool: row k marks k's top pct% most similar other classes
    (at least one), ties broken by class order."""
    sim = np.asarray(sim, dtype=np.float64)
    K = sim.shape[0]
    n = max(1, int(round(pct / 100.0 * (K - 1))))
    out = np.zeros((K, K), dtype=bool)
    for k in range(K):
        order = np.argsort(-sim[k], kind="stable")
        out[k, order[order != k][:n]] = True
    return out


def relation_types(Y, sim, pct=RELATED_PCT, min_shared=MIN_SHARED, lift_min=LIFT_MIN):
    """Relation values R (K x K) from training labels Y and class similarity
    sim, with the boolean masks behind them."""
    Yi = (np.asarray(Y) > 0.5).astype(np.int64)
    shared = Yi.T @ Yi
    n = np.diag(shared).copy()
    base = n / len(Yi)
    related = related_mask(sim, pct)
    K = shared.shape[0]
    off = ~np.eye(K, dtype=bool)
    rel_seen = related & (shared >= 1) & off
    rel_never = related & (shared == 0) & off
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (shared / np.maximum(n, 1)[:, None]) / np.where(base > 0, base, np.inf)[None, :]
    informative = (~related) & off & (shared >= min_shared) & (n[:, None] > 0) \
        & (base[None, :] > 0) & (lift >= lift_min)
    R = np.zeros((K, K), dtype=np.float32)
    R[informative] = 0.25
    R[rel_never] = 0.5
    R[rel_seen] = 1.0
    return R, dict(related=related, related_seen=rel_seen, related_never=rel_never,
                   informative=informative)


def relation_scores(R, Y):
    """N x K: for image j and class k, the largest R[k, c] over j's labels c."""
    Yf = np.asarray(Y, dtype=np.float32)
    score = np.zeros((Yf.shape[0], R.shape[0]), dtype=np.float32)
    for v in np.unique(R[R > 0]):
        has = (Yf @ (R == v).T.astype(np.float32)) > 0.5
        score = np.maximum(score, v * has)
    return score


# ======================================================= component 3 ====
# Where RTN acts: MulSupCon's negatives
ALPHA = 1.0


def negative_weights(R, Y_negatives, alpha=ALPHA):
    """K x M weights: w[k, j] = 1 + alpha * max over negative j's labels c of
    R[k, c] (w = 1 + R with alpha = 1; alpha = 0 gives MulSupCon)."""
    return (1.0 + alpha * relation_scores(R, Y_negatives).T).astype(np.float32)


# ========================================================== protocol ====
HOST_VERSION = "2026-10-30"          # round 6 code (MulSupCon, checked against the official code)
SEEDS = (42, 43, 44)
DEV_FRAC, SPLIT_SEED = 0.2, 0
BOOTSTRAP = 10000
COUNT_CAP, MIN_NEG = 10, 20
TYPES = ("related, seen together", "related, never together",
         "informative companion", "generic companion")
GROUP_NAMES = ("tail", "medium", "head", "all")
AMBIGUOUS = ("mouse", "tie", "remote", "kite", "hot dog", "orange", "keyboard")
DECISION_RULE = ("RTN works if, against MulSupCon, tail AP improves (95% interval above zero "
                 "and higher tail mAP in at least 2 of 3 seeds), the related-class gap shrinks "
                 "(intervals below zero under both controls) and head AP drops by at most 1 "
                 "point. PARTLY: one of the first two, with the third.")
COLOURS = ("#2a78d6", "#1f9e74", "#eb6834", "#8a63d2", "#b8860b")
SURFACE, MUTED, INK = "#fcfcfb", "#898781", "#52514e"
VARIANTS = (("mulsupcon", "MulSupCon (no relation types)", "uniform", {}),
            ("rtn", "RTN skeleton: MulSupCon + relation-typed negatives", "relation",
             {"alpha": ALPHA}))
SHORT = {"mulsupcon": "MulSupCon", "rtn": "RTN"}


def skeleton_table():
    """The fixed components and protocol, for display."""
    return pd.DataFrame([
        ("skeleton version", str(SKELETON_VERSION)),
        ("relation source", "GloVe 6B 300-d vectors of the class name + its WordNet "
                            "definition, equal weight (no CLIP)"),
        ("related classes", f"top {RELATED_PCT}% most similar names"),
        ("relation values", f"1 related + seen together, 0.5 related + never together, 0.25 "
                            f"informative companion (>= {MIN_SHARED} shared images, lift >= "
                            f"{LIFT_MIN:g}), 0 otherwise; from the training part of the labels"),
        ("where RTN acts", f"MulSupCon's negatives, weight w = 1 + {ALPHA:g} x R"),
        ("host and baseline", f"MulSupCon's official pipeline (round 6 code {HOST_VERSION})"),
        ("split and seeds", f"{1 - DEV_FRAC:.0%} / {DEV_FRAC:.0%} training / dev split "
                            f"(seed {SPLIT_SEED}); seeds {', '.join(map(str, SEEDS))}"),
        ("measurements", "test AP per class group; related-class gap on the skeleton's own "
                         "related pairs (Controls 1 and 2); 95% bootstrap intervals over classes"),
        ("decision rule", DECISION_RULE),
    ], columns=["component", "fixed choice"])


# ------------------------------------------------------------ helpers ----
def _show(obj):
    try:
        from IPython.display import display
        display(obj)
    except ImportError:
        print(obj.to_string() if hasattr(obj, "to_string") else obj)


def _show_image(path):
    try:
        from IPython.display import Image, display
        display(Image(filename=path))
    except ImportError:
        pass


def _download(url, path):
    if os.path.exists(path):
        return
    print(f"Downloading {url} (once; large files take a few minutes) ...", flush=True)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)


def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(*arrays, n=12):
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes() if isinstance(a, np.ndarray) else
                 json.dumps(a, sort_keys=True, default=str).encode())
    return h.hexdigest()[:n]


def _place(src, dst):
    """Hard-link (or copy) src to dst, unless dst already holds the same file."""
    if os.path.exists(dst):
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        if filecmp.cmp(src, dst, shallow=False):
            return
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _ci(r):
    return "-" if not np.isfinite(r[0]) else f"{r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}]"


def _boot(rng, x, n_boot):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    b = rng.choice(x, size=(n_boot, x.size)).mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return (float(x.mean()), float(lo), float(hi), int(x.size))


def _pair_gaps(b, IN_, OUT_, H):
    a, o = H.T @ IN_, H.T @ OUT_
    Hb = H * b[:, None]
    sa, so = Hb.T @ IN_, Hb.T @ OUT_
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = sa.sum(0) / a.sum(0) - so.sum(0) / o.sum(0)
        ok = (a > 0) & (o > 0)
        w = np.where(ok, a * o / np.maximum(a + o, 1), 0.0)
        diff = np.where(ok, sa / np.maximum(a, 1) - so / np.maximum(o, 1), 0.0)
        matched = (w * diff).sum(0) / w.sum(0)
    return 100 * matched, 100 * raw


def _style(ax):
    from matplotlib.ticker import MaxNLocator
    ax.set_facecolor(SURFACE)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw(results, path, title):
    """Figure of one or more encoders' results (dicts saved by Experiment.save)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = len(results)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 3.2 + 0.8 * max(n, 2)), facecolor=SURFACE,
                                   gridspec_kw=dict(width_ratios=[1, 1.2], wspace=0.55))
    yy = np.arange(len(GROUP_NAMES))[::-1]
    offs = np.linspace(0.18, -0.18, n) if n > 1 else [0.0]
    for off, res in zip(offs, results):
        for y, g in zip(yy, GROUP_NAMES):
            r = res["ap_change"][g]
            if np.isfinite(r[0]):
                ax0.hlines(y + off, r[1], r[2], color=res["colour"], lw=2.2)
                ax0.plot(r[0], y + off, "o", color=res["colour"], ms=6.5, mec=SURFACE, mew=1.2)
    ax0.axvline(0, color=MUTED, lw=1)
    ax0.set_yticks(yy)
    ax0.set_yticklabels([f"{g} classes" for g in GROUP_NAMES])
    ax0.set_xlabel("AP change, RTN vs MulSupCon (AP points, 95% interval)")
    ax0.set_title("(a) Test AP by class group (above 0 = RTN better)", loc="left", fontsize=10)
    yt = np.arange(len(TYPES))[::-1]
    half = min(0.36, 0.12 * (2 * n - 1))         # spread of the 2n lines around each type
    for k, res in enumerate(results):
        for i, mk in ((0, "o"), (1, "s")):
            off = half - 2 * half * (2 * k + i) / max(2 * n - 1, 1)
            for j in range(len(TYPES)):
                r = res["type_change"][i][j]
                if np.isfinite(r[0]):
                    ax1.hlines(yt[j] + off, r[1], r[2], color=res["colour"], lw=2,
                               alpha=1.0 if i == 0 else 0.55)
                    ax1.plot(r[0], yt[j] + off, mk, color=res["colour"], ms=6, mec=SURFACE,
                             mew=1.2)
    ax1.axvline(0, color=MUTED, lw=1)
    ax1.set_yticks(yt)
    ax1.set_yticklabels(TYPES)
    ax1.set_xlabel("Change in the gap, RTN vs MulSupCon (AUC points, 95% interval;\n"
                   "circles: Control 1, squares: Control 2; below 0 = less confusion)")
    ax1.set_title("(b) Related-class confusion by relation type (skeleton's related pairs)",
                  loc="left", fontsize=10)
    fig.legend(handles=[Line2D([0], [0], color=r["colour"], lw=2.2, marker="o", mec=SURFACE,
                               label=r["label"]) for r in results],
               loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=n, frameon=False,
               fontsize=9.5)
    for a in (ax0, ax1):
        _style(a)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ========================================================= experiment ====
class Experiment:
    """The skeleton on one dataset and one encoder.

    dataset: dict(name, folder, definitions=dict(file, url=None)); the folder
             holds check_inputs/ (training_positives.npz, val_positives.npz,
             labels.csv with display_name and cohort columns)
    encoder: dict(key, label, features, old_runs=None, extract=None, colour=None);
             features holds train_features.npy and val_features.npy (same
             image order as the labels); old_runs is a folder of saved
             MulSupCon runs of earlier rounds (reused: MulSupCon uses no
             relation types); extract is the command that writes missing
             features.
    """

    def __init__(self, dataset, encoder, quick_test=False, workers="auto", retrain=False,
                 allow_cpu=False):
        import torch
        self.torch = torch
        self.ds, self.enc = dict(dataset), dict(encoder)
        self.quick, self.workers, self.retrain = quick_test, workers, retrain
        self.base = self.ds["folder"]
        self.out = os.path.join(self.base, "rtn_skeleton", self.enc["key"])
        self.rel_dir = os.path.join(self.base, "relations")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tables = {}
        for f in ("context_reliance_check.py", "relation_contrastive_method.py",
                  "relation_contrastive_runs.py"):
            if not os.path.exists(f):
                raise SystemExit(f"Put {f} (written by the round 6 notebook) in this folder.")
        import context_reliance_check as crc
        import relation_contrastive_method as rcm
        import relation_contrastive_runs as rcr
        if not hasattr(crc, "related_labels") or rcm.CODE_VERSION != HOST_VERSION \
                or not hasattr(rcr, "prepare_data"):
            raise SystemExit("The relation_contrastive files are not the round 6 versions.")
        self.crc, self.rcm, self.rcr = crc, rcm, rcr
        self.pt, self.ft = dict(rcm.PRETRAIN), dict(rcm.FINETUNE)
        self.seeds, self.bootstrap = list(SEEDS), BOOTSTRAP
        if quick_test:
            self.pt.update(epochs=3, iters_per_epoch=40)
            self.ft.update(official_epochs=3, max_epochs=6, iters_per_epoch=10)
            self.bootstrap = 500
        # The host settings' tag, computed as in rounds 6-7, so their saved
        # MulSupCon runs keep their names and can be reused
        self.cfg_tag = hashlib.md5(json.dumps(
            dict(pt=self.pt, ft=self.ft, split=SPLIT_SEED, dev=DEV_FRAC, pct=RELATED_PCT,
                 code=rcm.CODE_VERSION), sort_keys=True).encode()).hexdigest()[:8]
        if self.device == "cpu" and not (quick_test or allow_cpu):
            raise SystemExit("No GPU is visible to this notebook. Pick the kernel you used for "
                             "rounds 1-7, then Restart & Run All (or pass allow_cpu=True).")
        self._features()
        os.makedirs(self.out, exist_ok=True)
        self.rng = np.random.default_rng(0)
        print(f"{self.ds['name']}, {self.enc['label']}: skeleton version {SKELETON_VERSION}; "
              f"results in {os.path.abspath(self.out)}"
              + ("\nQUICK_TEST is on: tiny runs, results are meaningless." if quick_test else ""))

    # ----------------------------------------------------------------
    def _features(self):
        feats = [os.path.join(self.enc["features"], f)
                 for f in ("train_features.npy", "val_features.npy")]
        if all(os.path.exists(p) for p in feats):
            return
        cmd = self.enc.get("extract")
        if not cmd:
            raise SystemExit(f"No features for {self.enc['label']} in {self.enc['features']}.")
        args = shlex.split(cmd)
        if args and args[0] in ("python", "python3"):
            args[0] = sys.executable
        if len(args) < 2 or not os.path.exists(args[1]):
            raise SystemExit(f"No features for {self.enc['label']}: run {cmd}")
        print(f"Extracting the {self.enc['label']} features (once; a few minutes with a GPU):\n"
              f"  {cmd}", flush=True)
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
        if p.wait() != 0 or not all(os.path.exists(f) for f in feats):
            raise SystemExit("Feature extraction failed (see above).")

    # ----------------------------------------------------------------
    def setup(self):
        """Show the skeleton, and self-test the host code (MulSupCon)."""
        torch, rcm = self.torch, self.rcm
        print(f"numpy {np.__version__}, pandas {pd.__version__}, torch {torch.__version__}")
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            print(f"Device: {props.name} ({props.total_memory / 2**30:.1f} GB)")
        else:
            print(f"Device: CPU (this PyTorch build has CUDA: {torch.version.cuda})")
        _show(skeleton_table())
        torch.manual_seed(0)

        def labels(n, C, p):
            Yr = (torch.rand(n, C) < p).float()
            Yr[Yr.sum(1) == 0, 0] = 1
            return Yr

        results = []
        for _ in range(3):
            bl = labels(64, 80, 0.08)
            ref = torch.cat([bl, labels(256, 80, 0.05)])
            base = torch.rand(64, 320) * 2 - 1
            s1, s2 = base.clone().requires_grad_(), base.clone().requires_grad_()
            sc, mref = rcm.official_generate_output_MulSupCon(bl, ref, s1)
            l1 = rcm.official_weighted_supcon(sc, mref, 0.1)
            l2 = rcm.mulsupcon_loss_vec(s2, bl, ref, 0.1, "uniform")
            l1.backward()
            l2.backward()
            results.append(("MulSupCon loss and gradients equal the official code",
                            abs(l1.item() - l2.item()) < 1e-5
                            and torch.allclose(s1.grad, s2.grad, atol=1e-6)))
            l3 = rcm.mulsupcon_loss_vec(base, bl, ref, 0.1, "relation",
                                        relation=[(1.0, torch.ones(80, 80))], alpha=0.0)
            results.append(("with alpha 0, RTN is exactly MulSupCon",
                            abs(l3.item() - l2.item()) < 1e-5))
            rel = [(1.0, (torch.rand(80, 80) < 0.05).float()),
                   (0.5, (torch.rand(80, 80) < 0.05).float())]
            a = rcm.mulsupcon_loss(base, bl, ref, 0.1, "relation", relation=rel, alpha=3.0).item()
            b = rcm.mulsupcon_loss_vec(base, bl, ref, 0.1, "relation", relation=rel,
                                       alpha=3.0).item()
            results.append(("RTN: fast loss equals the row-by-row definition", abs(a - b) < 1e-4))
        m = rcm.PretrainModel(512, 80, 256, 0.999, 256, 1)
        m.train()
        widths = [m(torch.randn(64, 512), torch.randn(64, 512), labels(64, 80, 0.05))[0].shape[1]
                  for _ in range(5)]
        results.append(("queue: batch keys only until full, then batch + queue",
                        widths == [64, 64, 64, 320, 320]))
        check = pd.DataFrame(results, columns=["check", "passed"]) \
            .groupby("check", sort=False)["passed"].all().reset_index()
        self.tables["host_checks"] = check
        _show(check)
        if not check["passed"].all():
            raise SystemExit("A self-test of the host code failed; stop here.")

    # ----------------------------------------------------------------
    def relations(self):
        """Component 1 and 2: GloVe + WordNet class similarity, related classes."""
        labels_df = pd.read_csv(os.path.join(self.base, "check_inputs", "labels.csv"))
        self.names = labels_df["display_name"].tolist()
        self.cohort = labels_df["cohort"].to_numpy()
        self.L = len(self.names)
        os.makedirs(self.rel_dir, exist_ok=True)
        d = self.ds["definitions"]
        if not os.path.exists(d["file"]):
            if not d.get("url"):
                raise SystemExit(f"No WordNet definitions file {d['file']}.")
            _download(d["url"], d["file"])
        with open(d["file"], encoding="utf-8") as f:
            WN = json.load(f)
        lost = [n for n in self.names if n not in WN]
        if lost:
            raise SystemExit(f"No WordNet definition for: {', '.join(lost)}")
        self.definitions = [WN[n]["meaning"] if isinstance(WN[n], dict) else str(WN[n])
                            for n in self.names]
        key = _md5(dict(names=self.names, definitions=self.definitions,
                        glove=os.path.basename(GLOVE_URL), stopwords=sorted(STOPWORDS),
                        method="name + definition, unit parts, sum"), n=10)
        emb_path = os.path.join(self.rel_dir, f"glove_wordnet_{key}.npy")
        miss_path = emb_path.replace(".npy", "_missing.json")
        if os.path.exists(emb_path) and os.path.exists(miss_path):
            self.E = np.load(emb_path)
            with open(miss_path, encoding="utf-8") as f:
                missing = json.load(f)
        else:
            glove_path = os.path.join(self.rel_dir, "glove-wiki-gigaword-300.gz")
            _download(GLOVE_URL, glove_path)
            self.E, missing = glove_text_embeddings(self.names, glove_path, self.definitions)
            np.save(emb_path, self.E)
            with open(miss_path, "w", encoding="utf-8") as f:
                json.dump(missing, f)
        self.sim = cosine_similarity(self.E)
        self.related = related_mask(self.sim)
        print("GloVe words not found (ignored):", missing or "none")
        show = [k for k in range(self.L) if self.cohort[k] == "tail"] + \
               [self.names.index(n) for n in AMBIGUOUS
                if n in self.names and self.cohort[self.names.index(n)] != "tail"]
        table = pd.DataFrame([{"class": self.names[k], "cohort": self.cohort[k],
                               "WordNet meaning": self.definitions[k],
                               "related classes": ", ".join(self.names[c] for c in
                                                            np.flatnonzero(self.related[k]))}
                              for k in show])
        self.tables["related_lists"] = table
        print("Related classes (top 5% most similar names) of the tail classes and of names "
              "with several meanings:")
        _show(table)

    # ----------------------------------------------------------------
    def prepare(self):
        """Data for the host code, and checks that it runs exactly this skeleton."""
        if not hasattr(self, "E"):
            self.relations()
        rcm, rcr = self.rcm, self.rcr
        data = os.path.join(self.out, "data")
        for sub in ("runs", "backbones", "logs", os.path.join("data", "check_inputs"),
                    os.path.join("data", "features")):
            os.makedirs(os.path.join(self.out, sub), exist_ok=True)
        for f in ("training_positives.npz", "val_positives.npz", "labels.csv"):
            _place(os.path.join(self.base, "check_inputs", f), os.path.join(data, "check_inputs", f))
        for f in ("train_features.npy", "val_features.npy"):
            _place(os.path.join(self.enc["features"], f), os.path.join(data, "features", f))
        # The host code reads class-name embeddings from this file name; here
        # it holds the GloVe + WordNet ones.
        np.save(os.path.join(data, "features", "label_text_embeddings.npy"), self.E)
        try:
            self.D = rcr.prepare_data(data, DEV_FRAC, SPLIT_SEED, RELATED_PCT, self.device)
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        D = self.D
        self.job = dict(pt=self.pt, ft=self.ft, device=self.device, data=os.path.abspath(data),
                        dev_frac=DEV_FRAC, split_seed=SPLIT_SEED, related_pct=RELATED_PCT,
                        heartbeat=os.path.abspath(os.path.join(self.out, "runs", "heartbeat")))
        self.R_fit, masks = relation_types(D["Y_fit"], self.sim)
        checks = [
            ("the host code's class similarity is the skeleton's (GloVe + WordNet)",
             bool(np.allclose(D["sim"], self.sim))),
            ("the host code's relation types equal the skeleton's",
             all(np.array_equal(D[a], masks[b]) for a, b in
                 (("rel_seen", "related_seen"), ("rel_never", "related_never"),
                  ("informative", "informative")))),
            ("the host code's negative weights are w = 1 + R",
             bool(np.allclose(rcm.neg_weight_matrix(D["Y_fit"], D["RELATION"], ALPHA),
                              negative_weights(self.R_fit, D["Y_fit"]).T))),
            ("the labels match the class list",
             D["Y_all"].shape[1] == D["Y_test"].shape[1] == self.L),
        ]
        checks = pd.DataFrame(checks, columns=["check", "passed"])
        self.tables["skeleton_checks"] = checks
        _show(checks)
        if not checks["passed"].all():
            raise SystemExit("A check failed: the host code would not run the skeleton. Stop "
                             "here and send this output.")
        self.Y_all, self.Y_test = D["Y_all"], D["Y_test"]
        self.rtag = _md5(self.R_fit, dict(alpha=ALPHA, version=SKELETON_VERSION), n=8)
        self.hashes = dict(split=_md5(np.asarray(D["fit"]), np.asarray(D["dev"])),
                           labels=_md5(self.Y_all, self.Y_test),
                           relations=_md5(self.R_fit),
                           embeddings=_md5(self.E))
        mods = (self.crc, self.rcm, self.rcr)
        parts = dict(version=SKELETON_VERSION, skeleton_file=_md5_file(__file__),
                     host_files={os.path.basename(m.__file__): _md5_file(m.__file__) for m in mods},
                     pretrain=self.pt, finetune=self.ft, seeds=self.seeds, dev=DEV_FRAC,
                     split_seed=SPLIT_SEED, bootstrap=self.bootstrap, quick_test=self.quick,
                     embeddings=self.hashes["embeddings"])
        self.fingerprint = _md5(parts)
        print(f"Training images: {len(D['fit'])} for training, {len(D['dev'])} for the dev set; "
              f"test images: {len(self.Y_test)}. Skeleton fingerprint: {self.fingerprint}. "
              "Relation pairs (training part):")
        counts = pd.DataFrame([dict(cohort=g, classes=int((self.cohort == g).sum()),
                                    related_seen=int(masks["related_seen"][self.cohort == g].sum()),
                                    related_never=int(masks["related_never"][self.cohort == g].sum()),
                                    informative=int(masks["informative"][self.cohort == g].sum()))
                               for g in ("head", "medium", "tail")])
        self.tables["relation_counts"] = counts
        _show(counts)

    # ----------------------------------------------------------------
    def _spec(self, key, kind, setting, seed):
        tag = (f"mulsupcon_default_{seed}_{self.cfg_tag}" if key == "mulsupcon" else
               f"rtn_s{SKELETON_VERSION}-{self.rtag}_{seed}_{self.cfg_tag}")
        return dict(key=key, kind=kind, setting=setting, seed=seed, tag=tag,
                    path=os.path.abspath(os.path.join(self.out, "runs", tag + ".npz")),
                    backbone_path=os.path.abspath(os.path.join(self.out, "backbones",
                                                               tag + ".pt")))

    def _matches(self, r):
        """A saved run belongs to this split and these images: right shapes, and
        its dev mAP recomputed from its dev scores with this split's dev labels
        equals the dev mAP saved with it."""
        D = self.D
        if r["test_logits"].shape != (len(self.Y_test), self.L) \
                or r["dev_logits"].shape != (len(D["dev"]), self.L):
            return False
        return abs(self.rcr.dev_map(r["dev_logits"], D["Y_dev"]) - float(r["dev_map"])) <= 1e-6

    def train(self):
        """MulSupCon and RTN, 3 seeds each (resumable)."""
        if not hasattr(self, "D"):
            self.prepare()
        torch, rcm, rcr, D = self.torch, self.rcm, self.rcr, self.D
        specs = [self._spec(key, kind, setting, seed) for key, _, kind, setting in VARIANTS
                 for seed in self.seeds]
        log_path = os.path.join(self.out, "runs", "reused.json")   # runs copied from earlier rounds
        log = {}
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                log = json.load(f)
        if self.retrain:
            for s in specs:
                if os.path.exists(s["path"]):
                    os.remove(s["path"])
                log.pop(os.path.basename(s["path"]), None)
        copied = []
        old = self.enc.get("old_runs")
        for s in specs:
            name = os.path.basename(s["path"])
            src = os.path.join(old, name) if old else None
            if s["key"] == "mulsupcon" and not self.retrain and not os.path.exists(s["path"]) \
                    and src and os.path.exists(src):
                shutil.copy2(src, s["path"])
                if self._matches(rcr.load_run(s["path"])):
                    copied.append(s["seed"])
                    log[name] = os.path.abspath(src)
                else:
                    os.remove(s["path"])
                    print(f"The saved MulSupCon run of seed {s['seed']} in {old} does not match "
                          "this split or these images, so it will be trained instead.")
        log = {k: v for k, v in log.items() if os.path.exists(os.path.join(self.out, "runs", k))}
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        self.reused = [s["seed"] for s in specs if os.path.basename(s["path"]) in log]
        if copied:
            print(f"Reused the saved MulSupCon runs of seeds {', '.join(map(str, copied))} from "
                  f"{old} (MulSupCon uses no relation types).")
        todo = [s for s in specs if not os.path.exists(s["path"])]
        print(f"{len(specs) - len(todo)} of {len(specs)} runs are saved; {len(todo)} to do.")

        def sync():
            if self.device == "cuda":
                torch.cuda.synchronize()

        speed, ft_epoch = {}, 0.0
        if todo:
            probe_setting = {"uniform": {}, "relation": {"alpha": ALPHA}}
            for kind in sorted({s["kind"] for s in todo} | {"uniform"}):
                kw = rcr.pretrain_kwargs(kind, probe_setting[kind], D)
                cfg = dict(self.pt, epochs=1, iters_per_epoch=150)
                rcm.pretrain(D["Xf"], D["Yf"], seed=0, device=self.device,
                             cfg=dict(cfg, iters_per_epoch=20), **kw)
                sync()
                t0 = time.time()
                rcm.pretrain(D["Xf"], D["Yf"], seed=0, device=self.device, cfg=cfg, **kw)
                sync()
                speed[kind] = (time.time() - t0) / 150
            probe = dict(self.ft, official_epochs=4, max_epochs=4)
            rcm.finetune(D["Xf"], D["Yf"], D["Xd"], D["Yd"], D["Xt"], 0, self.device,
                         dict(probe, official_epochs=1, max_epochs=1))
            sync()
            t0 = time.time()
            rcm.finetune(D["Xf"], D["Yf"], D["Xd"], D["Yd"], D["Xt"], 0, self.device, probe)
            sync()
            ft_epoch = (time.time() - t0) / 4
        iters = self.pt["epochs"] * self.pt["iters_per_epoch"]

        def minutes(s, slow=1.0):
            return slow * (speed.get(s["kind"], 0.0) * iters + ft_epoch * self.ft["max_epochs"]) / 60

        workers = 1
        if todo:
            serial = sum(minutes(s) for s in todo)
            estimate, workers = serial, self.workers
            if workers == "auto":
                workers, info = rcr.auto_workers(self.device)
                print(info)
            workers = int(workers)
            if workers > 1:
                print(f"Testing {workers} trainings at the same time (about a minute)...")
                t_par = rcr.benchmark_parallel(workers, self.job, self.out)
                if t_par is None:
                    print("The parallel test failed, so trainings will run one at a time.")
                    workers = 1
                else:
                    par = rcr.makespan([minutes(s, t_par / speed["uniform"]) for s in todo],
                                       workers)
                    if par < 0.9 * serial:
                        estimate = par
                        print(f"Running {workers} at once is faster.")
                    else:
                        workers = 1
                        print("Running several at once is not faster here, so one at a time.")
            print(f"Estimated time: up to about {estimate:.0f} min ({estimate / 60:.1f} h), "
                  f"{workers} at a time. Fine-tuning usually stops early, so it is often shorter.")

        def describe(s, r):
            return (f"{SHORT[s['key']]}, seed {s['seed']}: dev mAP {float(r['dev_map']):.2f}; "
                    f"fine-tuning best epoch {int(r['best_epoch'])} of {int(r['epochs_run'])} "
                    f"({float(r['seconds']) / 60:.1f} min)")

        t0 = time.time()
        rcr.execute(specs, workers, self.job, D, describe, {s["tag"]: minutes(s) for s in specs})
        self.runs = {key: {} for key in SHORT}
        rows = []
        for s in specs:
            r = rcr.load_run(s["path"])
            if not self._matches(r):
                raise SystemExit(f"{s['path']} does not match this split or these images, so it "
                                 "comes from different data. Delete it and run again.")
            self.runs[s["key"]][s["seed"]] = r
            rows.append({"variant": SHORT[s["key"]], "seed": s["seed"],
                         "dev mAP": round(float(r["dev_map"]), 2),
                         "best epoch": int(r["best_epoch"]),
                         "converged": bool(r["stopped_early"])
                         or float(r["final_lr"]) <= self.ft["lr"] * 1e-4,
                         "from an earlier round": s["key"] == "mulsupcon" and s["seed"] in self.reused})
        table = pd.DataFrame(rows)
        self.tables["runs"] = table
        print(f"All {len(specs)} runs are saved and match this split "
              f"({(time.time() - t0) / 60:.0f} min in this session):")
        _show(table)

    # ----------------------------------------------------------------
    def _class_ap(self, logits):
        Y = self.Y_test
        return np.array([self.crc.average_precision(logits[:, j], Y[:, j]) if Y[:, j].any()
                         else np.nan for j in range(self.L)])

    def results(self):
        """Test mAP per class group, mean +/- sd over the seeds."""
        self.groups = {g: np.flatnonzero(self.cohort == g) for g in ("head", "medium", "tail")}
        self.groups["all"] = np.arange(self.L)
        self.ap_seed = {key: [self._class_ap(self.runs[key][s]["test_logits"]) for s in self.seeds]
                        for key in SHORT}
        self.ap = {key: 100 * np.nanmean(v, axis=0) for key, v in self.ap_seed.items()}

        def gm(key, g):
            return [100 * float(np.nanmean(a[self.groups[g]])) for a in self.ap_seed[key]]

        self.group_maps = {key: {g: gm(key, g) for g in GROUP_NAMES} for key in SHORT}
        rows = []
        for key, name, _, _ in VARIANTS:
            row = {"encoder": self.enc["label"], "variant": name}
            for g in ("all", "head", "medium", "tail"):
                v = self.group_maps[key][g]
                row[f"{g} mAP"] = f"{np.mean(v):.2f} +/- {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}"
            rows.append(row)
        table = pd.DataFrame(rows)
        self.tables["results"] = table
        _show(table)

    # ----------------------------------------------------------------
    def confusion(self):
        """Related-class gap per relation type, on the skeleton's own related pairs."""
        crc, L, TAIL = self.crc, self.L, self.groups["tail"]
        shared = self.Y_all.T.astype(np.int64) @ self.Y_all.astype(np.int64)
        n_all = np.diag(shared).copy()
        base = n_all / len(self.Y_all)
        G = np.full((L, L), -1)               # 0-3 = TYPES, 4 = baseline, -1 = not used
        for t in range(L):
            for c in range(L):
                if c == t:
                    continue
                k = shared[t, c]
                lift = (k / n_all[t]) / base[c] if n_all[t] and base[c] else 0.0
                if self.related[t, c]:
                    G[t, c] = 0 if k >= 1 else 1
                elif k == 0:
                    G[t, c] = 4
                elif k >= MIN_SHARED:
                    G[t, c] = 2 if lift >= LIFT_MIN else 3
        flag = (G >= 0) & (G <= 2)
        Yb = self.Y_test > 0.5
        objects = np.minimum(Yb.sum(1), COUNT_CAP)
        levels = np.unique(objects)
        prep = {}
        for t in TAIL:
            pos = Yb[:, t]
            P = Yb[~pos]
            Fp = P & flag[t][None, :]
            allowed = (Fp.sum(1)[:, None] - Fp) == 0
            H = (objects[~pos][:, None] == levels[None, :]).astype(float)
            prep[t] = (pos, P & allowed, ~P & allowed, H, (P & allowed).sum(0) >= MIN_NEG)

        def gaps_of(score_list):
            tg = np.full((2, len(TYPES), L), np.nan)
            rg = np.full((2, L), np.nan)
            acc = {t: [np.zeros(L), np.zeros(L)] for t in TAIL}
            for S in score_list:
                S = np.asarray(S, dtype=np.float64)
                Sc = S - np.median(S, axis=1, keepdims=True)
                for t in TAIL:
                    pos, IN_, OUT_, H, _ = prep[t]
                    IN_, OUT_ = IN_.astype(np.float64), OUT_.astype(np.float64)
                    acc[t][0] += _pair_gaps(crc.beat_rates(S[~pos, t], S[pos, t]), IN_, OUT_, H)[0]
                    acc[t][1] += _pair_gaps(crc.beat_rates(Sc[~pos, t], Sc[pos, t]), IN_, OUT_,
                                            H)[1]
            for t in TAIL:
                used = prep[t][4]
                c = [a / len(score_list) for a in acc[t]]
                basel = used & (G[t] == 4)
                if not basel.any():
                    continue
                rel = used & ((G[t] == 0) | (G[t] == 1))
                for i in (0, 1):
                    b0 = np.nanmean(c[i][basel])
                    if rel.any():
                        rg[i, t] = np.nanmean(c[i][rel]) - b0
                    for j in range(len(TYPES)):
                        sel = used & (G[t] == j)
                        if sel.any():
                            tg[i, j, t] = np.nanmean(c[i][sel]) - b0
            return tg, rg

        t0 = time.time()
        self.tg, self.rg = {}, {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for key in SHORT:
                self.tg[key], self.rg[key] = gaps_of([self.runs[key][s]["test_logits"]
                                                      for s in self.seeds])
            levels_tab = pd.DataFrame(
                [{"variant": SHORT[key],
                  **{f"{tn}, C{i + 1}": round(float(np.nanmean(self.tg[key][i, j, TAIL])), 2)
                     for j, tn in enumerate(TYPES) for i in range(2)}} for key in SHORT])
        print(f"Computed in {time.time() - t0:.0f} s. Average gap per relation type (AUC points, "
              "minus the baseline group; lower is better):")
        _show(levels_tab)
        self.type_change = [[_boot(self.rng, self.tg["rtn"][i, j, TAIL]
                                   - self.tg["mulsupcon"][i, j, TAIL], self.bootstrap)
                             for j in range(len(TYPES))] for i in (0, 1)]
        changes = pd.DataFrame([{"type": tn, **{f"Control {i + 1}": _ci(self.type_change[i][j])
                                               for i in (0, 1)}}
                                for j, tn in enumerate(TYPES)])
        self.tables["gap_levels"], self.tables["gap_changes"] = levels_tab, changes
        print("Change in the gap, RTN vs MulSupCon (95% interval; below 0 = less confusion):")
        _show(changes)

    # ----------------------------------------------------------------
    def verdict(self):
        """RTN against MulSupCon, judged by the decision rule."""
        TAIL = self.groups["tail"]
        a, b = "rtn", "mulsupcon"
        self.ap_change = {g: _boot(self.rng, (self.ap[a] - self.ap[b])[self.groups[g]],
                                   self.bootstrap) for g in GROUP_NAMES}
        self.gap_change = [_boot(self.rng, self.rg[a][i, TAIL] - self.rg[b][i, TAIL],
                                 self.bootstrap) for i in (0, 1)]
        self.n_better = int(sum(np.nanmean(x[TAIL]) > np.nanmean(y[TAIL])
                                for x, y in zip(self.ap_seed[a], self.ap_seed[b])))
        tail_ok = self.ap_change["tail"][1] > 0 and self.n_better >= 2
        gap_ok = all(r[2] < 0 for r in self.gap_change)
        head_ok = self.ap_change["head"][0] >= -1
        self.verdict_text = ("WORKS" if tail_ok and gap_ok and head_ok else
                             "PARTLY" if (tail_ok or gap_ok) and head_ok else "NOT SHOWN")
        gm = self.group_maps
        row = {"dataset": self.ds["name"], "encoder": self.enc["label"],
               "MulSupCon all / tail mAP": f"{np.mean(gm[b]['all']):.2f} / {np.mean(gm[b]['tail']):.2f}",
               "RTN all / tail mAP": f"{np.mean(gm[a]['all']):.2f} / {np.mean(gm[a]['tail']):.2f}",
               "tail AP change": _ci(self.ap_change["tail"]),
               "higher tail mAP (seeds)": f"{self.n_better}/{len(self.seeds)}",
               "head AP change": _ci(self.ap_change["head"]),
               "all AP change": _ci(self.ap_change["all"]),
               "gap change, C1": _ci(self.gap_change[0]), "gap change, C2": _ci(self.gap_change[1]),
               "verdict": self.verdict_text}
        self.line = (f"{self.ds['name']}, {self.enc['label']}: RTN vs MulSupCon "
                     f"{self.verdict_text} (tail AP {_ci(self.ap_change['tail'])}, higher in "
                     f"{self.n_better} of {len(self.seeds)} seeds; related-class gap C1 "
                     f"{_ci(self.gap_change[0])}, C2 {_ci(self.gap_change[1])}; head AP "
                     f"{self.ap_change['head'][0]:+.2f}; overall AP {_ci(self.ap_change['all'])})")
        table = pd.DataFrame([row])
        self.tables["verdict"] = table
        print(self.line)
        _show(table)

    # ----------------------------------------------------------------
    def _result(self):
        return dict(dataset=self.ds["name"], encoder=self.enc["key"], label=self.enc["label"],
                    colour=self.enc.get("colour", COLOURS[0]), skeleton_version=SKELETON_VERSION,
                    fingerprint=self.fingerprint, hashes=self.hashes, quick_test=self.quick,
                    seeds=self.seeds, verdict=self.verdict_text, n_better=self.n_better,
                    ap_change={g: list(self.ap_change[g][:3]) for g in GROUP_NAMES},
                    gap_change=[list(r[:3]) for r in self.gap_change],
                    type_change=[[list(r[:3]) for r in row] for row in self.type_change],
                    group_maps=self.group_maps, reused_mulsupcon_seeds=self.reused,
                    row=self.tables["verdict"].iloc[0].to_dict(), line=self.line)

    def figure(self):
        """RTN against MulSupCon on this encoder."""
        path = os.path.join(self.out, "figure.png")
        _draw([self._result()], path, f"{self.ds['name']}, {self.enc['label']}: the RTN skeleton "
                                      f"(version {SKELETON_VERSION}) against MulSupCon")
        _show_image(path)
        print(f"Saved {path}")

    def save(self):
        """Tables, per-class AP, the result file the summary reads, and a text summary."""
        for name, table in self.tables.items():
            table.to_csv(os.path.join(self.out, f"{name}.csv"), index=False)
        pd.DataFrame({SHORT[k]: self.ap[k] for k in SHORT}, index=self.names) \
            .to_csv(os.path.join(self.out, "per_class_ap.csv"))
        with open(os.path.join(self.out, "result.json"), "w", encoding="utf-8") as f:
            json.dump(self._result(), f, indent=2, default=str)
        with open(os.path.join(self.out, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"{self.line}\n\nSkeleton version {SKELETON_VERSION}, fingerprint "
                    f"{self.fingerprint}\n\n" + skeleton_table().to_string(index=False) + "\n\n"
                    + self.tables["results"].to_string(index=False) + "\n\nBy relation type:\n"
                    + self.tables["gap_changes"].to_string(index=False) + "\n")
        print(f"Saved the tables, result.json and summary.txt to {os.path.abspath(self.out)}")

    def run_all(self):
        for step in (self.setup, self.relations, self.prepare, self.train, self.results,
                     self.confusion, self.verdict, self.figure, self.save):
            step()


# ============================================================ summary ====
def summarize(folder, encoders, name=None):
    """One table and one figure for the encoders of one dataset. Refuses to
    compare results that do not come from the same skeleton, split, labels
    and relation types."""
    res, missing = [], []
    for k in encoders:
        p = os.path.join(folder, "rtn_skeleton", k, "result.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                res.append(json.load(f))
        else:
            missing.append(k)
    if missing:
        print(f"No results yet for: {', '.join(missing)} (run their notebooks first).")
    if not res:
        raise SystemExit("Nothing to summarize yet.")
    checks = [(f"same skeleton (fingerprint {res[0]['fingerprint']})",
               len({r["fingerprint"] for r in res}) == 1)]
    for h in ("split", "labels", "relations"):
        checks.append((f"same {h}", len({r["hashes"][h] for r in res}) == 1))
    checks = pd.DataFrame(checks, columns=["check", "passed"])
    _show(checks)
    if not checks["passed"].all():
        _show(pd.DataFrame([dict(encoder=r["label"], fingerprint=r["fingerprint"],
                                 skeleton_version=r["skeleton_version"], **r["hashes"])
                            for r in res]))
        raise SystemExit("These results come from different skeletons or data: rerun the "
                         "encoder notebooks with the current rtn_skeleton.py first.")
    for i, r in enumerate(res):
        r["colour"] = COLOURS[i % len(COLOURS)]
    out = os.path.join(folder, "rtn_skeleton", "summary")
    os.makedirs(out, exist_ok=True)
    table = pd.DataFrame([r["row"] for r in res])
    for r in res:
        print(r["line"])
    _show(table)
    dataset = name or res[0]["dataset"]
    path = os.path.join(out, "figure.png")
    _draw(res, path, f"{dataset}: the RTN skeleton (version {res[0]['skeleton_version']}) "
                     f"against MulSupCon on {len(res)} encoder{'s' if len(res) > 1 else ''}")
    _show_image(path)
    table.to_csv(os.path.join(out, "skeleton_all_encoders.csv"), index=False)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"{dataset}: the RTN skeleton, version {res[0]['skeleton_version']}, fingerprint "
                f"{res[0]['fingerprint']}\n\n" + "\n".join(r["line"] for r in res) + "\n\n"
                + table.to_string(index=False) + "\n")
    print(f"Saved the table, figure and summary.txt to {os.path.abspath(out)}")
