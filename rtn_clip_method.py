"""Round 3: relation-typed negatives (RTN) on a CLIP-initialised classifier.

Everything works on frozen CLIP ViT-B/16 features (L2-normalised image
embeddings) and the CLIP text embeddings of the class names, so one training
run takes seconds.

Model ("TA"): a residual adapter that starts as the identity, followed by a
cosine classifier whose class vectors start at the class-name text
embeddings. At initialisation it ranks images exactly as zero-shot CLIP does.
Loss: classification (BCE or ASL) + lambda x label-wise supervised
contrastive loss (MulSupCon's anchors: one per image and label) computed
against every training image, with relation-typed weights on the negatives.

Round 4 adds two classifier-level forms of RTN: a class contrast (each
positive label of an image against the other classes' vectors, related
classes weighted more) and relation-typed margins in the classification
loss. Settings without them behave exactly as in round 3.

Round 5 adds (i) a class-anchored contrast: each class vector is an anchor,
its training images are the positives and every other training image is a
negative, with relation-typed weights on the negatives, so the loss ranks
images within each class (what AP measures); (ii) relation-typed re-scoring
at test time; (iii) a headroom (oracle) analysis. Settings without them
behave exactly as in rounds 3 and 4.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------- data utils ----
def iterative_split(Y, test_frac, seed):
    """Iterative stratification (Sechidis et al., 2011), two parts.
    Identical to round 1's function: 0 = fit, 1 = dev."""
    return iterative_folds(Y, None, seed, fracs=[1.0 - test_frac, test_frac])


def iterative_folds(Y, k, seed, fracs=None):
    """Iterative stratification into k folds of equal size (or the given
    fractions). Returns the fold index of every row."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=bool)
    n = Y.shape[0]
    frac = np.asarray(fracs if fracs is not None else [1.0 / k] * k, dtype=float)
    desired = frac * n
    desired_label = np.outer(frac, Y.sum(0)).astype(float)
    remaining = np.ones(n, dtype=bool)
    assign = -np.ones(n, dtype=int)
    while True:
        rem = Y[remaining].sum(0)
        if rem.sum() == 0:
            break
        label = int(np.argmin(np.where(rem > 0, rem, np.iinfo(np.int64).max)))
        idx = np.flatnonzero(remaining & Y[:, label])
        rng.shuffle(idx)
        for i in idx:
            d = desired_label[:, label]
            cand = np.flatnonzero(d == d.max())
            if cand.size > 1:
                ds = desired[cand]
                cand = cand[ds == ds.max()]
            s = int(rng.choice(cand)) if cand.size > 1 else int(cand[0])
            assign[i] = s
            remaining[i] = False
            desired_label[s] -= Y[i]
            desired[s] -= 1
    for i in np.flatnonzero(remaining):          # images without labels
        s = int(np.argmax(desired))
        assign[i] = s
        desired[s] -= 1
    return assign


def relation_masks(sim, shared, n_train, n_images, related_fn, pct,
                   min_shared=2, lift_min=2.0):
    """Relation types from training labels only (same as rounds 1 and 2):
    related & seen together, related & never together, informative companion,
    and 'any companion' (seen together at least once)."""
    L = sim.shape[0]
    rel_seen = np.zeros((L, L), bool)
    rel_never = np.zeros((L, L), bool)
    informative = np.zeros((L, L), bool)
    companion = np.zeros((L, L), bool)
    base = n_train / n_images
    for t in range(L):
        related = related_fn(sim[t], t, pct)
        for c in range(L):
            if c == t:
                continue
            k = shared[t, c]
            if k >= 1:
                companion[t, c] = True
            if related[c]:
                (rel_seen if k >= 1 else rel_never)[t, c] = True
            elif k >= min_shared and n_train[t] and base[c]:
                if (k / n_train[t]) / base[c] >= lift_min:
                    informative[t, c] = True
    return rel_seen, rel_never, informative, companion


def relation_values(Y, sim, related_fn, pct):
    """K x K relation values R[k, c] from the given training labels:
    1 = related & seen together, 0.5 = related & never together,
    0.25 = informative companion, 0 = anything else."""
    Yi = np.asarray(Y, dtype=np.int64)
    shared = Yi.T @ Yi
    masks = relation_masks(sim, shared, np.diag(shared).copy(), len(Yi), related_fn, pct)
    rel_seen, rel_never, informative, _ = masks
    R = np.zeros(shared.shape, dtype=np.float32)
    R[informative] = 0.25
    R[rel_never] = 0.5
    R[rel_seen] = 1.0
    return R, masks


def relation_scores(R, Y):
    """N x K: for image j and class k, the largest R[k, c] over j's labels c
    (how confusable image j is for class k)."""
    Yf = np.asarray(Y, dtype=np.float32)
    score = np.zeros((Yf.shape[0], R.shape[0]), dtype=np.float32)
    for v in np.unique(R[R > 0]):
        has = (Yf @ (R == v).T.astype(np.float32)) > 0.5
        score = np.maximum(score, v * has)
    return score


def candidate_weights(R, Y, alpha, target):
    """K x N weight of every training image j as a negative for anchor label
    k: 1 + alpha x max over j's labels c of R[k, c]. Positives (j has k) keep
    weight 1, and so do anchor labels outside `target` (a K-length bool)."""
    Yf = np.asarray(Y, dtype=np.float32)
    score = relation_scores(R, Yf)                                       # N x K
    W = 1.0 + alpha * score.T                                            # K x N
    W[Yf.T > 0.5] = 1.0
    W[~np.asarray(target, dtype=bool), :] = 1.0
    return W.astype(np.float32)


def row_weights(Y, q):
    """Weight of anchor rows with label k, proportional to n_k^-q and scaled
    so that the average over all anchor rows is 1 (q = 0: all equal)."""
    n = np.asarray(Y, dtype=np.float64).sum(0)
    w = np.where(n > 0, np.maximum(n, 1.0) ** (-q), 0.0)
    return (w / ((n * w).sum() / n.sum())).astype(np.float32)


def standardise_by(train_scores, scores):
    """Per-class standardisation with statistics of the training images."""
    mu = train_scores.mean(0, keepdims=True)
    sd = train_scores.std(0, keepdims=True) + 1e-6
    return (scores - mu) / sd


# ----------------------------------------------------------------- losses ----
def bce_mean(logits, y):
    """BCE averaged over all entries (the linear-probe reference)."""
    return F.binary_cross_entropy_with_logits(logits, y)


def bce_sum(logits, y):
    """BCE summed over classes, averaged over images (same scale as ASL)."""
    return F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(1).mean()


def asl_sum(logits, y, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
    """Asymmetric loss (Ridnik et al., 2021) with the usual settings, as in
    the context-check notebook."""
    p = torch.sigmoid(logits)
    p_neg = (1 - p + clip).clamp(max=1)
    log_terms = (y * torch.log(p.clamp(min=eps))
                 + (1 - y) * torch.log(p_neg.clamp(min=eps)))
    pt = p * y + p_neg * (1 - y)
    focus = torch.pow(1 - pt, gamma_pos * y + gamma_neg * (1 - y))
    return -(log_terms * focus).sum(1).mean()


def label_supcon(z_all, y_all, idx, W, row_w, tau):
    """Label-wise supervised contrastive loss with weighted negatives.

    Anchors: every (image i in the batch `idx`, label k of i). Candidates:
    every training image except i. Positives: candidates with label k;
    negatives: the others, weighted by W[k, j] (W = 1 gives MulSupCon's
    uniform loss). Row loss: log(sum_pos e^s + sum_neg W e^s) - mean_pos s,
    with s = cosine / tau; rows are averaged with weights row_w[k].
    z_all: N x d (unit vectors), y_all: N x K (0/1 float), W: K x N."""
    z_b = z_all[idx]
    S = z_b @ z_all.T / tau                                   # B x N
    self_mask = F.one_hot(idx, S.shape[1]).bool()
    m = S.max(dim=1, keepdim=True).values.detach()
    E = torch.exp(S - m).masked_fill(self_mask, 0.0)
    den = m + torch.log((E @ W.T).clamp_min(1e-30))            # B x K
    y_b = y_all[idx]
    s_self = S.gather(1, idx.view(-1, 1))
    pos_sum = S @ y_all - s_self * y_b                        # self excluded
    n_pos = y_all.sum(0, keepdim=True) - y_b
    rows = (y_b > 0.5) & (n_pos > 0.5)
    ell = den - pos_sum / n_pos.clamp_min(1.0)
    w = row_w.view(1, -1).expand_as(ell)
    return (ell * w)[rows].sum() / w[rows].sum().clamp_min(1e-12)


def label_supcon_reference(z_all, y_all, idx, W, row_w, tau):
    """Row-by-row definition of label_supcon, for the self-test."""
    total, wsum = 0.0, 0.0
    N, K = y_all.shape
    for i in idx.tolist():
        s = z_all[i] @ z_all.T / tau
        others = [j for j in range(N) if j != i]
        for k in range(K):
            if y_all[i, k] < 0.5:
                continue
            pos = [j for j in others if y_all[j, k] > 0.5]
            if not pos:
                continue
            den = torch.log(sum(W[k, j] * torch.exp(s[j]) for j in others))
            ell = -sum(s[p] - den for p in pos) / len(pos)
            total = total + row_w[k] * ell
            wsum = wsum + row_w[k]
    return total / wsum


def class_contrast_weights(R, alpha, target):
    """K x K weights for the class contrast: Om[k, c] = 1 + alpha x R[c, k]
    for a negative class c when the image has label k (images of k are
    confusable for c), applied only to classes c in `target`."""
    tgt = np.asarray(target, dtype=np.float32)
    return (1.0 + alpha * R.T * tgt[None, :]).astype(np.float32)


def relation_class_contrast(cos_logits, y, Om, row_w):
    """For every positive label k of an image:
    -log(e^{l_k} / (e^{l_k} + sum over negative classes c of Om[k, c] e^{l_c})),
    with l = scale x cosine to the class vectors (no bias). The image's other
    positive labels are left out. Om = 1 gives the uniform class contrast."""
    m = cos_logits.max(dim=1, keepdim=True).values.detach()
    E = torch.exp(cos_logits - m)
    neg = (E * (1 - y)) @ Om.T                                  # B x K
    ell = m + torch.log(E + neg) - cos_logits
    rows = y > 0.5
    w = row_w.view(1, -1).expand_as(ell)
    return (ell * w)[rows].sum() / w[rows].sum().clamp_min(1e-12)


def relation_class_contrast_reference(cos_logits, y, Om, row_w):
    """Row-by-row definition of relation_class_contrast, for the self-test."""
    total, wsum = 0.0, 0.0
    B, K = y.shape
    for b in range(B):
        for k in range(K):
            if y[b, k] < 0.5:
                continue
            den = torch.exp(cos_logits[b, k]) + sum(
                Om[k, c] * torch.exp(cos_logits[b, c]) for c in range(K) if y[b, c] < 0.5)
            total = total + row_w[k] * (torch.log(den) - cos_logits[b, k])
            wsum = wsum + row_w[k]
    return total / wsum


def class_anchored_log_weights(R, Y, alpha, target):
    """K x N log-weights of every training image j as a negative for anchor
    class k: log(1 + alpha x rho_jk), rho_jk = max over j's labels c of
    R[k, c] (how confusable image j is for class k). Only anchor classes in
    `target` (a K-length bool) get relation weights."""
    rho = relation_scores(R, np.asarray(Y, dtype=np.float32))            # N x K
    tgt = np.asarray(target, dtype=np.float32)
    return np.log1p(alpha * rho.T * tgt[:, None]).astype(np.float32)     # K x N


def class_anchored_contrast(S, y, logW, class_w):
    """Class-anchored contrast. S: K x N logits of every class vector against
    every image; y: N x K labels (0/1 float); logW: K x N log-weights of the
    negatives. For class k and each positive image p of k:
    -log(e^{S_kp} / (e^{S_kp} + sum over negatives j of w_kj e^{S_kj})),
    averaged over the positives of k, then over classes with weights
    class_w (classes without positives are skipped)."""
    pos = y.T > 0.5                                                      # K x N
    lse_neg = torch.logsumexp((S + logW).masked_fill(pos, float("-inf")), dim=1, keepdim=True)
    ell = F.softplus(lse_neg - S)                                        # K x N
    n_pos = pos.sum(1)
    per_class = (ell * pos).sum(1) / n_pos.clamp_min(1)
    w = class_w * (n_pos > 0)
    return (per_class * w).sum() / w.sum().clamp_min(1e-12)


def class_anchored_contrast_reference(S, y, logW, class_w):
    """Row-by-row definition of class_anchored_contrast, for the self-test."""
    total, wsum = 0.0, 0.0
    K, N = S.shape
    for k in range(K):
        pos = [j for j in range(N) if y[j, k] > 0.5]
        if not pos:
            continue
        den_neg = sum(torch.exp(logW[k, j] + S[k, j]) for j in range(N) if y[j, k] < 0.5)
        ell = sum(torch.log(torch.exp(S[k, p]) + den_neg) - S[k, p] for p in pos) / len(pos)
        total = total + class_w[k] * ell
        wsum = wsum + class_w[k]
    return total / wsum


# ------------------------------------------------------- test-time tools ----
def relation_penalties(Fs, R, values, referee=None):
    """For every relation value v in `values`: an N x K array whose column c
    is max over classes k with R[c, k] = v of max(0, ref[:, k] - ref[:, c])
    (0 where c has no such class), with ref = referee (default Fs)."""
    Fs = np.asarray(Fs, dtype=np.float64)
    ref = Fs if referee is None else np.asarray(referee, dtype=np.float64)
    out = {}
    for v in values:
        P = np.zeros_like(Fs)
        for c in range(Fs.shape[1]):
            ks = np.flatnonzero(np.isclose(R[c], v))
            if ks.size:
                P[:, c] = np.maximum(ref[:, ks] - ref[:, [c]], 0.0).max(1)
        out[v] = P
    return out


def relation_rescore(Fs, R, gammas, target, referee=None, penalties=None):
    """Relation-typed re-scoring. For class c (in `target`) and each relation
    value v with strength gammas[v]: lower c's score on images where a class
    k with R[c, k] = v scores higher than c:
    F'[:, c] = F[:, c] - sum over v of gammas[v] x max over such k of
    max(0, ref[:, k] - ref[:, c]), with ref = referee (default F).
    `penalties` (from relation_penalties) can be passed to avoid recomputing."""
    Fs = np.asarray(Fs, dtype=np.float64)
    used = [v for v, g in gammas.items() if g]
    if not used:
        return Fs.copy()
    P = penalties if penalties is not None else relation_penalties(Fs, R, used, referee)
    tgt = np.asarray(target, bool)
    total = sum(gammas[v] * P[v] for v in used)
    out = Fs.copy()
    out[:, tgt] -= total[:, tgt]
    return out


def headroom_ap(S, Y, R, objects, values, draws, seed, ap_fn):
    """Oracle: per class c, the negatives whose most confusable label has a
    relation value R[c, k] in `values` get scores drawn from c's unrelated
    negatives (no label with R[c, k] > 0) with the same number of objects.
    Returns (AP, oracle AP averaged over draws) per class."""
    rng = np.random.default_rng(seed)
    S = np.asarray(S, dtype=np.float64)
    Yb = np.asarray(Y) > 0.5
    K = Yb.shape[1]
    vals = np.round(np.asarray(sorted(values), dtype=np.float64), 4)
    base, orc = np.full(K, np.nan), np.full(K, np.nan)
    for c in range(K):
        y = Yb[:, c]
        if not y.any():
            continue
        base[c] = ap_fn(S[:, c], y.astype(np.float64))
        neg = np.flatnonzero(~y)
        rho = np.round((Yb[neg] * R[c][None, :]).max(1), 4)
        hit = neg[np.isin(rho, vals) & (rho > 0)]
        clean = neg[rho == 0]
        if hit.size == 0 or clean.size == 0:
            orc[c] = base[c]
            continue
        res = []
        for _ in range(draws):
            s = S[:, c].copy()
            for lev in np.unique(objects[hit]):
                h = hit[objects[hit] == lev]
                pool = clean[objects[clean] == lev]
                pool = pool if pool.size else clean
                s[h] = S[rng.choice(pool, size=h.size), c]
            res.append(ap_fn(s, y.astype(np.float64)))
        orc[c] = float(np.mean(res))
    return base, orc


# ------------------------------------------------------------------ model ----
class TextAdapter(nn.Module):
    """Residual adapter (identity at start) + cosine classifier initialised
    with the class-name text embeddings, with a learned scale and per-class
    bias set so that initial probabilities match the class priors."""

    def __init__(self, T, prior_logit, zs_center, hidden=512, dropout=0.1, scale=20.0):
        super().__init__()
        d = T.shape[1]
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("T0", F.normalize(T, dim=-1))
        self.delta = nn.Parameter(torch.zeros_like(self.T0))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        self.bias = nn.Parameter(prior_logit - scale * zs_center)

    def embed(self, x):
        return F.normalize(x + self.fc2(self.drop(F.relu(self.fc1(x)))), dim=-1)

    def class_vectors(self):
        return F.normalize(self.T0 + self.delta, dim=-1)

    def logits_from(self, z):
        return self.log_scale.exp() * z @ self.class_vectors().T + self.bias


def make_text_adapter(X, Y, T, cfg, device):
    p = np.clip(np.asarray(Y, dtype=np.float64).mean(0), 1e-4, 1 - 1e-4)
    prior_logit = torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float32)
    zs_center = torch.as_tensor((np.asarray(X) @ np.asarray(T).T).mean(0), dtype=torch.float32)
    return TextAdapter(torch.as_tensor(T, dtype=torch.float32), prior_logit, zs_center,
                       cfg["hidden"], cfg["dropout"], cfg["scale"]).to(device)


# --------------------------------------------------------------- training ----
def train_text_adapter(X, Y, T, cfg, seed, device, R=None, target=None, eval_X=(),
                       ca_target=None):
    """Train the CLIP-initialised classifier on (X, Y). cfg keys: loss
    ('BCE'/'ASL'), epochs, lr, wd, batch, hidden, dropout, scale, tau, lam
    (0 = no contrastive term), alpha (0 = uniform negatives), q, target,
    cls_lr (learning-rate multiplier for the class vectors; 0 keeps them
    at the CLIP text embeddings); round 4: cc, cc_alpha, margin; round 5:
    ca (weight of the class-anchored contrast), ca_alpha (its relation
    weight; 0 = uniform negatives), with anchor classes `ca_target`.

    Returns, for every array in eval_X, the pair (trained, zero_shot) of
    per-class standardised scores (statistics from the training images)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N, K = Y.shape
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    model = make_text_adapter(X, Y, T, cfg, device)
    groups = [{"params": [*model.fc1.parameters(), *model.fc2.parameters(),
                          model.log_scale, model.bias], "lr": cfg["lr"]}]
    cls_lr = float(cfg.get("cls_lr", 1.0))
    if cls_lr > 0:
        groups.append({"params": [model.delta], "lr": cfg["lr"] * cls_lr})
    else:
        model.delta.requires_grad_(False)
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["epochs"] * math.ceil(N / cfg["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, steps) / steps)))
    cls_loss = {"BCE": bce_sum, "ASL": asl_sum}[cfg["loss"]]
    lam = float(cfg.get("lam", 0.0))
    if lam > 0:
        tgt = np.ones(K, bool) if target is None else np.asarray(target, bool)
        Rr = R if R is not None else np.zeros((K, K), np.float32)
        W = torch.as_tensor(candidate_weights(Rr, Y, float(cfg.get("alpha", 0.0)), tgt),
                            device=device)
        rw = torch.as_tensor(row_weights(Y, float(cfg.get("q", 0.0))), device=device)
    lam_c = float(cfg.get("cc", 0.0))          # class contrast weight (round 4)
    margin = float(cfg.get("margin", 0.0))     # relation-typed margin (round 4)
    if lam_c > 0 or margin > 0:
        tgt_c = np.ones(K, bool) if target is None else np.asarray(target, bool)
        Rc = R if R is not None else np.zeros((K, K), np.float32)
    if lam_c > 0:
        Om = torch.as_tensor(class_contrast_weights(Rc, float(cfg.get("cc_alpha", 0.0)), tgt_c),
                             device=device)
        rw_c = torch.as_tensor(row_weights(Y, float(cfg.get("q", 0.0))), device=device)
    if margin > 0:
        Mg = torch.as_tensor(margin * relation_scores(Rc, Y) * tgt_c[None, :]
                             * (1.0 - np.asarray(Y, dtype=np.float32)), device=device)
    lam_ca = float(cfg.get("ca", 0.0))         # class-anchored contrast weight (round 5)
    if lam_ca > 0:
        tgt_ca = np.ones(K, bool) if ca_target is None else np.asarray(ca_target, bool)
        Rca = R if R is not None else np.zeros((K, K), np.float32)
        logW_ca = torch.as_tensor(class_anchored_log_weights(
            Rca, Y, float(cfg.get("ca_alpha", 0.0)), tgt_ca), device=device)
        cw_ca = torch.ones(K, device=device)
    model.train()
    for _ in range(cfg["epochs"]):
        order = rng.permutation(N)
        for start in range(0, N, cfg["batch"]):
            b = torch.as_tensor(order[start:start + cfg["batch"]], device=device)
            if lam_ca > 0:                            # round 5 (new code path)
                z_all = model.embed(Xt)               # every training image, with gradient
                z_b = z_all[b]
                logits = model.logits_from(z_b)
                loss = cls_loss(logits + Mg[b] if margin > 0 else logits, Yt[b])
                if lam > 0:
                    loss = loss + lam * label_supcon(z_all, Yt, b, W, rw, cfg["tau"])
                if lam_c > 0:
                    cos_l = model.log_scale.exp() * z_b @ model.class_vectors().T
                    loss = loss + lam_c * relation_class_contrast(cos_l, Yt[b], Om, rw_c)
                S_all = model.log_scale.exp() * model.class_vectors() @ z_all.T
                loss = loss + lam_ca * class_anchored_contrast(S_all, Yt, logW_ca, cw_ca)
            elif lam_c > 0 or margin > 0:
                z_all = model.embed(Xt) if lam > 0 else None
                z_b = z_all[b] if lam > 0 else model.embed(Xt[b])
                logits = model.logits_from(z_b)
                loss = cls_loss(logits + Mg[b] if margin > 0 else logits, Yt[b])
                if lam > 0:
                    loss = loss + lam * label_supcon(z_all, Yt, b, W, rw, cfg["tau"])
                if lam_c > 0:
                    cos_l = model.log_scale.exp() * z_b @ model.class_vectors().T
                    loss = loss + lam_c * relation_class_contrast(cos_l, Yt[b], Om, rw_c)
            elif lam > 0:
                z_all = model.embed(Xt)                   # every image, with gradient
                loss = (cls_loss(model.logits_from(z_all[b]), Yt[b])
                        + lam * label_supcon(z_all, Yt, b, W, rw, cfg["tau"]))
            else:
                loss = cls_loss(model.logits_from(model.embed(Xt[b])), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        def scores(A):
            At = torch.as_tensor(A, dtype=torch.float32, device=device)
            return (model.logits_from(model.embed(At)).cpu().numpy().astype(np.float64),
                    (At @ model.T0.T).cpu().numpy().astype(np.float64))
        tr_t, tr_z = scores(X)
        out = []
        for A in eval_X:
            t, z = scores(A)
            out.append((standardise_by(tr_t, t), standardise_by(tr_z, z)))
    return out


def train_linear_probe(X_std, Y, loss_name, lr, epochs, seed, device, X_zs_train,
                       T, eval_X=(), eval_X_zs=(), batch=64, wd=1e-4):
    """Linear head on standardised features, exactly as in the context-check
    notebook (same seeds give the same model). Returns standardised
    (trained, zero_shot) scores like train_text_adapter; zero-shot scores use
    the unstandardised (L2-normalised) features."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    loss_fn = {"BCE": bce_mean, "ASL": asl_sum}[loss_name]
    Xt = torch.as_tensor(X_std, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    head = nn.Linear(X_std.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        order = rng.permutation(len(Y))
        for start in range(0, len(order), batch):
            b = torch.as_tensor(order[start:start + batch], device=device)
            loss = loss_fn(head(Xt[b]), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        f = lambda A: head(torch.as_tensor(A, dtype=torch.float32, device=device)).cpu().numpy()
        tr_t = f(X_std).astype(np.float64)
        tr_z = np.asarray(X_zs_train, dtype=np.float64) @ np.asarray(T, dtype=np.float64).T
        out = []
        for A, Az in zip(eval_X, eval_X_zs):
            z = np.asarray(Az, dtype=np.float64) @ np.asarray(T, dtype=np.float64).T
            out.append((standardise_by(tr_t, f(A).astype(np.float64)), standardise_by(tr_z, z)))
    return out
