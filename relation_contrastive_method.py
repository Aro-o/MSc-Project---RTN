# ---------------------------------------------------------------------------
# Core code for the relation-aware contrastive experiment (frozen CLIP features)
#
# MulSupCon follows the official implementation (Zhang & Wu, AAAI 2024,
# github: MulSupCon, vector-data configuration): MLP backbone, projection
# head, MoCo-style momentum key encoder and queue, MulSupCon positives (each
# label of an anchor separately), WeightedSupCon loss with 1/len(rows)
# weights, 50% feature masking, Adam + cosine warm restarts, then fine-tuning
# with BCE (backbone lr x0.1, ReduceLROnPlateau, early stopping on a dev set).
# The other variants change only how negatives are weighted inside that loss.
#
# SCHaNe (Long et al. 2023) and LCL (Suresh & Ong 2021) have no official code
# for this setting; their weights are implemented from the papers' equations:
#   SCHaNe Eq. 3: negative k of anchor i gets
#                 beta_k = |N| exp(s_ik/tau) / sum_{n in N} exp(s_in/tau)
#   LCL Eq. 3:    every term of anchor i is weighted by w_{i, y_a}, the
#                 anchor's weighting-network probability for candidate a's
#                 class (w_{i, y_i} for positives, numerator included).
#                 Multi-label adaptation: sigmoid probabilities (BCE-trained
#                 weighting network) and, for a negative with several labels,
#                 the largest of the anchor's probabilities for those labels.
# ---------------------------------------------------------------------------
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_VERSION = "2026-10-30"     # part of the saved-run tag (unchanged: same computation)

# Official settings (codes/config/pretrain_yeast.yaml and train_yeast.yaml)
PRETRAIN = dict(epochs=150, iters_per_epoch=1000, batch_size=64, cap_Q=256,
                momentum=0.999, temperature=0.1, out_dim=256, n_hidden=1,
                lr=4e-4, weight_decay=1e-5, T_0=50, T_mult=2, eta_min=1e-4,
                mask_p=0.5, running_alpha=0.98)
# Fine-tuning: official rules (BCE, Adam, backbone lr x0.1, ReduceLROnPlateau,
# early stopping after 20 epochs without a lower dev loss, test predictions
# at the lowest dev loss). One change: the official cap of 30 epochs (set for
# the 14-label yeast data) is raised to max_epochs so that early stopping,
# not the cap, ends training. The official-cap result is read from the same
# run (its first 30 epochs are identical) and reported as a secondary check.
FINETUNE = dict(official_epochs=30, max_epochs=400, iters_per_epoch=50,
                batch_size=32, lr=4e-4, lrp=0.1, plateau_factor=0.1,
                plateau_patience=2, plateau_cooldown=1, plateau_threshold=1e-3,
                early_stop_patience=20)


# --------------------------- data helpers ----------------------------------
def iterative_split(Y, test_frac, seed):
    """Iterative stratification for multi-label data (Sechidis et al., 2011).
    Returns 0 (fit) / 1 (dev) for each row, keeping every label's share close
    to the target in both parts."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=bool)
    n = Y.shape[0]
    frac = np.array([1.0 - test_frac, test_frac])
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


class Scaler:
    """sklearn.preprocessing.StandardScaler (population std, zero std -> 1)."""

    def fit(self, X):
        self.mean = X.mean(0)
        std = X.std(0)
        self.scale = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        return ((X - self.mean) / self.scale).astype(np.float32)


class OfficialSampler:
    """Batches exactly as the official RandomSampler: indices shuffled once
    by parse_data (RandomState(seed)), again by the sampler (RandomState(seed)),
    then read cyclically, reshuffling each time the pointer wraps."""

    def __init__(self, n, batch_size, seed):
        indices = list(range(n))
        np.random.RandomState(seed=seed).shuffle(indices)     # parse_data
        self.indices = np.array(indices)
        self.rs = np.random.RandomState(seed=seed)            # Sampler
        self.rs.shuffle(self.indices)
        self.ptr, self.batch_size = 0, batch_size

    def next(self):
        out = np.empty(self.batch_size, dtype=np.int64)
        for j in range(self.batch_size):
            out[j] = self.indices[self.ptr]
            self.ptr = (self.ptr + 1) % len(self.indices)
            if not self.ptr:
                self.rs.shuffle(self.indices)
        return out


def mask_views(x, p, generator):
    """Official vector augmentation: each feature kept with probability 1-p."""
    if not p:
        return x
    keep = (torch.rand(x.shape, generator=generator, device=x.device) > p)
    return x * keep.to(x.dtype)


# --------------------------- models (official) ------------------------------
class Backbone(nn.Module):
    """codes/backbone.py: MLP (vector data)."""

    def __init__(self, in_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=dropout))

    def forward(self, x):
        return self.mlp(x)


class ProjHead(nn.Module):
    """codes/models.py: MLP projection head."""

    def __init__(self, in_dim=256, out_dim=256, n_hidden=1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_hidden):
            nxt = d // 2 if d >= 2 * out_dim else out_dim
            layers += [nn.Linear(d, nxt), nn.BatchNorm1d(nxt), nn.ReLU(),
                       nn.Dropout(p=0.1)]
            d = nxt
        layers.append(nn.Linear(d, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim=256, n_hidden=1):
        super().__init__()
        self.backbone = Backbone(in_dim)
        self.proj_head = ProjHead(256, out_dim, n_hidden)

    def forward(self, x):
        return self.proj_head(self.backbone(x))


class Classifier(nn.Module):
    """codes/models.py 'Simple': backbone + linear output layer."""

    def __init__(self, in_dim, n_class):
        super().__init__()
        self.backbone = Backbone(in_dim)
        self.output_layer = nn.Linear(256, n_class)

    def forward(self, x):
        return self.output_layer(self.backbone(x))


class PretrainModel(nn.Module):
    """codes/models.py PretrainModel (MoCo queue + momentum key encoder)."""

    def __init__(self, in_dim, n_class, cap_Q, momentum, out_dim, n_hidden):
        super().__init__()
        self.encoder_q = Encoder(in_dim, out_dim, n_hidden)
        self.encoder_k = Encoder(in_dim, out_dim, n_hidden)
        self.m, self.cap_Q = momentum, cap_Q
        self.register_buffer("QueuePtr", torch.tensor([0], dtype=torch.int32))
        self.register_buffer("QueueFull", torch.tensor([False]))
        self.register_buffer("MoCoQueue", nn.functional.normalize(
            torch.randn(out_dim, cap_Q), dim=0))
        self.register_buffer("LabelQueue", torch.zeros(cap_Q, n_class))
        for pq, pk in zip(self.encoder_q.parameters(),
                          self.encoder_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False
        self._q = list(self.encoder_q.parameters())
        self._k = list(self.encoder_k.parameters())
        # Queue pointer and 'full' flag kept on the CPU as well, so no GPU
        # synchronisation is needed each step (same values as the buffers).
        self._ptr, self._full = 0, False

    @torch.no_grad()
    def momentum_update(self):
        torch._foreach_mul_(self._k, self.m)
        torch._foreach_add_(self._k, self._q, alpha=1 - self.m)

    def forward(self, x1, x2, labels):
        """Returns (scores, reference labels). Same order of operations as the
        official forward(): momentum update, queries, keys, clone of the
        queue, enqueue, then scores (batch keys only until the queue is
        full, then batch keys + queue)."""
        if self.training:
            self.momentum_update()
        q = nn.functional.normalize(self.encoder_q(x1), dim=1)
        with torch.no_grad():
            k = nn.functional.normalize(self.encoder_k(x2), dim=1)
        queue_k = self.MoCoQueue.clone().detach()
        queue_labels = self.LabelQueue.clone().detach()
        if self.training:
            B = len(labels)
            assert not self.cap_Q % B
            ptr = self._ptr
            self.MoCoQueue[:, ptr:ptr + B] = k.T
            self.LabelQueue[ptr:ptr + B] = labels
            ptr = (ptr + B) % self.cap_Q
            self._ptr = ptr
            self.QueuePtr.fill_(ptr)
            if not ptr:
                self._full = True
                self.QueueFull.fill_(True)
        batch_score = q @ k.T
        if not self._full:
            return batch_score, labels
        scores = torch.cat([batch_score, q @ queue_k], dim=1)
        ref_labels = torch.cat([labels, queue_labels], dim=0)
        return scores, ref_labels


# --------------------------- losses -----------------------------------------
def mulsupcon_loss(scores, batch_labels, ref_labels, tau, mode="uniform",
                   relation=None, alpha=0.0, p_anchor=None):
    """MulSupCon loss with optional negative weights.

    One row per (anchor image, anchor label k), in the official order
    (torch.where on the batch labels). Positives of a row: candidates that
    carry label k (the anchor's own key included). Loss per row:
        -(1/|P|) sum_p log( w_p e^{s_p/tau} / sum_a w_a e^{s_a/tau} )
    averaged over rows (official weights 1/len(rows)). With all w = 1 this is
    exactly the official MulSupCon/WeightedSupCon loss.

    mode:
      uniform    : w = 1 (MulSupCon)
      schane     : negatives w = |N| e^{s/tau} / sum_N e^{s/tau}
                   (Long et al. 2023, Eq. 3), positives 1
      lcl        : LCL Eq. 3 (Suresh & Ong 2021), weights from the anchor's
                   weighting-network probabilities p_anchor (B x C):
                   positives w = p_anchor[i, k] (numerator included),
                   negatives w = max over the candidate's labels c of
                   p_anchor[i, c]
      relation   : negatives w = 1 + alpha * max_c R[k, c] over the
                   candidate's labels c (relation given as list of
                   (value, mask matrix L x L)); positives 1
      relation_schane : negatives w = beta (SCHaNe, Eq. 3) x (1 + alpha *
                   max_c R[k, c]); positives 1 (round-2 combination)
    """
    img_idx, lab_idx = torch.where(batch_labels == 1)
    logits = scores[img_idx] / tau                        # R x Q
    pos = ref_labels[:, lab_idx].T > 0.5                  # R x Q
    n_pos = pos.sum(1)
    if mode == "uniform":
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    else:
        if mode == "schane":
            masked = logits.masked_fill(pos, float("-inf"))
            n_neg = (~pos).sum(1, keepdim=True).clamp_min(1)
            log_beta = (masked - torch.logsumexp(masked, dim=1, keepdim=True)
                        + torch.log(n_neg.to(logits.dtype)))
            logw = torch.where(pos, torch.zeros_like(logits), log_beta)
        elif mode == "lcl":
            pa = p_anchor[img_idx]                                      # R x C
            w_pos = pa[torch.arange(len(lab_idx)), lab_idx].unsqueeze(1)  # R x 1
            w_cand = (pa.unsqueeze(1) * ref_labels.unsqueeze(0)).amax(2)  # R x Q
            w = torch.where(pos, w_pos.expand_as(logits), w_cand)
            logw = torch.log(w.clamp_min(1e-6))
        elif mode == "relation":
            score = torch.zeros_like(logits)
            refT = ref_labels.T                                         # C x Q
            for value, M in relation:
                has = (M[lab_idx] @ refT) > 0.5                         # R x Q
                score = torch.maximum(score, has.to(logits.dtype) * value)
            logw = torch.where(pos, torch.zeros_like(logits),
                               torch.log1p(alpha * score))
        elif mode == "relation_schane":
            masked = logits.masked_fill(pos, float("-inf"))
            n_neg = (~pos).sum(1, keepdim=True).clamp_min(1)
            log_beta = (masked - torch.logsumexp(masked, dim=1, keepdim=True)
                        + torch.log(n_neg.to(logits.dtype)))
            score = torch.zeros_like(logits)
            refT = ref_labels.T
            for value, M in relation:
                has = (M[lab_idx] @ refT) > 0.5
                score = torch.maximum(score, has.to(logits.dtype) * value)
            logw = torch.where(pos, torch.zeros_like(logits),
                               log_beta + torch.log1p(alpha * score))
        else:
            raise ValueError(mode)
        z = logits + logw
        log_prob = z - torch.logsumexp(z, dim=1, keepdim=True)
    per_row = -(log_prob * pos).sum(1) / n_pos
    return per_row.mean()


def mulsupcon_loss_vec(scores, batch_labels, ref_labels, tau, mode="uniform",
                       relation=None, alpha=0.0, p_anchor=None):
    """Same loss as mulsupcon_loss, computed for every (anchor, label) pair at
    once with fixed shapes (no data-dependent indexing, so no GPU sync).
    Rows whose label is absent from the anchor are computed and discarded.
    Shapes: scores B x Q, batch_labels B x C, ref_labels Q x C."""
    logits = scores / tau                                   # B x Q
    ref = ref_labels                                        # Q x C (0/1 float)
    refT = ref.T                                            # C x Q
    n_pos = ref.sum(0)                                      # C
    Yb = batch_labels                                       # B x C
    pos_sum = logits @ ref                                  # B x C: sum_p logit_p
    if mode == "uniform":
        den = torch.logsumexp(logits, dim=1, keepdim=True)  # B x 1
        row = -(pos_sum / n_pos.clamp_min(1) - den)         # B x C
    else:
        posmask = refT.unsqueeze(0) > 0.5                   # 1 x C x Q
        if mode == "relation":
            score = torch.zeros_like(refT)                  # C x Q
            for value, M in relation:
                has = (M @ refT) > 0.5                      # C x Q
                score = torch.maximum(score, has.to(logits.dtype) * value)
            lw = torch.where(refT > 0.5, torch.zeros_like(score),
                             torch.log1p(alpha * score))    # C x Q
            z = logits.unsqueeze(1) + lw.unsqueeze(0)       # B x C x Q
            pos_lw = 0.0
        elif mode == "schane":
            L3 = logits.unsqueeze(1).expand(-1, refT.shape[0], -1)
            # finite fill (not -inf) keeps gradients finite for rows without
            # negatives; exp(-1e4) is exactly 0 for the others
            neg_logits = L3.masked_fill(posmask, -1e4)
            n_neg = (refT < 0.5).sum(1).clamp_min(1).to(logits.dtype)  # C
            lse_neg = torch.logsumexp(neg_logits, dim=2, keepdim=True)
            log_beta = L3 - lse_neg + torch.log(n_neg).view(1, -1, 1)
            z = torch.where(posmask, L3, L3 + log_beta)
            pos_lw = 0.0
        elif mode == "relation_schane":
            L3 = logits.unsqueeze(1).expand(-1, refT.shape[0], -1)
            neg_logits = L3.masked_fill(posmask, -1e4)
            n_neg = (refT < 0.5).sum(1).clamp_min(1).to(logits.dtype)
            lse_neg = torch.logsumexp(neg_logits, dim=2, keepdim=True)
            log_beta = L3 - lse_neg + torch.log(n_neg).view(1, -1, 1)
            score = torch.zeros_like(refT)                  # C x Q
            for value, M in relation:
                has = (M @ refT) > 0.5
                score = torch.maximum(score, has.to(logits.dtype) * value)
            lw_rel = torch.log1p(alpha * score).unsqueeze(0)            # 1 x C x Q
            z = torch.where(posmask, L3, L3 + log_beta + lw_rel)
            pos_lw = 0.0
        elif mode == "lcl":
            lw_pos = torch.log(p_anchor.clamp_min(1e-6))               # B x C
            w_cand = (p_anchor.unsqueeze(1) * ref.unsqueeze(0)).amax(2)  # B x Q
            lw_neg = torch.log(w_cand.clamp_min(1e-6))                 # B x Q
            lw = torch.where(posmask, lw_pos.unsqueeze(2),
                             lw_neg.unsqueeze(1))                      # B x C x Q
            z = logits.unsqueeze(1) + lw
            pos_lw = lw_pos                                            # B x C
        else:
            raise ValueError(mode)
        den = torch.logsumexp(z, dim=2)                                # B x C
        row = -(pos_sum / n_pos.clamp_min(1) + pos_lw - den)
    keep = Yb > 0.5
    return (row * keep).sum() / keep.sum()


def relation_masks(sim, shared, n_train, n_images, related_fn, pct,
                   min_shared=2, lift_min=2.0):
    """Relation types for negative weights, from training labels only.
    Returns masks (L x L) for: related & seen together, related & never
    together, informative companion; plus 'any companion' (seen together)."""
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


# --------------------------- official reference (for the self-test) ---------
def official_generate_output_MulSupCon(batch_labels, ref_labels, scores,
                                       with_weight=False):
    """Verbatim logic of codes/utils.py generate_output_MulSupCon."""
    B = len(batch_labels)
    indices = torch.where(batch_labels == 1)
    scores = scores[indices[0]]
    labels = torch.zeros(len(scores), batch_labels.shape[1], device=scores.device)
    labels[range(len(labels)), indices[1]] = 1
    masks = (labels @ ref_labels.T).to(torch.bool)
    n_score_per_sample = batch_labels.sum(dim=1).to(torch.int16).tolist()
    if with_weight:
        weights_per_sample = [1 / (n * B) for n in n_score_per_sample
                              for _ in range(n)]
    else:
        weights_per_sample = [1 / len(scores) for n in n_score_per_sample
                              for _ in range(n)]
    weights_per_sample = torch.tensor(weights_per_sample, device=scores.device,
                                      dtype=torch.float32)
    return scores, [masks.to(torch.long), weights_per_sample]


def official_weighted_supcon(score, ref, temperature=0.1):
    """Verbatim logic of codes/losses.py WeightedSupCon.forward."""
    mask, weight = ref
    num_pos = mask.sum(1)
    loss = - (torch.log((F.softmax(score / temperature, dim=1))) * mask).sum(1) \
        / num_pos
    return (loss * weight).sum()


# --------------------------- training ---------------------------------------
def set_seeds(seed):
    """As the official Runner: torch, numpy and CUDA seeds, deterministic cuDNN."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain(X, Y, mode, seed, device, cfg=PRETRAIN, relation=None, alpha=0.0,
             tau=None, report=None, epoch_hook=None):
    """Contrastive pretraining with the official MulSupCon schedule.
    X: scaled fit-split features (float tensor on device); Y: float labels.
    Returns the backbone weights at the lowest running training loss (the
    official 'train_best' checkpoint) and the loss history."""
    set_seeds(seed)
    tau = cfg["temperature"] if tau is None else tau
    N, D = X.shape
    C = Y.shape[1]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = PretrainModel(D, C, cfg["cap_Q"], cfg["momentum"], cfg["out_dim"],
                          cfg["n_hidden"]).to(device)
    groups = [{"params": model.encoder_q.backbone.parameters(), "lr": cfg["lr"]},
              {"params": model.encoder_q.proj_head.parameters(), "lr": cfg["lr"]}]
    weighting = None
    if mode == "lcl":
        # LCL weighting network: same architecture as the fine-tuning model,
        # trained with its own (here BCE) loss; only its probabilities are
        # used by the contrastive loss (detached).
        weighting = Classifier(D, C).to(device)
        groups.append({"params": weighting.parameters(), "lr": cfg["lr"]})
    opt = torch.optim.Adam(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cfg["T_0"], T_mult=cfg["T_mult"], eta_min=cfg["eta_min"])
    sampler = OfficialSampler(N, cfg["batch_size"], seed)
    rel = None
    if relation is not None:
        rel = [(v, torch.as_tensor(m, dtype=torch.float32, device=device))
               for v, m in relation]
    best, best_state, history = math.inf, None, []
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        running = None        # ignite RunningAverage restarts every epoch
        model.train()
        if weighting is not None:
            weighting.train()
        # the epoch's batches, in the official sampler order, sent to the
        # device once (avoids a GPU synchronisation every iteration)
        batches = torch.from_numpy(np.stack(
            [sampler.next() for _ in range(cfg["iters_per_epoch"])])).to(device)
        for it in range(cfg["iters_per_epoch"]):
            idx = batches[it]
            x, y = X[idx], Y[idx]
            x1 = mask_views(x, cfg["mask_p"], gen)
            x2 = mask_views(x, cfg["mask_p"], gen)
            if mode == "lcl":
                logits_w = weighting(x1)                 # anchor's weighting output
                loss_w = F.binary_cross_entropy_with_logits(logits_w, y)
                p_anchor = torch.sigmoid(logits_w).detach()
                scores, ref = model(x1, x2, y)
                loss_c = mulsupcon_loss_vec(scores, y, ref, tau, "lcl",
                                            p_anchor=p_anchor)
                # LCL Eq. 6 with alpha = 0.5; the encoder's own classification
                # term L_e is left out (MulSupCon pretraining is contrastive
                # only; every variant is fine-tuned with BCE afterwards)
                loss = 0.5 * loss_w + 0.5 * loss_c
            else:
                scores, ref = model(x1, x2, y)
                loss = mulsupcon_loss_vec(scores, y, ref, tau, mode,
                                          relation=rel, alpha=alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            v = loss.detach()
            running = v if running is None else \
                cfg["running_alpha"] * running + (1 - cfg["running_alpha"]) * v
        sched.step()
        r = float(running)
        if not math.isfinite(r):
            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
        history.append(r)
        if r < best:                       # official ModelCheckpoint 'train_best'
            best = r
            best_state = copy.deepcopy(model.encoder_q.backbone.state_dict())
        if report is not None and ((epoch + 1) % 25 == 0 or epoch == 0):
            report(f"    epoch {epoch + 1:>3}/{cfg['epochs']}: running loss {r:.4f} "
                   f"({time.time() - t0:.0f} s)")
        if epoch_hook is not None:          # lets a worker stop between epochs
            epoch_hook(epoch + 1)
    return best_state, history


def neg_weight_matrix(Ybin, masks, alpha):
    """Relation weights for BCE negatives: 1 + alpha x the largest relation
    value between the class (column) and any label of the image (row)."""
    score = np.zeros(Ybin.shape, dtype=np.float32)
    for value, M in masks:
        has = (Ybin.astype(np.float32) @ M.T.astype(np.float32)) > 0.5
        score = np.maximum(score, value * has)
    return 1.0 + alpha * score


def weighted_bce(logits, y, neg_weights=None):
    """BCEWithLogitsLoss (mean); optional weights on the negative terms."""
    if neg_weights is None:
        return F.binary_cross_entropy_with_logits(logits, y)
    w = torch.where(y > 0.5, torch.ones_like(neg_weights), neg_weights)
    return F.binary_cross_entropy_with_logits(logits, y, weight=w)


def finetune(Xfit, Yfit, Xdev, Ydev, Xtest, seed, device, cfg=FINETUNE,
             backbone_state=None, w_fit=None, w_dev=None, epoch_hook=None):
    """Official 'train' stage: backbone (pretrained or random) + linear layer,
    BCE, Adam (backbone lr x lrp), ReduceLROnPlateau and early stopping on the
    dev loss; test predictions taken at the best dev loss.

    Returns (main, official). Training runs until early stopping or
    cfg['max_epochs']; 'main' is the lowest-dev-loss epoch of the whole run.
    The first cfg['official_epochs'] epochs are exactly the official run, so
    'official' is the lowest-dev-loss epoch among them (identical to 'main'
    when early stopping comes first)."""
    set_seeds(seed)
    D, C = Xfit.shape[1], Yfit.shape[1]
    model = Classifier(D, C).to(device)
    if backbone_state is not None:
        model.backbone.load_state_dict(backbone_state)
    opt = torch.optim.Adam(
        [{"params": model.backbone.parameters(), "lr": cfg["lr"] * cfg["lrp"]},
         {"params": model.output_layer.parameters(), "lr": cfg["lr"]}],
        lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg["plateau_factor"],
        patience=cfg["plateau_patience"], cooldown=cfg["plateau_cooldown"],
        threshold=cfg["plateau_threshold"])
    sampler = OfficialSampler(len(Xfit), cfg["batch_size"], seed)
    best, bad, official, main = math.inf, 0, None, None
    n_epochs = max(cfg["official_epochs"], cfg["max_epochs"])
    stopped = False
    for epoch in range(n_epochs):
        model.train()
        batches = torch.from_numpy(np.stack(
            [sampler.next() for _ in range(cfg["iters_per_epoch"])])).to(device)
        for it in range(cfg["iters_per_epoch"]):
            idx = batches[it]
            loss = weighted_bce(model(Xfit[idx]), Yfit[idx],
                                None if w_fit is None else w_fit[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            dev_logits = model(Xdev)
            dev_loss = float(weighted_bce(dev_logits, Ydev, w_dev))
        sched.step(dev_loss)
        if dev_loss < best:
            best, bad = dev_loss, 0
            with torch.no_grad():
                main = dict(test_logits=model(Xtest).cpu().numpy(),
                            dev_logits=dev_logits.cpu().numpy(),
                            best_epoch=epoch + 1, dev_loss=dev_loss)
            if epoch < cfg["official_epochs"]:
                official = main
        else:
            bad += 1
            if bad >= cfg["early_stop_patience"]:
                stopped = True
                break
        if epoch_hook is not None:
            epoch_hook(epoch + 1)
    if official is None:
        raise FloatingPointError("fine-tuning never produced a finite dev loss")
    official = dict(official, epochs_run=min(epoch + 1, cfg["official_epochs"]))
    main = dict(main, epochs_run=epoch + 1, stopped_early=stopped,
                final_lr=float(opt.param_groups[1]["lr"]))
    return main, official
