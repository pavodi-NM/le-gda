"""Proto-Adapter, implemented from the paper (no official code is available).

Kato, Nota & Aoki, "Proto-Adapter: Efficient Training-Free CLIP-Adapter for
Few-Shot Image Classification", Sensors 24(11):3624, 2024.
DOI 10.3390/s24113624.

This module is the method only; splits, support draws and reporting are in
scripts/protocol_adapter.py and the runner is scripts/run_proto_adapter.py.

The architecture:

    P  = class means of the support features                       # S1, Eq.(5)
    P /= ||P||_2 along axis 0 ; P /= ||P||_2 along axis 1          # S2, Table 3
    p_proto = f @ P.T                                             # S3, Eq.(6)
    logits  = 100 * f @ W_text + alpha * p_proto                   # S4/U9, Eq.(7)

and the -F variant adds an ArcFace margin on `p_proto` during training
(S5/U4, Eq.(8)) and changes nothing at inference (S11).

Below, section A lists what the paper states and section B what it leaves
unstated, with the choice made here. Equation and table numbers are the
paper's.

==========================================================================
A. What the paper states
==========================================================================

| # | Clause | Where |
|---|---|---|
| **S1** | Prototype = mean of the class's training-image visual features. `c_n = (1/K) Σ_k CLIP_vis(I_{n,k})`. Adapter weights `F_proto = [c_1, …, c_N]`. | §2.2.1, Eq. (4)–(5) |
| **S2** | The prototypes are L2-normalized **twice**: channel-wise (first axis of `F_proto ∈ R^{N×D}`) **then** class-wise (second axis); the best of the four Table-3 settings. | §3.4, Table 3 |
| **S3** | Adapter logits are a bare matrix product, `p_proto = F_proto^T f`, with no activation and no `β`. | §2.2.1, Eq. (6); Fig. 3 |
| **S4** | Fusion is linear: `logits = α · p_proto + W_text^T f`. | Eq. (7) |
| **S5** | Fine-tuning uses the **ArcFace additive angular margin**: with `W_j^T f_i = cos θ_j`, the positive-class logit becomes `cos(θ_{y_i} + m)`; cross-entropy on those logits. | §2.2.2, Eq. (8) |
| **S6** | Only the adapter weights `W = F_proto` are trained. The encoders are frozen; `α` and `m` are searched, not learned. | §2.2.2, §3.1 |
| **S7** | Adam, batch 256, 20 epochs, lr 4e-4 → 4e-5, cosine annealing. | §3.1 |
| **S8** | Augmentation = random crop, resize, random horizontal flip, as in Tip-Adapter. | §3.1 |
| **S9** | `m` and `α` are tuned per dataset on the validation set. | §3.1 |
| **S10** | Prompt templates as in Tip-Adapter (7 for ImageNet). | §3.1, §3.2 |
| **S11** | The margin is applied only during training; inference is unchanged. | §2.2.2 |
| **S12** | The `m` grid is `{0, 0.1, 0.2, 0.3, 0.4}`. | Table 4 |
| **S13** | Two variants: Proto-Adapter (training-free) and Proto-Adapter-F (fine-tuned). | §3.1 |

S2 in coordinates. Eq. (4) declares `F_proto ∈ R^{D×N}`, §3.4 `R^{N×D}`. The
Table-3 wording, "channel-wise (along the first axis) followed by class-wise
(along the second axis)", is consistent only in the `N×D` convention, so with
`P ∈ R^{N×D}`:

    P ← P / ‖P‖₂ along axis 0     # channel-wise, one divisor per feature dim
    P ← P / ‖P‖₂ along axis 1     # class-wise,   one divisor per class

The class-wise step comes last, which leaves every class weight on the unit
sphere, as S5 requires for `W_j^T f = cos θ_j`.

==========================================================================
B. What the paper does not state, and the choice made here
==========================================================================

| # | Question | Choice | Alternative | Reason |
|---|---|---|---|---|
| **U1** | The ArcFace scale `s` (not given in the paper). | **Selected on validation, then fixed**: per scale, the mean over the 11 datasets of each dataset's best validation accuracy over margins. Grid `s ∈ {1, 4, 8, 16, 30, 64, 128, 256, 512}`; the maximum is interior at **s = 64**, used for every dataset and backbone. | per-dataset `s`; fitting `s` to Table 4 | Without a scale, cross-entropy acts on logits in `[-1, 1]` and the softmax is nearly uniform. S9 tunes hyperparameters on validation; fitting to Table 4 would be test-set selection, so Table 4 is used only as a check. |
| **U2** | Is each image feature L2-normalized before averaging in Eq. (5)? | **Yes.** | raw mean, then normalize | §2.1 defines `CLIP_vis(·)` as L2-normalized features, and Eq. (5) reuses that symbol. |
| **U3** | Prototypes from augmented-mean features (Tip-Adapter's `augment_epoch = 10`) or one clean pass? | **Augmented mean, 10 passes.** | 1 clean pass | S8 says the augmentation is Tip-Adapter's; this gives Proto-Adapter and Tip-Adapter identical features. |
| **U4** | Cross-entropy on the adapter's logits alone, or on the fused `α·p + text`? | **The adapter's alone.** | fused (Tip-Adapter-F's convention), `--loss-on fused` | §2.2.2 applies the loss to "the obtained logits", i.e. Eq. (8)'s margin logits; neither `α` nor `W_text` appears there. |
| **U5** | Is `W` re-normalized in the forward pass? | **Yes, class-wise, every forward (train and test).** | normalize at init only | S5 defines the logit as `cos θ_j`, which needs `‖W_j‖ = 1`; ArcFace also normalizes `W` in the forward. The channel-wise half of S2 is therefore an initialization step. |
| **U6** | The `α` grid. | `logspace(1e-3, 10^(3+60/29), 40)`, the grid used for every method here. | Tip-Adapter's `search_scale` grid | Tip-Adapter's α ceiling (13, or 3 on ImageNet) is too low here: `p_proto` is a cosine in `[-1, 1]` against a zero-shot logit scaled by 100. |
| **U7** | The `m` grid. | `{0, 0.1, 0.2, 0.3, 0.4}` (S12). | — | Read off Table 4. |
| **U8** | Are `m` and `α` searched jointly, and `α` before or after fine-tuning? | **Jointly, after**: for each `m`, fine-tune, then search `α` on validation; keep the best `(m, α)` on validation. | `α` fixed from the training-free variant | The literal reading of S9. The margins share one augmentation stream, so this costs one training run (see below). |
| **U9** | The zero-shot logit scale (Eq. (7) has none). | **`100 ·`**, Tip-Adapter's convention. | no scale | Only the ratio to `α · p_proto` matters, so this just reparameterizes the α grid and keeps α comparable with Tip-Adapter. |
| **U10** | ImageNet support draw (the paper draws from the full 1.28 M training set). | **The 32-per-class ImageNet subset** used for every method here. | — | The training data available; 16 random images of 32 and of ~1300 are both random draws. |
| **U11** | ImageNet validation set (Tip-Adapter's ImageNet code uses the test set as validation). | **10 per class, disjoint from the 40-per-class test set.** | — | No hyperparameter is selected on test. |
| **U12** | ArcFace monotonicity guard: `cos(θ+m)` stops decreasing once `θ+m > π`. | **The official ArcFace fallback**, `cos θ − m·sin(π−m)` beyond the threshold. | "easy margin": fall back to `cos θ` | What the ArcFace release does; it only affects samples wrong by more than `π − m`. |
| **U13** | Randomness of the augmented-mean prototype (U3 averages 10 random views). | **Encode once and cache** under `caches/<ds>/proto_adapter/`, keyed by the support set's image paths. | seed the augmentation stream | Tip-Adapter's `build_cache_model` has the same property. Re-encoding ImageNet RN50 K=16 with different worker seeding moved Proto-Adapter by 0.07 (62.08 → 62.01). |

Cost of U8. The expensive part of fine-tuning is pushing the augmented images
through the frozen encoder, which does not depend on `m`. With the
augmentation stream and batch order fixed by a seed, training the five margin
heads side by side on the same batches is identical, step for step, to five
separate runs, since the heads do not interact.
`scripts/run_proto_adapter.py --prove-margin-sharing` checks this by training
each head separately and asserting bit-equality.

==========================================================================
C. Check against the paper
==========================================================================

`run_proto_adapter.py --protocol theirs` reproduces the paper's own setting
(their 16-shot draw, their validation and test splits, RN50, one draw).
Published targets, RN50, 16 shots:

| target | Proto-Adapter | Proto-Adapter-F |
|---|---|---|
| ImageNet (Table 1) | 63.89 | 66.17 |
| 11-dataset average (Table 2) | 72.69 | 75.59 |
| RN101 / ViT-B/32 / ViT-B/16 ImageNet (Table 5) | 66.81 / 67.07 / 72.25 | 69.12 / 69.27 / 74.23 |

Table 3 (ImageNet, training-free, 1/4/16 shots, `--norm-ablation`) checks the
normalization:

| channel-wise | class-wise | 1 | 4 | 16 |
|---|---|---|---|---|
| — | — | 60.40 | 60.61 | 61.84 |
| ✓ | — | 60.65 | 60.96 | 61.75 |
| — | ✓ | 60.66 | 61.34 | 62.70 |
| ✓ | ✓ | 60.77 | 61.92 | 63.89 |

Table 4 (ImageNet, 16-shot): without fine-tuning 63.89, then
`m = 0 / .1 / .2 / .3 / .4` → 65.56 / 65.96 / 66.17 / 65.95 / 65.54.

Under this setting our implementation gives an 11-dataset average of 72.49
(training-free) and 75.06 (fine-tuned) against the paper's 72.69 and 75.59.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# S12 / U7 -- the paper's own sweep, read off Table 4.
MARGINS = (0.0, 0.1, 0.2, 0.3, 0.4)

# S7 -- Adam, batch 256, 20 epochs, 4e-4 -> 4e-5 cosine.
TRAIN = dict(lr=4e-4, lr_min=4e-5, epochs=20, batch_size=256)

# U9 -- Tip-Adapter's zero-shot logit scale, so alpha is comparable with
# Tip-Adapter's.
CLIP_LOGIT_SCALE = 100.0


# ---------------------------------------------------------------- S1, S2 ----
def class_means(feats, labels, n_cls):
    """Eq. (5): c_n = (1/K) sum_k CLIP_vis(I_{n,k}).

    `feats` are the per-image L2-normalized visual features (U2). On the
    cached path these are `caches/<ds>/collapse/<tag>_pool_f.pt` rows,
    `normalize(mean_a encode(aug_a(img)))` over 10 augmentation passes, the
    same features `utils.build_cache_model` puts in Tip-Adapter's cache (U3).

    Returns P [n_cls, D], fp32, un-normalized. Raises if a class is empty.
    """
    feats = feats.float()
    P = torch.zeros(n_cls, feats.shape[1], device=feats.device, dtype=feats.dtype)
    cnt = torch.zeros(n_cls, device=feats.device, dtype=feats.dtype)
    P.index_add_(0, labels, feats)
    cnt.index_add_(0, labels, torch.ones_like(labels, dtype=feats.dtype))
    if (cnt == 0).any():
        raise AssertionError(
            f"class_means: {(cnt == 0).sum().item()} of {n_cls} classes have no "
            f"support image; Eq.(5) is undefined there")
    return P / cnt[:, None]


def normalize_prototypes(P, channel_wise=True, class_wise=True):
    """S2 / Table 3, in the paper's `N x D` convention.

    "channel-wise (along the first axis) followed by class-wise (along the
    second axis)". With P [N, D], reducing axis 0 gives one norm per feature
    dimension (channel-wise) and axis 1 one norm per class (class-wise). The
    class-wise step is last, so every row ends on the unit sphere, as Eq. (8)
    needs for `W_j^T f = cos theta_j`.

    `channel_wise`/`class_wise` can be switched off only to reproduce the four
    rows of Table 3; the method uses both.
    """
    if channel_wise:
        P = P / P.norm(dim=0, keepdim=True).clamp_min(1e-12)
    if class_wise:
        P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return P


def build_prototypes(feats, labels, n_cls, channel_wise=True, class_wise=True):
    """S1 + S2 in one call: support features -> adapter weights F_proto."""
    return normalize_prototypes(class_means(feats, labels, n_cls),
                                channel_wise, class_wise)


# -------------------------------------------------------------------- S3 ----
class ProtoAdapter(nn.Module):
    """S3/S6: a single linear layer of fixed size `n_cls x D`, no activation.

    U5 -- the weight is L2-normalized class-wise on every forward, training and
    inference alike, because Eq. (8) defines the logit as `cos theta_j`, which
    needs `||W_j|| = 1`. The channel-wise half of S2 is an initialization step.
    """

    def __init__(self, prototypes):
        super().__init__()
        self.weight = nn.Parameter(prototypes.clone().float())

    def forward(self, f):
        """`f` L2-normalized [n, D] -> cos theta [n, n_cls]."""
        W = F.normalize(self.weight, dim=1)
        return f @ W.t()


# -------------------------------------------------------------------- S5 ----
def arcface_logits(cos, target, m, s):
    """Eq. (8): replace the positive-class logit with `cos(theta_y + m)`.

    `cos(theta+m) = cos theta cos m - sin theta sin m`.

    U12 -- monotonicity guard. `cos(theta+m)` stops decreasing in theta once
    `theta + m > pi`, so beyond that point we use the official ArcFace
    fallback `cos theta - m sin(pi - m)`.

    `s` is the ArcFace scale (U1): selected on validation (s = 64) and fixed
    for every dataset and backbone.
    """
    if m == 0.0:
        return s * cos
    cos_m, sin_m = math.cos(m), math.sin(m)
    th, mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
    sin = (1.0 - cos.pow(2)).clamp_min(0.0).sqrt()
    phi = cos * cos_m - sin * sin_m
    phi = torch.where(cos > th, phi, cos - mm)
    onehot = F.one_hot(target, cos.shape[1]).bool()
    return s * torch.where(onehot, phi, cos)


# ------------------------------------------------------------- S4 / U6/U9 ----
def fused_logits(cos, clip_logits, alpha):
    """Eq. (7), with U9's `100 *` already inside `clip_logits`."""
    return clip_logits + alpha * cos


def zeroshot_logits(feats, clip_weights):
    """`100 * f @ W_text` -- Tip-Adapter's convention, U9."""
    return CLIP_LOGIT_SCALE * feats.float() @ clip_weights.float()


def search_alpha(cos, clip_logits, y, alphas, chunk_elems=6e7):
    """S9/U6: pick alpha by top-1 on the VALIDATION set. Returns (alpha, acc).

    The alpha axis is evaluated in chunks so that ImageNet fits in GPU memory;
    argmax is per alpha, so chunking is exact. Ties go to the smallest alpha,
    matching `search_hp`'s strict `>`.
    """
    A = torch.tensor(alphas, device=cos.device, dtype=torch.float32)
    n, C = cos.shape
    step = max(1, int(chunk_elems // max(1, n * C)))
    accs = []
    for i in range(0, len(A), step):
        a = A[i:i + step][:, None, None]
        pred = (clip_logits[None] + a * cos[None]).argmax(-1)
        accs.extend((pred == y[None]).float().mean(-1).mul(100).tolist())
    best = max(range(len(accs)), key=lambda i: (accs[i], -i))
    return alphas[best], accs[best]


def accuracy(cos, clip_logits, y, alpha):
    return (fused_logits(cos, clip_logits, alpha).argmax(-1) == y).float().mean().item() * 100


# -------------------------------------------------------------------- U8 ----
def cosine_lr(step, total, lr=TRAIN["lr"], lr_min=TRAIN["lr_min"]):
    """S7: 4e-4 -> 4e-5 on a cosine, per optimizer step."""
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / max(1, total)))
