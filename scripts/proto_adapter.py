"""Proto-Adapter, implemented from the paper. No upstream code exists.

Kato, Nota & Aoki, "Proto-Adapter: Efficient Training-Free CLIP-Adapter for
Few-Shot Image Classification", Sensors 24(11):3624, 2024.
Reading of record, clause by clause: THE DESIGN NOTE REPRODUCED AT THE END
OF THIS DOCSTRING -- section A is what the paper states (S1-S13), section B
is what it leaves unstated and we had to choose (U1-U12). It is inlined here
rather than kept as a separate document precisely because the method is a
reading of prose: the choices and the code implementing them must not be
able to drift apart.

THIS MODULE IS THE METHOD ONLY. It touches no split, no draw and no reporting;
those are `scripts/protocol_adapter.py` (P1-P6). Keeping them apart is what
lets a reader check the method against the paper without also having to hold
our protocol in their head.

The whole architecture is four lines:

    P  = class means of the support features                       # S1, Eq.(5)
    P /= ||P||_2 along axis 0 ; P /= ||P||_2 along axis 1          # S2, Table 3
    p_proto = f @ P.T                                             # S3, Eq.(6)
    logits  = 100 * f @ W_text + alpha * p_proto                   # S4/U9, Eq.(7)

and the -F arm adds an ArcFace margin on `p_proto` alone during training
(S5/U4, Eq.(8)) while changing nothing at inference (S11).


==========================================================================
DESIGN NOTE -- the reading of record, clause by clause
==========================================================================

# Proto-Adapter — implementation from the paper

**Plan of record:** the README, *Comparative methods under our protocol*.
**Source:** Kato, Nota, Aoki, *"Proto-Adapter: Efficient Training-Free
CLIP-Adapter for Few-Shot Image Classification"*, Sensors 24(11):3624, 2024.
DOI 10.3390/s24113624 · PMC11175357. Full text archived at
`docs/proto-adapter.xml` (JATS XML from EuropePMC — the MDPI and PMC PDF
endpoints both refuse a scripted fetch; the XML is the publisher's own
typesetting source and carries every table verbatim).

**Why this file exists.** Tip-Adapter and ClusterAdapter were *ports*: we ran
their code and the diff was the deviation. Proto-Adapter has **no official
open-source repository**, so the whole implementation is a reading of prose,
and that section requires that every choice the paper leaves unstated be written
down as a choice rather than absorbed into the code. Sections A and B below
separate the two: **A is what the paper states**, **B is what it does not**.

---

## A. What the paper states

Equation and table numbers are the paper's.

| # | Clause | Where |
|---|---|---|
| **S1** | Prototype = mean of the class's training-image visual features. `c_n = (1/K) Σ_k CLIP_vis(I_{n,k})`. Adapter weights `F_proto = [c_1, …, c_N]`. | §2.2.1, Eq. (4)–(5) |
| **S2** | The prototypes are L2-normalized **twice**: channel-wise (first axis of `F_proto ∈ R^{N×D}`) **then** class-wise (second axis). This is the best of the four Table-3 settings and is worth **+2.05** at 16 shots on ImageNet. | §3.4, Table 3 |
| **S3** | Adapter logits are a bare matrix product, `p_proto = F_proto^T f`. **No activation function and no `β`** — this is the whole architectural difference from Tip-Adapter. | §2.2.1, Eq. (6); Fig. 3 |
| **S4** | Fusion is linear: `logits = α · p_proto + W_text^T f`. | Eq. (7) |
| **S5** | Fine-tuning uses the **ArcFace additive angular margin**: with `W_j^T f_i = cos θ_j`, the positive-class logit becomes `cos(θ_{y_i} + m)`; cross-entropy on those logits. | §2.2.2, Eq. (8) |
| **S6** | **Only the adapter weights `W = F_proto` are trained.** The encoders are frozen; `α` and `m` are searched, not learned. | §2.2.2, §3.1 |
| **S7** | Adam, **batch 256**, **20 epochs**, lr **4e-4 → 4e-5**, cosine annealing. | §3.1 |
| **S8** | Augmentation = random crop, resize, random horizontal flip, "consistent with Tip-Adapter". | §3.1 |
| **S9** | `m` and `α` are tuned **per dataset on the validation set**. | §3.1 |
| **S10** | Prompt **ensembling**, "templates for each dataset are the same as those used for Tip-Adapter" (7 for ImageNet). | §3.1, §3.2 |
| **S11** | The margin is applied **only during training**; inference is unchanged and costs no extra parameters. | §2.2.2 |
| **S12** | The `m` grid actually swept is `{0, 0.1, 0.2, 0.3, 0.4}`. | Table 4 |
| **S13** | Both arms exist: **Proto-Adapter** (training-free) and **Proto-Adapter-F** (fine-tuned). | §3.1 |

### S2 in coordinates, because the axis convention flips mid-paper

Eq. (4) declares `F_proto ∈ R^{D×N}`; §3.4 declares it `R^{N×D}`. The Table-3
wording — "channel-wise (along the first axis) followed by class-wise (along
the second axis)" — is only self-consistent in the **`N×D`** convention:
reducing over axis 0 (length `N`) gives one norm per *channel*, and reducing
over axis 1 (length `D`) gives one norm per *class*. So, with `P ∈ R^{N×D}`:

```
P ← P / ‖P‖₂ along axis 0     # channel-wise, one divisor per feature dim
P ← P / ‖P‖₂ along axis 1     # class-wise,   one divisor per class
```

The order matters and is not symmetric. The **class-wise step is last**, which
leaves every class weight on the unit sphere — exactly the precondition S5
needs for `W_j^T f = cos θ_j`. That agreement between two independent sections
is the strongest evidence the reading is right.

---

## B. What the paper does not state — our choices

Each is a real fork. The default is what `scripts/proto_adapter.py` does; the
alternative is implemented behind a flag so the choice can be *measured*, not
argued.

| # | Question | Default | Alternative | Why the default |
|---|---|---|---|---|
| **U1** | **The ArcFace scale `s`.** Never mentioned anywhere in the paper. | **Selected on VALIDATION — the mean val accuracy over the 11 calibration cells — then FROZEN globally**, one value for all datasets and all four backbones. | per-dataset `s`, or fitting `s` to Table 4 | ArcFace without a scale puts CE on logits in `[-1,1]`, where the softmax is nearly uniform and the gradient is tiny; with one, it is the standard face-recognition recipe. S9 says the paper tunes "other hyperparameters … using the validation set", so validation is the paper's own instrument for exactly this kind of constant. **Fitting `s` to Table 4 instead would be test-set selection** — Table 4 is a column of *test* accuracies on a split that overlaps ours, and choosing a hyperparameter to reproduce it would put a test-derived constant inside every number this method later reports (P3). So Table 4 is used the other way round: as an independent **check** on a scale chosen without it. Freezing globally rather than per dataset keeps it from becoming a hidden per-cell knob the paper does not have. |
| **U2** | Is each image feature L2-normalized **before** averaging in Eq. (5)? | **Yes.** | raw mean, then normalize (Tip-Adapter's `build_cache_model` convention) | The paper's own §2.1 defines `CLIP_vis(·)` as producing "`D`-dimensional L2 normalized visual features"; Eq. (5) reuses that symbol. |
| **U3** | Are prototypes built from **augmented-mean** features (Tip-Adapter's `augment_epoch = 10`) or one deterministic `preprocess` pass? | **Augmented-mean, 10 passes.** | 1 clean pass | S8 says the augmentation is Tip-Adapter's, and this makes the Proto-Adapter/Tip-Adapter comparison like-for-like on *features* so that the only difference is the architecture — which is the paper's claim. It is also what our P2 pool already is. |
| **U4** | Cross-entropy on **which** logits — the adapter's alone, or the fused `α·p + text`? | **The adapter's alone.** | fused (Tip-Adapter-F's convention) | §2.2.2 says "apply the cross-entropy loss to **the obtained logits**", and the only logits obtained in that section are Eq. (8)'s margin-modified `cos θ`. Neither `α` nor `W_text` appears anywhere in §2.2.2. **This is the sharpest divergence from Tip-Adapter-F and the most consequential guess in this file.** |
| **U5** | Is `W` re-normalized in the forward pass during and after training? | **Yes, class-wise, every forward — train and test.** | normalize at init only | S5 *defines* the logit as `cos θ_j`, which is false unless `‖W_j‖ = 1`; ArcFace itself normalizes `W` in the forward. Channel-wise normalization (S2) is therefore an **initialization-time** step only — there is no per-channel statistic to preserve once training starts. |
| **U6** | The **`α` grid**. Not stated. | **Ours** — `ALPHAS = logspace(1e-3, 10^(3+60/29), 40)`, per P4, with pinning reported. | Tip-Adapter's `search_scale[1]` grid | P4: an upstream arm keeps *its* grid, but there is no upstream grid here. Tip-Adapter's own α ceiling (13, or 3 on ImageNet) is far too low for this method — see U9 — so inheriting it would report a **bound**, not a value. |
| **U7** | The **`m` grid**. | `{0, 0.1, 0.2, 0.3, 0.4}` — S12, exactly the paper's own sweep. | — | Not really a choice; recorded because it is a grid we did not invent but did have to read off a table rather than a methods section. |
| **U8** | Are `m` and `α` searched **jointly**, and is `α` searched before or after fine-tuning? | **Jointly, after**: for each `m`, fine-tune, then search `α` on val; keep the `(m, α)` pair with the best **val** accuracy. | `α` fixed from the training-free arm | The literal reading of S9 ("we tune other hyperparameters, such as `m` and `α`, for each dataset using the validation set"). See the cost note below — the five margins share one augmentation stream, so this costs one training run, not five. |
| **U9** | The **zero-shot logit scale**. Eq. (7) writes `W_text^T f` with no scale; Tip-Adapter's code uses `100 · f^T W_text`. | **`100 ·`**, Tip-Adapter's convention. | no scale | Only the *ratio* to `α · p_proto` matters, so this is a reparameterization of the α grid — but keeping the `100` makes every α in this run directly comparable to the Run B / `tab:backbone_main` numbers, and makes U6's ceiling check meaningful. |
| **U10** | ImageNet support draw. Their protocol draws 16-shot from the full 1.28 M train split. | **Our 32/class subset** (`imagenet_subset`), the only ImageNet train data on disk. | — | A data limit, not a choice (§1.5, and the `capped-test-splits` record). 16 random images of 32 vs 16 of ~1300 are both random draws; it is recorded because it is a real difference in the *calibration* run, where everything else is supposed to be theirs. |
| **U11** | ImageNet validation set. Tip-Adapter's ImageNet code sets `val = test =` the 50 k val split, i.e. it searches hyperparameters on test. | **Never.** Our 10/class val, disjoint from the 40/class test. | — | P3. Applied in the calibration run too, so our ImageNet calibration number is expected to sit **below** theirs for a reason we can name. |

| **U12** | The ArcFace **monotonicity guard**. `cos(θ+m)` stops decreasing in `θ` once `θ+m > π`, so beyond that a *worse* prediction earns a *better* logit and the gradient flips sign. The paper writes Eq. (8) bare. | **The official ArcFace fallback** — use `cos θ − m·sin(π−m)` beyond the threshold. | "easy margin": fall back to `cos θ` | It is what the ArcFace release the paper cites actually does, and the paper says only "following ArcFace". It binds only on samples already wrong by more than `π − m`, a small minority at 16 shots. |

| **U13** | **Augmentation nondeterminism in the prototype.** U3 builds each prototype from 10 *random* augmented views, so `F_proto` is a random variable even with the support set held fixed. The paper says nothing about seeding, and neither does Tip-Adapter — `build_cache_model` has the identical property. | **Inherit it, then pin it.** The first encode of a given (dataset, backbone, K, protocol) is written to `caches/<ds>/proto_adapter/`, keyed by the support set's image paths, and every later run reads that file. | seed the augmentation stream | Claiming determinism we had not verified would be worse than measuring the spread. **Measured:** re-encoding ImageNet RN50 K=16 under a different worker-seeding regime moved Proto-Adapter by **0.07** (62.08 → 62.01). That is the size of the effect, stated rather than assumed away, and the cache makes every number downstream of it reproducible. |

### The cost note attached to U8

Searching five margins looks like five fine-tuning runs, i.e. 5× the
Tip-Adapter-F grid (§1.8: 4 h 41 m for 40 cells). It is not, and the reason is
exact rather than approximate: the expensive object is the **augmented image
batch pushed through the frozen encoder**, and it does not depend on `m`. If
the augmentation stream and the batch order are fixed by a seed, then training
five adapters side by side on the same batches is **identical, step for step,
to five independent runs on that stream** — only the loss head differs, and
the heads do not interact. So the five margins cost one encode.

`scripts/run_proto_adapter.py --prove-margin-sharing` re-runs a cell the slow
way, five separate trainings, and asserts bit-equality against the shared-stream
result. Same standard as the `search_hp` hoist in §1.9(b): **proven exact, not
argued.**

---

## C. The bar

There is no "Run A = their code, their protocol" here, so §3's two-run rule is
replaced by §1.11.1(1): implement, then check against **their published
numbers**, and if it cannot land near them, say so and stop.

**Calibration targets, RN50, 16-shot unless stated** (all from the paper):

| target | Proto-Adapter | Proto-Adapter-F |
|---|---|---|
| ImageNet (Table 1) | 63.89 | 66.17 |
| 11-dataset average (Table 2) | **72.69** | **75.59** |
| their Tip-Adapter / Tip-Adapter-F on the same rows | 70.32 | 75.81 |
| RN101 / ViT-B/32 / ViT-B/16 ImageNet (Table 5) | 66.81 / 67.07 / 72.25 | 69.12 / 69.27 / 74.23 |

**Table 3** (ImageNet, training-free, 1/4/16 shots) additionally pins the
*construction* independently of everything else:

| channel-wise | class-wise | 1 | 4 | 16 |
|---|---|---|---|---|
| — | — | 60.40 | 60.61 | 61.84 |
| ✓ | — | 60.65 | 60.96 | 61.75 |
| — | ✓ | 60.66 | 61.34 | 62.70 |
| ✓ | ✓ | **60.77** | **61.92** | **63.89** |

**Table 4** (ImageNet, 16-shot) pins `s` via U1: `w/o FT` 63.89, then
`m = 0 / .1 / .2 / .3 / .4` → 65.56 / 65.96 / **66.17** / 65.95 / 65.54.

The 10 non-ImageNet rows of Table 2 are the **cleanest** check, because U10 and
U11 do not touch them: our `testfull` *is* the standard CoOp test split those
numbers are on, and our val equals upstream's val wherever
`protocol_adapter.val_is_upstream_val` returns True.

---

## D. What this is NOT allowed to be

§1.11.6, verbatim: *"Do not start from our `CC-F-1` and patch it toward the
paper."* `CC-F-1` (our compact-cache m=1 trained arm) and `KM-1` (m=1
training-free) are architecturally sibling methods and are used **only as
cross-checks after the fact**. They differ from this implementation in at
least the loss (plain CE vs S5), the calibration rule, the normalization (S2 is
not what `KM-1` does), and the trainer (frozen features vs S7/S8 live
augmentation — the thing §1.4 measured at −4.60 on one cell and §1.8 at +2.30
grid-wide).

Carry in, too, the second finding named in §1.11.5: **prototype collapse is
lossy and the loss grows with K** (measured separately: prototype collapse is lossy and worsens with K — dtd
K=16 `tip − prototype` = **+4.08 ± 0.48**), which contradicts this paper's
claim that aggregation costs nothing. That is a prediction about what a
faithful implementation should show, and it is stated *before* the run.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# S12 / U7 -- the paper's own sweep, read off Table 4.
MARGINS = (0.0, 0.1, 0.2, 0.3, 0.4)

# S7 -- Adam, batch 256, 20 epochs, 4e-4 -> 4e-5 cosine.
TRAIN = dict(lr=4e-4, lr_min=4e-5, epochs=20, batch_size=256)

# U9 -- Tip-Adapter's zero-shot logit scale, kept so every alpha here is
# directly comparable to the Run B / tab:backbone_main numbers.
CLIP_LOGIT_SCALE = 100.0


# ---------------------------------------------------------------- S1, S2 ----
def class_means(feats, labels, n_cls):
    """Eq. (5): c_n = (1/K) sum_k CLIP_vis(I_{n,k}).

    `feats` are the per-image L2-normalized visual features -- U2. On the
    cached path those are `caches/<ds>/collapse/<tag>_pool_f.pt` rows, which
    are `normalize(mean_a encode(aug_a(img)))` over 10 augmentation passes,
    i.e. byte-for-byte what `utils.build_cache_model` puts in Tip-Adapter's
    cache (U3). So the two methods see the SAME features and differ only in
    the architecture, which is the paper's claim.

    Returns P [n_cls, D], fp32, un-normalized. No class may be empty: a
    missing class would silently become a zero row that then survives
    `normalize` as a zero (and wins nothing), rather than failing.
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
    second axis)". With P [N, D], reducing axis 0 (length N) gives one norm per
    FEATURE DIMENSION -> channel-wise; reducing axis 1 (length D) gives one norm
    per CLASS -> class-wise. The order is not symmetric and the paper's best row
    is both, in that order.

    The class-wise step is last, so every row leaves this function on the unit
    sphere -- which is exactly the precondition Eq. (8) needs for
    `W_j^T f = cos theta_j`. Two independent sections of the paper agreeing on
    that is the evidence the axis reading is right (design note, S2).

    `channel_wise`/`class_wise` are switchable ONLY so the four rows of Table 3
    can be reproduced as a calibration check. The method is both-on.
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

    U5 -- the weight is L2-normalized class-wise on EVERY forward, training and
    inference alike, because Eq. (8) defines the logit as `cos theta_j` and that
    is false unless `||W_j|| = 1`. ArcFace itself normalizes W in the forward.
    The channel-wise half of S2 is therefore an INITIALIZATION step only: once
    the rows start moving there is no per-channel statistic left to preserve.
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

    U12 -- MONOTONICITY GUARD. `cos(theta+m)` stops decreasing in theta once
    `theta + m > pi`, so past that point a WORSE prediction would get a BETTER
    logit and the gradient flips sign. The paper does not mention this; the
    official ArcFace release handles it by falling back to the linear extension
    `cos theta - m sin(pi - m)` beyond the threshold, and that is what we do.
    The alternative ("easy margin", fall back to `cos theta`) is the other
    common choice. It matters only for samples already on the wrong side of the
    boundary by more than `pi - m`, which at 16 shots is a small minority.

    `s` is U1: the paper never states a scale. It is calibrated ONCE against
    Table 4 and then frozen for every dataset and backbone.
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

    The alpha axis is evaluated in chunks for the same reason
    `run_baselines_e4.acc_batched` does it -- on imagenet one 40-alpha tensor
    of [10000, 1000] is 1.6 GB. argmax is per alpha, so chunking is exact.

    Ties go to the FIRST (smallest) alpha, matching `search_hp`'s strict `>`.
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
