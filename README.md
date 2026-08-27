# LE-GDA — Layer-Ensemble Gaussian Discriminant Analysis for training-free few-shot CLIP


LE-GDA fits the same shared-covariance Gaussian discriminant head independently in
several feature spaces of **one** CLIP forward pass — intermediate transformer
blocks and ResNet stages, not just the final projected embedding — standardises
each space's class scores per test sample, selects a subset on validation, and
fuses everything under a single searched `α` together with the zero-shot logit.

**Zero trainable parameters. One forward pass. No fine-tuning.**

The finding that motivates it is that the intermediate spaces are individually
*worse* — 4–7 points below the final embedding on ViT-CLS readouts — and are
still selected and still help. Standalone representation quality is a poor
predictor of what a space contributes to an ensemble.

---

## Results

K=16, 11 datasets, 3 seeds, standard CoOp test splits. Every number below was
produced by `scripts/make_le_gda_head_to_head.py` from the JSON files in
`evidence/`, and you can regenerate this table in about ten seconds with no GPU
(see [Reproducing without a GPU](#1-reproducing-the-tables-and-figures-no-gpu)).

**ViT-B/16**

| method | caltech | dtd | eurosat | fgvc | food101 | in1k | flowers | pets | cars | sun397 | ucf101 | **mean** | trained? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot CLIP | 93.27 | 44.03 | 48.49 | 24.81 | 85.88 | 67.84 | 70.81 | 89.10 | 65.55 | 62.56 | 67.41 | **65.43** | — |
| Tip-Adapter *(orig.)* | 94.86 | 65.54 | 77.87 | 40.46 | 86.36 | 69.47 | 93.90 | 91.74 | 75.52 | 71.71 | 78.26 | **76.88** | no |
| Proto-Adapter | 95.36 | 67.45 | 79.10 | 38.83 | 86.93 | 70.77 | 95.60 | 92.15 | 79.20 | 74.62 | 81.69 | **78.34** | no |
| GPCache (16 keys) | 95.36 | 69.94 | 83.26 | 43.91 | 86.94 | 70.58 | 96.91 | 92.40 | 78.59 | 74.38 | 81.32 | **79.42** | no |
| Tip-Adapter-F *(orig.)* | 95.81 | 72.16 | 87.56 | 45.75 | 87.30 | **71.98** | 96.52 | 92.96 | 82.76 | 76.17 | 83.79 | **81.16** | yes (20 ep) |
| Proto-Adapter-F | 95.66 | 70.41 | 81.15 | 45.42 | 87.12 | **72.19** | 97.28 | 92.80 | 82.78 | 76.16 | 83.35 | **80.39** | yes (20 ep) |
| GDA-CLIP (final-feature) | 95.10 | 72.52 | 87.53 | 48.91 | 87.16 | 71.44 | 97.36 | 93.73 | 83.01 | 75.77 | 82.84 | **81.40** | no |
| **LE-GDA (ours)** | 95.84 | 72.12 | 90.55 | 51.94 | 87.12 | 71.36 | 97.90 | 93.66 | 83.36 | 76.50 | 85.10 | **82.31** | no |

*(orig.)* marks rows produced by **the original authors' own code** under our
protocol — see [Comparative methods](#3-comparative-methods-under-our-protocol).

**All four backbones**, 11-dataset means, `±` one s.d. over 3 seeds:

| backbone | GPCache | GDA-CLIP | LE-GDA | LE-GDA − GDA-CLIP |
|---|---|---|---|---|
| ViT-B/16 | 79.42 ± 0.17 | 81.40 ± 0.08 | **82.31 ± 0.05** | **+0.92 ± 0.03** |
| ViT-B/32 | 75.61 ± 0.12 | 77.69 ± 0.11 | **78.69 ± 0.02** | **+1.00 ± 0.11** |
| RN50 | 72.81 ± 0.06 | 75.56 ± 0.15 | **77.47 ± 0.11** | **+1.91 ± 0.06** |
| RN101 | 75.26 ± 0.25 | 77.41 ± 0.22 | **79.68 ± 0.30** | **+2.26 ± 0.24** |

LE-GDA has the highest 11-dataset mean on all four backbones, ahead of both
20-epoch trained arms while training nothing, and `LE-GDA − GDA-CLIP` is
positive at every shot count K ∈ {1, 2, 4, 8, 16} on every backbone.

On ImageNet LE-GDA is behind the two gradient-trained adapters, while staying
ahead of the training-free ones on average and on both ResNets (GDA-CLIP edges
it there by 0.21 and 0.08 on ViT-B/32 and ViT-B/16). The paper has the full
per-dataset analysis; `scripts/make_per_dataset_table.py` prints the breakout
this repository generates it from.

**Read the increment honestly.** `LE-GDA − GDA-CLIP` is what this work adds.
The large jump from zero-shot CLIP to GDA-CLIP (≈ +16 points) belongs to the
Gaussian discriminant head, which is prior work, not to us. Our `GDA-CLIP` row
is also a *strengthened* baseline rather than a literal reproduction of the
published number: GDA-CLIP's Kubokawa–Srivastava estimator is a member of the
ridge family we search over (at `λ = D/(N−1)`), so our single-space baseline
contains theirs up to grid resolution.

---

## What is and is not in this repository

**In:** all the code, and the result JSONs (`evidence/`, 179 files) that
every table and figure in the paper is generated from. Nothing in the paper is
hand-copied from a terminal — the generators read these files and assert their
own consistency.

**Out:** the images and the extracted feature caches. Those are ~20 GB of
derived tensors plus the source datasets, and they are rebuilt rather than
shipped — `scripts/download_and_prepare.py` fetches the CLIP weights, checks the
dataset layout, and drives the whole extraction.

---

## Setup

```bash
git clone https://github.com/pavodi-NM/le-gda.git
cd le-gda

conda create -n legda python=3.10
conda activate legda
pip install torch torchvision          # match your CUDA build
pip install -r requirements.txt
```

Tested on Python 3.10 with one RTX 4060 (8 GB). Every runner fits in 8 GB;
ImageNet is the only cell that ever came close.

### Data

Point `LEGDA_DATA_ROOT` at wherever the images live (default `./data/tip-adapter`):

```bash
export LEGDA_DATA_ROOT=/path/to/data

python scripts/download_and_prepare.py --weights   # 4 CLIP checkpoints
python scripts/download_and_prepare.py --check     # which datasets are ready
```

`--check` prints a per-dataset OK/MISS line and, for anything missing, where to
get it. The 11 datasets use **CoOp's** directory names and **CoOp's**
`split_zhou_*.json` splits — the same splits every baseline here is fitted on.
Follow [CoOp's DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)
for the download links, with two exceptions this repo handles for you:

```bash
# SUN397 — the official Princeton hosts are dead. Rebuilds the exact original
# tree from a HF mirror, so split_zhou_SUN397.json still applies (~39 GB).
python scripts/reconstruct_sun397_from_hf.py

# ImageNet — a 2-3 GB subset from a public mirror, no ImageNet account needed.
python scripts/fetch_imagenet_subset.py
```

> **ImageNet caveat, stated up front.** ImageNet here is the official 50k
> validation set, split into 40/class test + 10/class val, plus 32/class of
> train as the few-shot draw pool. That is a data limit, not a code one: it is
> the same split every arm in the table is evaluated on, so the comparison is
> fair, but the ImageNet column is not a full-test-set number.

### Extraction

```bash
python scripts/download_and_prepare.py --extract
```

This writes `caches/`. Measured on one RTX 4060: base projected features
~35 min, per-layer / per-stage features ~50 min, ~20 GB of disk. It is
resumable — existing files are skipped, so rerun the same command after an
interruption. To do one backbone or dataset at a time:

```bash
python scripts/download_and_prepare.py --extract \
    --backbones ViT-B/16 --datasets dtd eurosat
```

---

## Reproducing

Set `PYTHONPATH=.` (or use `python -m scripts.<name>`) for every command below.

> A note on the comments: the runners and scorers cite `§` section numbers and
> named kill conditions (`KC-S1`, `KC-N2`, ...). Those point into the paper and
> its supplement, which state what each gate was and what it decided. The code
> asserts them; the paper explains them.

```bash
TEN="caltech101 dtd eurosat fgvc food101 oxford_flowers \
     oxford_pets stanford_cars sun397 ucf101"
```

### 1. Reproducing the tables and figures (no GPU)

Everything in this section reads `evidence/` only. No caches, no datasets, no GPU.

```bash
# the head-to-head tables printed above
PYTHONPATH=. python scripts/make_le_gda_head_to_head.py --suffix _testfull

# the 396-cell verdict and its kill conditions
PYTHONPATH=. python scripts/analyze_layer_ensemble.py --file \
    evidence/layer_ensemble_{vitb16,vitb32,imagenet_b16,imagenet_b32}_testfull.json \
    evidence/layer_ensemble_{RN50,RN50_imagenet,RN101,RN101_imagenet}_testfull.json

# accuracy vs. labelled examples per class, all 7 methods, K in {0,1,2,4,8,16}
# -> figures/le_gda/*.pdf (+ PNG previews), 12 figures
PYTHONPATH=. python scripts/plot_le_gda_figures.py --strict --suffix _testfull

# which spaces get selected, and how badly they score alone
PYTHONPATH=. python scripts/analyze_f1_selection.py
PYTHONPATH=. python scripts/plot_le_gda_selection.py --strict

# the fusion rule against the test-oracle linear ceiling
PYTHONPATH=. python scripts/analyze_nonlinear_fusion.py \
    --file evidence/nonlinear_fusion_full_*.json

# per-backbone main table (LaTeX). Refuses to emit unless all 44 cells of the
# upstream Tip-Adapter run exist -- nothing is projected or carried over.
PYTHONPATH=. python scripts/rebuild_backbone_main.py

# the same table broken out per dataset: 11 x 8 methods x 4 backbones.
# Prints to stdout; this repository keeps a single .md file.
PYTHONPATH=. python scripts/make_per_dataset_table.py

# LE-GDA vs GDA-CLIP across backbones, and the shot sweep with both checks
PYTHONPATH=. python scripts/plot_le_gda_vs_gda.py
PYTHONPATH=. python scripts/analyze_shot_sweep.py
```

One generator is not in this list: `scripts/measure_le_gda_storage.py` reads the
per-space feature dimensions off `caches/`, so it needs the extraction step
below (though still no GPU).

`--strict` is not decoration: the figure and table generators refuse to render
if a method is missing a shot count, if two sources disagree about zero-shot
accuracy, or if the evidence disagrees with the published selection table. A
silent fallback is how a figure goes stale, so there isn't one.

### 2. Reproducing the numbers from the features (GPU)

Run these after `--extract`. Each writes the `evidence/*.json` file that
section 1 consumes, so you can overwrite ours and rerun the generators.

```bash
# --- the LE-GDA sweep: 11 datasets x 4 backbones x 3 seeds x K in {1,4,16}
PYTHONPATH=. python scripts/run_layer_ensemble.py --backbones ViT-B/16 \
    --datasets $TEN --test-split testfull \
    --out evidence/layer_ensemble_vitb16_testfull.json
PYTHONPATH=. python scripts/run_layer_ensemble.py --backbones ViT-B/32 \
    --datasets $TEN --test-split testfull \
    --out evidence/layer_ensemble_vitb32_testfull.json
# ResNets have no `pre` space and their stages are R1..R4, so pass --spaces
for BB in RN50 RN101; do
  PYTHONPATH=. python scripts/run_layer_ensemble.py --backbones $BB \
      --datasets $TEN --spaces R1 R2 R3 R4 last --test-split testfull \
      --out evidence/layer_ensemble_${BB}_testfull.json
done
# imagenet runs LAST, separately, and always on --test-split test
PYTHONPATH=. python scripts/run_layer_ensemble.py --backbones ViT-B/16 \
    --datasets imagenet --test-split test \
    --out evidence/layer_ensemble_imagenet_b16_testfull.json

# --- per-sample standardised fusion: the deployed LE-GDA row
PYTHONPATH=. python -m scripts.run_nonlinear_fusion --datasets $TEN \
    --backbones ViT-B/16 --out evidence/nonlinear_fusion_full_ViT-B-16.json
PYTHONPATH=. python -m scripts.run_nonlinear_fusion --datasets $TEN \
    --backbones RN50 --spaces R1 R2 R3 R4 last \
    --out evidence/nonlinear_fusion_full_RN50.json
PYTHONPATH=. python -m scripts.run_nonlinear_fusion --datasets imagenet \
    --backbones RN50 --spaces R1 R2 R3 R4 last --test-split test \
    --out evidence/nonlinear_fusion_full_in_RN50.json
PYTHONPATH=. python -m scripts.analyze_nonlinear_fusion \
    --file evidence/nonlinear_fusion_full_*.json
```

**Cost, measured on one RTX 4060 (8 GB):** the LE-GDA sweep is ~9 min per
backbone over the ten datasets; ImageNet's ResNet cells dominate what is left.
The trained baseline arms are the long pole at roughly 2 h.

**Filenames are load-bearing.** The generators glob for exact names —
`layer_ensemble_{vitb16,vitb32,imagenet_b16,imagenet_b32,RN50,RN50_imagenet,RN101,RN101_imagenet}_testfull.json`,
`nonlinear_fusion_full_*.json` (K=16), `nonlinear_fusion_k14_*.json` (K ∈ {1,4}),
`nonlinear_fusion_k28_*.json` (K ∈ {2,8}) — and assert that no cell is counted
twice. Write to a name outside that scheme and the table will silently lack the
cell; write to the wrong split's name and the generator raises.

### 3. Comparative methods under our protocol

The baselines are **re-run here rather than quoted**, on the identical features,
splits, seeds and support sets as LE-GDA, so every row of the head-to-head is
comparable cell by cell.

Two of them run **the original authors' own code**. That distinction is the
`Impl.` column of the main table and it is not cosmetic — an earlier version of
this repository scored Tip-Adapter-F with our own frozen-feature trainer and
**understated it by 2.30 points**.

| row | arm | `Impl.` | runner |
|---|---|---|---|
| Tip-Adapter | `TIP` | **orig.** | `main.py` (upstream Tip-Adapter, patched) |
| Tip-Adapter-F (20 ep) | `TIP-F` | **orig.** | `main.py` |
| Proto-Adapter | `PROTO` | ours | `run_proto_adapter.py` |
| Proto-Adapter-F (20 ep) | `PROTO-F` | ours | `run_proto_adapter.py` |
| GPCache (16 keys) | `gpcache16` | ours | `run_baselines_e4.py` |
| GDA-CLIP (final-feature) | `gda_last` | ours | `run_layer_ensemble.py` (the 1-member ensemble) |

#### What "our protocol" means

`scripts/protocol_adapter.py` defines six clauses, `P1`–`P6`, and is the only
thing an upstream implementation is made to obey. Nothing else about it is
touched — not the model, the loss, the optimiser, the schedule, the batch size,
or the augmentation pipeline:

- **P1** our validation split and the standard uncapped test split
- **P2** our seeded K-shot draw from the same cached support pool every other
  row is fitted on, guarded by a byte-identical label-order assert
- **P3** no test-set model selection of any kind
- **P4** an upstream arm keeps **its own** hyperparameter grid
- **P5** the shared zero-shot head
- **P6** three seeds, `[0, 1, 2]`

`P4` is worth spelling out: the Tip-Adapter arms keep upstream's own α/β search
grid, which is edge-pinned on a minority of cells, so those two rows are lower
bounds. The grid is not widened — widening an upstream arm's grid stops it being
that arm.

`patches/` carries the exact diff applied to upstream Tip-Adapter, so the
deviation is auditable rather than described:

| patch | what it does |
|---|---|
| `tip_adapter_our_protocol.patch` | P1/P2/P3/P6 — our splits, our seeded draw, deletion of upstream's per-epoch test-set checkpoint selection, and `--shots`/`--backbone`/`--seeds`/`--test-split`/`--out` |
| `tip_adapter_searchhp_hoist.patch` | loop-invariant hoisting in `search_hp`, **numerically exact** (verified bit-identical on a completed cell); required only because ImageNet is otherwise ~a day per call |

**Proto-Adapter has no public code at all.** It is implemented from
Kato, Nota & Aoki, *Sensors* 24(11):3624 (2024). Every clause the paper states
(`S1`–`S13`) and every choice it leaves unstated (`U1`–`U12`) is written down in
the design note in the docstring of `scripts/proto_adapter.py`, and both arms
were calibrated against the paper's own Tables 2–4 before being reported at all
(−0.11 training-free, −0.43 trained, net of the zero-shot gap). The ArcFace
scale `s = 64` is selected once on validation and then frozen; it is never
re-selected per shot count.

```bash
BBS="RN50 RN101 ViT-B/32 ViT-B/16"

# --- Tip-Adapter and Tip-Adapter-F: UPSTREAM code, our protocol.
# One invocation per (dataset, backbone); 3 seeds inside.
for ds in caltech101 dtd eurosat fgvc food101 oxford_flowers \
          oxford_pets stanford_cars sun397 ucf101 imagenet; do
  PYTHONPATH=. python main.py --config configs/$ds.yaml \
      --backbone ViT-B/16 --seeds 0 1 2 --test-split testfull \
      --out evidence/runB/runB_${ds}_ViT-B-16.json
done

# --- Proto-Adapter and Proto-Adapter-F, from the paper. s=64 is frozen.
PYTHONPATH=. python scripts/run_proto_adapter.py \
    --protocol ours --arms free ft --scale 64 --seeds 0 1 2 \
    --backbones ViT-B/16 --k 16 \
    --out evidence/proto_adapter/ours_ViT-B-16.json

# --- the K in {1,2,4,8} sweep behind the shot-sweep figures
bash scripts/run_jobA_shots.sh     # Tip-Adapter arms  -> evidence/runB_shots/
bash scripts/run_jobB_shots.sh     # Proto-Adapter arms -> evidence/proto_adapter/

# --- GPCache and the cache-model baselines
PYTHONPATH=. python scripts/run_baselines_e4.py --datasets $TEN \
    --backbones $BBS --test-split testfull \
    --out evidence/baselines_e4_testfull.json
```

GDA-CLIP needs no separate run: `analyze_layer_ensemble.py` asserts that the
1-member ensemble at `last` equals the `gda_last` arm exactly (`max|Δ| = 0`), so
the baseline is nested inside our own sweep and cannot drift from it.

`run_calibrated_ft.py` is still shipped, but it **no longer feeds any
comparative row.** Its `TIP`/`TIP-F`/`CC-F-1` arms were our own
reimplementations; `CC-F-1` in particular is *not* Proto-Adapter-F (plain
cross-entropy instead of ArcFace, no channel-wise normalisation, a different
calibration rule). It remains only as the generator for our own cache-compression
arms.

### 4. Ablation — leave-one-out configuration transfer

Does a configuration selected on other datasets transfer, or does validation
access matter? Run against the submitted fusion operator (`F1`):

```bash
# preflight on the shipped evidence — no GPU, seconds
PYTHONPATH=. python -m scripts.analyze_config_transfer_preflight \
    --derive evidence/config_transfer_derive_f1.json

# ~15 min derive + ~30 min eval on a 4060
PYTHONPATH=. python -m scripts.run_config_transfer --phase derive --fusion F1 \
    --ks 16 --derived evidence/config_transfer_derive_f1.json
PYTHONPATH=. python -m scripts.run_config_transfer --phase eval --fusion F1 \
    --ks 16 --derived evidence/config_transfer_derive_f1.json \
    --out evidence/config_transfer_f1.json

PYTHONPATH=. python -m scripts.analyze_config_transfer \
    --file evidence/config_transfer_f1.json \
    --derive evidence/config_transfer_derive_f1.json --k 16
```

Scoring unit is the (backbone, dataset) pair — 44 units, not 132 cells; seeds of
the same cell are not independent and scoring them separately triples `n` for
free. `--unit cell` reverts to 132-cell scoring for diagnosis only.

---

## Repository map

```
scripts/
  download_and_prepare.py       weights + dataset check + extraction driver
  fetch_imagenet_subset.py      2-3 GB ImageNet subset from a public mirror
  reconstruct_sun397_from_hf.py rebuilds SUN397 (official hosts are dead)
  extract_backbone_all.py       base projected features, all backbones
  extract_layers.py             per-block ViT CLS features (+ ln_post via --with-pre)
  extract_layers_rn.py          per-stage ResNet features (GAP of layer1..4)
  extract_imagenet_subset.py    ImageNet base features (no CoOp builder entry)
  run_layer_ensemble.py         the LE-GDA sweep; also emits the GDA-CLIP arm
  run_nonlinear_fusion.py       per-sample standardised fusion (the deployed rule)
  run_baselines_e4.py           GPCache and the cache-model baselines
  protocol_adapter.py           P1-P6: the only clauses an upstream arm obeys
  proto_adapter.py              Proto-Adapter from the paper + the design note
  run_proto_adapter.py          Proto-Adapter / Proto-Adapter-F runner
  run_jobA_shots.sh             Tip-Adapter arms, K in {1,2,4,8}
  run_jobB_shots.sh             Proto-Adapter arms, K in {1,2,4,8}
  analyze_shot_sweep.py         the shot sweep + its two pre-registered checks
  rebuild_backbone_main.py      per-backbone main table, measured cells only
  make_per_dataset_table.py     the same table broken out per dataset
  run_calibrated_ft.py          our cache-compression arms (NOT the baselines)
  run_config_transfer.py        leave-one-out configuration-transfer ablation
  run_weighting_ceiling.py      test-oracle ceiling for any linear combiner
  analyze_*.py                  scorers: gates, fusion, selection, ablation
  make_*.py / plot_*.py         every paper table and figure, from evidence/
  preflight.py                  instrument checks each runner asserts before scoring
main.py      upstream Tip-Adapter, patched to obey P1-P6 (see patches/)
patches/     the exact diff applied to upstream Tip-Adapter
clip/        CLIP model + tokenizer (from openai/CLIP)
datasets/    CoOp-style dataset builders and splits
configs/     per-dataset YAML (root_path, hyperparameter search grid)
evidence/    result JSONs — the source of every published number
  runB/          Tip-Adapter + Tip-Adapter-F, upstream code, K=16
  runB_shots/    the same, K in {1,2,4,8}
  proto_adapter/ Proto-Adapter + Proto-Adapter-F, all K
utils.py     shared helpers
LICENSE      MIT
```

Feature caches land in `caches/`, figures in `figures/le_gda/`; both are
git-ignored because both are regenerated.

---

## Notes on the measurement protocol

A few conventions are worth knowing before you compare against these numbers:

- **Splits.** All 11 datasets use CoOp's standard test splits. Files carrying
  the `_testfull` token are on those splits, and every cell records its own
  `test_split` so the generators can assert provenance when they pool files.
  ImageNet is the exception described above and always reports `test`.
- **Seeds.** 3 seeds; a seed redraws the K-shot support set. `±` in the
  cross-backbone table is run-to-run variability of the whole 11-dataset mean,
  not the spread across datasets. With n=3 it is a crude estimate.
- **Batch size** is 128 throughout.
- **The matched null.** GDA is affine-equivariant, so random projections and
  invertible reparameterisations return identical logits and control for
  nothing. The control used here is `null_ridge`: the identical greedy selection
  and `α` search over seven *information-equivalent* candidates (the final layer
  at seven ridge strengths). It buys **+0.000**, which is what makes the LE-GDA
  increment mean anything.
- **Storage.** LE-GDA never deploys `Σ⁻¹`; the per-space cost is class means
  plus a whitening transform. It is 3–5× GDA-CLIP and smaller than Tip-Adapter
  on all 132 measured cells. `scripts/measure_le_gda_storage.py` recomputes it.

---

## License

MIT — see [LICENSE](LICENSE). The fork lineage (Tip-Adapter, CLIP, CoOp) is MIT
throughout; the datasets keep their own terms.

## Acknowledgements

This code base began as a fork of
[Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) and inherits its
dataset builders and CoOp-style splits. It benefits from
[CLIP](https://github.com/openai/CLIP),
[CoOp](https://github.com/KaiyangZhou/CoOp),
[CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter), and the GDA-CLIP
line of work on training-free discriminant heads for CLIP. Thanks to all of
them.

## Citation

```bibtex
@misc{legda2026,
  title  = {LE-GDA: Layer-Ensemble Gaussian Discriminant Analysis for
            Training-Free Few-Shot CLIP},
  author = {Pavodi Ndoyi Maniamfu, Vogel Kiketa, Kuetu Kuzulu Zezina, Jaking Mukendi Kabongo, David Mayoya Kutangila, Mampuya Kinkani Pescie Honore, Sieka Wolo L'Or},
  year   = {2026}
}
```




















"""
