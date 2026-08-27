"""Rebuild tab:backbone_main from MEASURED cells only.

Refuses to emit until all 44 Run B cells (11 datasets x 4 backbones) exist.
Nothing here is projected, interpolated, or carried over from a previous
version of the table -- every number is recomputed from the evidence JSONs.

`+-` follows the project convention (the write-up §5b.6): the standard deviation,
over the 3 seeds, OF THE 11-DATASET MEAN -- paired within seed, not the spread
across datasets.

    PYTHONPATH=. python scripts/rebuild_backbone_main.py
"""
import glob
import json
import sys
from statistics import mean, pstdev

BB = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
DS = ["caltech101", "dtd", "eurosat", "fgvc", "food101", "imagenet",
      "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
SEEDS = [0, 1, 2]


def cells(path):
    d = json.load(open(path))
    if isinstance(d, list):
        return d
    return d.get("rows", d.get("cells", []))


def add(store, bb, ds, seed, v):
    if v is not None:
        store[(bb, ds, seed)] = v


def gather():
    R = {}

    # --- Tip-Adapter / Tip-Adapter-F: Run B, upstream implementation ---
    R["Tip-Adapter"], R["Tip-Adapter-F"] = {}, {}
    for p in glob.glob("evidence/runB/runB_*.json"):
        for r in cells(p):
            for arm, key in (("TIP", "Tip-Adapter"), ("TIP-F", "Tip-Adapter-F")):
                for s, v in zip(r.get("seeds", SEEDS), r["per_seed"][arm]):
                    add(R[key], r["backbone"], r["dataset"], s, v)

    # --- Proto-Adapter and Proto-Adapter-F: implemented from the paper -------
    #
    # RETIRED 2026-08-26: this row used to be our `CC-F-1`, read out of
    # evidence/calibrated_ft_e20_*_seeded.json. `CC-F-1` is NOT Proto-Adapter-F
    # (section 1.12): different loss (plain CE, not ArcFace), different
    # normalization (no channel-wise step, S2), different calibration rule, and
    # -- decisively -- our frozen-feature trainer, which section 1.8 measured at
    # -2.30 grid-wide against upstream's live-augmentation loop. Keeping it here
    # is what made section 1.10 flag the table as internally inconsistent: one
    # gradient row on upstream's trainer, the other on ours.
    #
    # Both rows now come from `scripts/run_proto_adapter.py`, implemented from
    # Sensors 24(11):3624 (reading of record:
    # the design note in scripts/proto_adapter.py) and calibrated against the paper's
    # own Tables 2-4 before being reported at all -- -0.11 training-free and
    # -0.43 trained, net of the zero-shot gap (sections 1.12, 1.12b).
    #
    # `impl.` stays "ours" for both: there is no upstream Proto-Adapter code to
    # run. That asymmetry against the "orig." Tip-Adapter rows is real and is
    # why the calibration step existed.
    R["Proto-Adapter"], R["Proto-Adapter-F"] = {}, {}
    for p in glob.glob("evidence/proto_adapter/ours_*.json"):
        for r in cells(p):
            if r.get("K") != 16:
                continue
            for arm in ("PROTO", "PROTO-F"):
                key = "Proto-Adapter" if arm == "PROTO" else "Proto-Adapter-F"
                for s, v in zip(r.get("seeds", SEEDS), r["per_seed"][arm]):
                    add(R[key], r["backbone"], r["dataset"], s, v)

    # --- GPCache: our implementation (no upstream code, section 4.4) ---
    R["GPCache"] = {}
    for p in glob.glob("evidence/baselines_e4*_testfull.json"):
        for c in cells(p):
            if c.get("K") != 16:
                continue
            add(R["GPCache"], c["backbone"], c["dataset"], c.get("seed"),
                c.get("acc", {}).get("gpcache16"))

    # --- GDA-CLIP: gda_last ---
    R["GDA-CLIP"] = {}
    for p in glob.glob("evidence/layer_ensemble_*_testfull.json"):
        for c in cells(p):
            if c.get("K") != 16:
                continue
            add(R["GDA-CLIP"], c["backbone"], c["dataset"], c.get("seed"),
                c.get("acc", {}).get("gda_last"))

    # --- LE-GDA: F1 ---
    R["LE-GDA"] = {}
    for g in ("evidence/nonlinear_fusion_full_*.json",
              "evidence/nonlinear_fusion_k14_*.json",
              "evidence/nonlinear_fusion_k28_*.json"):
        for p in glob.glob(g):
            for r in cells(p):
                if r.get("K") != 16 or "F1" not in r.get("rules", {}):
                    continue
                add(R["LE-GDA"], r["backbone"], r["dataset"], r.get("seed"),
                    r["rules"]["F1"]["test"])

    # --- zero-shot: seed-independent ---
    R["Zero-shot CLIP"] = {}
    for c in cells("evidence/calibrated_ft_e20_testfull.json"):
        if c.get("K") == 16:
            for s in SEEDS:
                add(R["Zero-shot CLIP"], c["backbone"], c["dataset"], s,
                    c.get("zs"))
    return R


def per_backbone(store):
    """(mean, sd) per backbone: sd over seeds of the 11-dataset mean."""
    out = []
    for bb in BB:
        per_seed = []
        for s in SEEDS:
            v = [store.get((bb, ds, s)) for ds in DS]
            if any(x is None for x in v):
                return None, [ds for ds, x in zip(DS, v) if x is None]
            per_seed.append(mean(v))
        out.append((mean(per_seed), pstdev(per_seed)))
    return out, None


ROWS = [("Zero-shot CLIP", "--", "--"),
        ("Tip-Adapter", "--", "orig."),
        ("Proto-Adapter", "--", "ours"),
        ("GPCache (W+C)", "--", "ours"),
        ("Tip-Adapter-F", r"$\checkmark$", "orig."),
        ("Proto-Adapter-F", r"$\checkmark$", "ours"),
        ("GDA-CLIP (final-feature)", "--", "ours"),
        ("LE-GDA", "--", "ours")]
KEY = {"Zero-shot CLIP": "Zero-shot CLIP", "Tip-Adapter": "Tip-Adapter",
       "Proto-Adapter": "Proto-Adapter",
       "GPCache (W+C)": "GPCache", "Tip-Adapter-F": "Tip-Adapter-F",
       "Proto-Adapter-F": "Proto-Adapter-F",
       "GDA-CLIP (final-feature)": "GDA-CLIP", "LE-GDA": "LE-GDA"}


def main():
    R = gather()
    n = len({(bb, ds) for (bb, ds, _) in R["Tip-Adapter-F"]})
    if n < 44:
        have = {(bb, ds) for (bb, ds, _) in R["Tip-Adapter-F"]}
        missing = [f"{ds}/{bb}" for ds in DS for bb in BB
                   if (bb, ds) not in have]
        print(f"REFUSING TO EMIT: Run B has {n}/44 cells.")
        print("missing: " + ", ".join(missing))
        sys.exit(1)

    res, bad = {}, {}
    for label, _, _ in ROWS:
        v, miss = per_backbone(R[KEY[label]])
        if v is None:
            bad[label] = miss
        else:
            res[label] = v

    print("MEASURED tab:backbone_main -- 11 datasets, K=16, 3 seeds\n")
    print(f"{'Method':26s}{'impl':7s}" + "".join(f"{b:>16s}" for b in BB)
          + f"{'Mean':>9s}")
    print("-" * 100)
    for label, grad, impl in ROWS:
        if label in bad:
            print(f"{label:26s}{impl:7s}  INCOMPLETE -- missing "
                  f"{sorted(set(bad[label]))[:4]}")
            continue
        v = res[label]
        print(f"{label:26s}{impl:7s}"
              + "".join(f"{m:11.2f}+-{s:<3.2f}" for m, s in v)
              + f"{mean(m for m, _ in v):9.2f}")

    if bad:
        print("\nSome rows incomplete; LaTeX not emitted.")
        return
    best = {i: max(range(len(ROWS)), key=lambda r: res[ROWS[r][0]][i][0])
            for i in range(4)}
    print("\n\n% ---- LaTeX ----")
    print(r"\begin{tabular}{lccrrrrr}")
    print(r"\toprule")
    print(r"Method & Grad. & Impl. & RN50 & RN101 & ViT-B/32 & ViT-B/16 & Mean\\")
    print(r"\midrule")
    for ri, (label, grad, impl) in enumerate(ROWS):
        v = res[label]
        cs = []
        for i, (m, s) in enumerate(v):
            c = f"{m:.2f}$\\pm$.{int(round(s*100)):02d}"
            cs.append(f"\\textbf{{{c}}}" if best[i] == ri else c)
        mu = mean(m for m, _ in v)
        nm = f"\\textbf{{{label}}}" if label == "LE-GDA" else label
        muf = f"\\textbf{{{mu:.2f}}}" if label == "LE-GDA" else f"{mu:.2f}"
        print(f"{nm} &{grad}& {impl} &" + "&".join(cs) + f"&{muf}\\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
