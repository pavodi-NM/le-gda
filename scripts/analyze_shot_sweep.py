"""S1.13/S1.14 shot sweep -> the S1.14 results table and the two pre-registered
checks. ViT-B/16, K in {1,2,4,8,16}, 11 datasets, seeds [0,1,2].

    PYTHONPATH=. python scripts/analyze_shot_sweep.py

Reads exactly what the figure reads, through the figure's own loader, so the
table and the figure cannot disagree:
    Tip-Adapter / -F   evidence/runB/ + evidence/runB_shots/
    Proto-Adapter / -F evidence/proto_adapter/ours_ViT-B-16*.json
K=16 comes from the K=16 grids (S1.10, S1.12c); this run supplies K<16.
"""

import argparse
import glob
import importlib.util
import re
from statistics import mean

SHOTS = [1, 2, 4, 8, 16]
ARMS = ["Tip-Adapter", "Proto-Adapter", "Tip-Adapter-F", "Proto-Adapter-F"]
ORDER = ["imagenet", "caltech101", "oxford_pets", "stanford_cars",
         "oxford_flowers", "food101", "fgvc", "sun397", "dtd", "eurosat",
         "ucf101"]


def load(backbone="ViT-B/16"):
    spec = importlib.util.spec_from_file_location(
        "plg", "scripts/plot_le_gda_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.collect(backbone, False, "_testfull")[0]


def pinning():
    """E13 edge-pinning per K, straight out of the run logs' preflight block."""
    txt = open("logs/jobB_shots_b16.log").read()
    blocks = re.split(r"^=== K=(\d+)", txt, flags=re.M)
    out = {}
    for k, body in zip(blocks[1::2], blocks[2::2]):
        for arm, param, n, lo, hi in re.findall(
                r"^\s+(\S+)\s+(alpha|margin)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%",
                body, flags=re.M):
            out[(int(k), arm, param)] = (int(n), float(lo), float(hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ViT-B/16")
    args = ap.parse_args()
    data = load(args.backbone)
    have = [d for d in ORDER if d in data]

    print(f"\n== {args.backbone}: 11-dataset mean by K "
          "(a K is printed only where all 11 datasets have it) ==\n")
    print(f"{'method':<18}" + "".join(f"{f'K={k}':>9}" for k in SHOTS))
    curves = {}
    for a in ARMS:
        row, cur = "", {}
        for k in SHOTS:
            vs = [data[d][a][k] for d in have if k in data[d].get(a, {})]
            if len(vs) == len(have) and vs:
                cur[k] = mean(vs)
                row += f"{cur[k]:>9.2f}"
            else:
                row += f"{f'({len(vs)}/11)':>9}"
        curves[a] = cur
        print(f"{a:<18}{row}")

    print("\n== pre-registered check 1 -- E13 edge-pinning by K ==\n")
    pin = pinning()
    print(f"{'K':>4} {'arm':<10}{'param':<8}{'n':>4}{'lo%':>8}{'hi%':>8}")
    for (k, arm, param), (n, lo, hi) in sorted(pin.items()):
        print(f"{k:>4} {arm:<10}{param:<8}{n:>4}{lo:>8.1f}{hi:>8.1f}")
    print("\n  A cell at an edge is a BOUND, not a value. The margin grid"
          "\n  {0,...,0.4} is the paper's own (S12/U7) and is NOT widened,"
          "\n  so top-edge cells can only UNDERSTATE Proto-Adapter-F.")

    print("\n== pre-registered check 2 -- eurosat: Proto-Adapter-F - "
          "Tip-Adapter-F vs K ==\n")
    print(f"{'K':>4}{'PROTO-F':>10}{'TIP-F':>10}{'delta':>9}"
          f"{'delta (other 10)':>19}")
    for k in SHOTS:
        try:
            pf, tf = data["eurosat"]["Proto-Adapter-F"][k], \
                     data["eurosat"]["Tip-Adapter-F"][k]
            oth = [data[d]["Proto-Adapter-F"][k] - data[d]["Tip-Adapter-F"][k]
                   for d in have if d != "eurosat"
                   and k in data[d].get("Proto-Adapter-F", {})
                   and k in data[d].get("Tip-Adapter-F", {})]
        except KeyError:
            print(f"{k:>4}{'--':>10}{'--':>10}{'--':>9}{'--':>19}")
            continue
        o = f"{mean(oth):+.2f} (n={len(oth)})" if oth else "--"
        print(f"{k:>4}{pf:>10.2f}{tf:>10.2f}{pf - tf:>+9.2f}{o:>19}")
    print("\n  shrinking toward 0 as K falls => capacity effect "
          "(prototype collapse worsens with K)"
          "\n  flat in K                     => capacity is not the story, "
          "the S1.12c residual stays unexplained")
    print()


if __name__ == "__main__":
    main()
