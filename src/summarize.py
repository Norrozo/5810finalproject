"""Print a per-version summary table from results/results.csv (Markdown, ready for reports).

Usage:
    python -m src.summarize
    python -m src.summarize --versions B0 I1
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

METRICS = [("JF", "J&F ↑"), ("psnr", "PSNR ↑"), ("psnr_mask", "PSNR mask ↑"), ("ssim", "SSIM ↑"),
           ("flicker", "Flicker ↓"), ("spf_total", "s/frame ↓")]


def to_float(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--versions", nargs="*")
    args = ap.parse_args(argv)

    by_version = defaultdict(list)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            by_version[row["version"]].append(row)
    versions = args.versions or list(by_version)

    print("| Version | Clips | " + " | ".join(label for _, label in METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for v in versions:
        rows = by_version.get(v, [])
        cells = []
        for key, _ in METRICS:
            vals = [x for x in (to_float(r.get(key)) for r in rows) if x is not None]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "–")
        print(f"| {v} | {len(rows)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
