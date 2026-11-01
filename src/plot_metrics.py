"""Bar charts of every metric, one panel per metric, one bar per version.

Reads only the result CSVs (nothing is re-measured) and writes PNGs to results/figures/:

  scorecard.png       the five plain-language numbers (see src/scorecard.py)
  masks.png           J (region overlap), F (outline accuracy), J&F, on each test/dev set
                      (last row: plural prompts like "pigs", I2c = one object vs I7 = all)
  fill_end_to_end.png PSNR, PSNR mask, SSIM, LPIPS mask, flicker (Farneback and RAFT),
                      full pipeline on the pasted benchmark
  fill_only.png       the same fill metrics with every method given the TRUE masks
  metric_check.png    the fill metrics on fake outputs of known quality (src/metric_check.py)

Each panel title says whether higher or lower is better. Flicker panels draw the clean
video's own flicker as a dashed line: that is the floor a perfect fill would reach.

    python -m src.plot_metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.scorecard import fill_score, read  # noqa: E402

FILL_VERSIONS = ["B0", "I1", "I3", "I3b", "I4"]
MASK_VERSIONS = ["B0", "I8", "I2", "I2b", "I2c", "I5", "I5b", "I7"]
FILL_METRICS = [("psnr", "PSNR (whole frame)", "higher"), ("psnr_mask", "PSNR mask", "higher"),
                ("ssim", "SSIM", "higher"), ("lpips_mask", "LPIPS mask", "lower"),
                ("flicker", "Flicker (Farneback)", "lower"), ("flicker_raft", "Flicker (RAFT)", "lower")]
CLICK, TEXT, FILL = "#4C72B0", "#DD8452", "#55A868"


def bars(ax, labels, values, title, better, color=FILL, ref=None, fmt="{:.2f}"):
    colors = color if isinstance(color, list) else [color] * len(values)
    b = ax.bar(labels, values, color=colors)
    ax.bar_label(b, [fmt.format(v) for v in values], fontsize=8, padding=2)
    ax.set_title(f"{title}  ({'↑' if better == 'higher' else '↓'} {better} is better)", fontsize=10)
    if ref is not None:
        ax.axhline(ref, ls="--", color="gray", lw=1)
        ax.text(-0.45, ref, "clean video", va="bottom", ha="left", fontsize=8, color="gray")
    ax.margins(y=0.15)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def means(df, versions, cols):
    d = df[df.version.isin(versions)].groupby("version")[cols].mean()
    return d.reindex([v for v in versions if v in d.index])


def plot_fill(df, versions, title, out):
    m = means(df, versions, [c for c, _, _ in FILL_METRICS] + ["flicker_ref", "flicker_raft_ref"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, name, better) in zip(axes.flat, FILL_METRICS):
        ref = m[f"{col}_ref"].mean() if col.startswith("flicker") else None
        bars(ax, m.index, m[col], name, better, ref=ref)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_masks(sets, out):
    fig, axes = plt.subplots(len(sets), 3, figsize=(14, 3.3 * len(sets)))
    for row, (set_name, df) in zip(axes, sets):
        m = means(df, MASK_VERSIONS, ["J", "F", "JF"])
        colors = [CLICK if v in ("B0", "I8") else TEXT for v in m.index]
        for ax, (col, name) in zip(row, [("J", "J: region overlap"), ("F", "F: outline accuracy"),
                                         ("JF", "J&F: mean of the two")]):
            bars(ax, m.index, m[col], f"{set_name}: {name}", "higher", color=colors)
            ax.set_ylim(0, 1.1)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CLICK), plt.Rectangle((0, 0), 1, 1, color=TEXT)],
               labels=["click prompt", "text prompt"], loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_scorecard(syn, real, check, out):
    syn = syn.assign(fill=fill_score(syn, check), steady=syn.flicker_raft / syn.flicker_raft_ref)
    s = means(syn, FILL_VERSIONS, ["fill", "steady", "spf_total"])
    r = means(real, MASK_VERSIONS, ["JF"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    bars(axes[0], s.index, s.fill, "Fill score /100", "higher", fmt="{:.0f}")
    bars(axes[1], s.index, s.steady, "Steadiness (1.0x = real video)", "lower", ref=1.0, fmt="{:.1f}x")
    bars(axes[2], s.index, s.spf_total, "Speed, s/frame", "lower", color="#8172B3")
    bars(axes[3], r.index, 100 * r.JF, "Real-video mask accuracy %", "higher", fmt="{:.0f}%",
         color=[CLICK if v in ("B0", "I8") else TEXT for v in r.index])
    fig.suptitle("Scorecard (first three: pasted benchmark, full pipeline)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_check(check, out):
    m = check.groupby("case", sort=False)[[c for c, _, _ in FILL_METRICS if c != "psnr"]].mean()
    ref = check.groupby("case", sort=False)["flicker_raft_ref"].mean().mean()
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (col, name, better) in zip(axes, [f for f in FILL_METRICS if f[0] != "psnr"]):
        bars(ax, [c.replace("_", "\n") for c in m.index], m[col], name, better, color="#937860",
             ref=ref if col == "flicker_raft" else None)
    fig.suptitle("Metric check: fake outputs of known quality (which metric catches which problem?)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/B0.yaml", help="unused; kept for the common CLI")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    syn, real, check = read("results/results.csv"), read("results/real_results.csv"), read("results/metric_check.csv")
    real_dev = read("results/real_dev_results.csv")
    if not (real_dev.version == "B0").any():  # same fallback as src/stage_report.py
        tuning = read("results/tuning.csv")
        real_dev = pd.concat([real_dev, tuning[(tuning.version == "B0_hf_model-sam2.1-hiera-large")
                                               & tuning["clip"].str.startswith("rdev_")].assign(version="B0")])

    plot_scorecard(syn, real, check, out / "scorecard.png")
    plot_masks([("Pasted test", syn), ("Real test", real), ("Real dev", real_dev),
                ("Several objects, test", read("results/multi_results.csv"))], out / "masks.png")
    plot_fill(syn, FILL_VERSIONS, "Fill, full pipeline (pasted benchmark, 10 clips)", out / "fill_end_to_end.png")
    plot_fill(read("results/fill_results.csv"), FILL_VERSIONS,
              "Fill only: every method given the TRUE masks (pasted benchmark, 10 clips)", out / "fill_only.png")
    plot_check(check, out / "metric_check.png")
    print("wrote", ", ".join(p.name for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
