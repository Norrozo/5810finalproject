"""Do our fill metrics measure what we claim? Two checks, no pipeline needed.

1. Known cases. On every synthetic test clip we build fake "outputs" whose quality we
   know in advance and score them with the exact metric code evaluate.py uses
   (evaluate.fill_scores, true masks as the region):

     perfect      the clean video itself              -> every metric at its best
     not_removed  the input, object still there       -> every metric should be bad
     gray_box     region painted flat gray            -> bad
     blurred      clean video, blurred inside region  -> right colors, no detail
     shifted      clean video, moved 4 px in region   -> sharp and realistic, slightly misplaced
     flickering   clean video + a different brightness
                  offset in every frame (+/-12)       -> each frame fine, video flickers

   "blurred" vs "shifted" is the test for PSNR vs LPIPS: a blurry fill looks worse to a
   person than a sharp, slightly shifted one. "flickering" tests the temporal metrics:
   per-frame scores barely move, flicker must jump.

2. Agreement. Spearman rank correlation between metrics over every (version, clip) row
   of results/results.csv: do the metrics rank outputs the same way, or does each one
   add information?

    python -m src.metric_check          # writes results/metric_check.csv, results/metric_agreement.csv
Nothing is tuned here; the cases and their settings were fixed before looking at scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluate import fill_scores
from src.utils import clip_paths, list_frames, load_config, read_clip_list, read_image, read_mask

METRICS = ["psnr_mask", "ssim", "lpips_mask", "flicker", "flicker_raft"]
BETTER = {"psnr_mask": "higher", "ssim": "higher", "lpips_mask": "lower", "flicker": "lower", "flicker_raft": "lower"}


def fake_outputs(case: str, inputs, cleans, regions, rng) -> list[np.ndarray]:
    outs = []
    for t, (inp, clean, region) in enumerate(zip(inputs, cleans, regions)):
        if case == "perfect":
            out = clean.copy()
        elif case == "not_removed":
            out = inp.copy()
        else:
            out = clean.copy()
            if case == "gray_box":
                fill = np.full_like(clean, 128)
            elif case == "blurred":
                fill = cv2.GaussianBlur(clean, (0, 0), 8)
            elif case == "shifted":
                fill = np.roll(clean, (4, 4), axis=(0, 1))
            elif case == "flickering":
                fill = np.clip(clean.astype(np.int16) + int(rng.integers(-12, 13)), 0, 255).astype(np.uint8)
            else:
                raise ValueError(case)
            out[region] = fill[region]
        outs.append(out)
    return outs


CASES = ["perfect", "not_removed", "gray_box", "blurred", "shifted", "flickering"]


def known_cases(cfg: dict, clips: list[str]) -> pd.DataFrame:
    ev = cfg.get("evaluation", {})
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ev.get("region_dilation", 5) + 1,) * 2)
    rows = []
    for clip in clips:
        paths = clip_paths(cfg, clip)
        frames = list_frames(paths["frames"])
        inputs = [read_image(f) for f in frames]
        cleans = [read_image(paths["gt_clean"] / f.name) for f in frames]
        regions = [cv2.dilate(read_mask(paths["gt_masks"] / f.name).astype(np.uint8), kernel) > 0 for f in frames]
        for case in CASES:
            outs = fake_outputs(case, inputs, cleans, regions, np.random.default_rng(0))
            row = fill_scores(outs, cleans, regions, raft_cache=Path(cfg["data_root"]) / clip / "raft_flow.npy")
            rows.append({"case": case, "clip": clip, **{k: float(v) for k, v in row.items()}})
        print(f"  [metric_check] {clip}: done")
    return pd.DataFrame(rows)


def agreement(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv).dropna(subset=METRICS)
    # Flip "lower is better" metrics so +1 always means "rank outputs the same way"
    signed = pd.DataFrame({m: df[m] if BETTER[m] == "higher" else -df[m] for m in METRICS})
    return signed.corr(method="spearman").round(2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/B0.yaml", help="Only data paths and evaluation settings are used")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out_dir)

    cases = known_cases(cfg, [c["name"] for c in read_clip_list(cfg["eval_clips"])])
    cases.to_csv(out / "metric_check.csv", index=False)
    table = cases.groupby("case", sort=False)[METRICS].mean().round(3)
    print("\nKnown cases (mean over clips):\n" + table.to_string())

    agree = agreement(Path(cfg["results_csv"]))
    agree.to_csv(out / "metric_agreement.csv")
    print("\nSpearman agreement over results.csv rows (+1 = same ranking; lower-is-better metrics flipped):\n"
          + agree.to_string())


if __name__ == "__main__":
    main()
