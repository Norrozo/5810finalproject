"""A short presentation video of what our iterations changed in the MASKS.

All clips play at once, one row per clip: the true mask (green) next to two versions' masks
(red), each labelled with its J&F for that clip from the results CSVs. Only existing run
outputs are used; nothing is re-run.

    python -m src.mask_video                     # writes runs/mask_story.mp4 (videos are not committed)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import list_frames, read_image, read_mask
from src.visualize import write_video

# (clip, results csv, versions shown next to the true mask); one row each in the video
STORY = [
    ("real_mbike-trick", "results/real_results.csv", ["B0", "I8"]),    # click: part vs whole
    ("real_loading", "results/real_results.csv", ["I2", "I2c"]),       # text: wrong look-alike, then ask
    ("real_india", "results/real_results.csv", ["I2c", "I8"]),         # text confidently wrong vs click
    ("multi_pigs", "results/multi_results.csv", ["I2c", "I7"]),        # "pigs": one vs all
    ("multi_india", "results/multi_results.csv", ["I2c", "I7"]),       # "women": one vs all
]
H = 216  # panel height (5 rows = 1080 px)


def overlay(img, mask, color):
    out = img.copy()
    out[mask] = (0.45 * out[mask] + 0.55 * np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return cv2.resize(out, (round(out.shape[1] * H / out.shape[0]), H))


def label(panel, text):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(panel, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return panel


def section(clip, csv, versions):
    """One row per frame: true mask, then each version's mask, labelled with its J&F."""
    scores = pd.read_csv(csv).set_index(["version", "clip"]).JF
    rows = []
    for f in list_frames(Path("data") / clip / "frames"):
        img = read_image(f)
        panels = [label(overlay(img, read_mask(Path("data") / clip / "gt_masks" / f.name), (0, 200, 0)), "True")]
        for v in versions:
            m = read_mask(Path("runs") / v / clip / "masks" / f.name)
            panels.append(label(overlay(img, m, (0, 0, 255)), f"{v}  J&F {scores[(v, clip)]:.2f}"))
        rows.append(np.hstack(panels))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/B0.yaml", help="unused; kept for the common CLI")
    ap.add_argument("--out", default="runs/mask_story.mp4")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--loops", type=int, default=3, help="play the clips this many times")
    args = ap.parse_args(argv)
    clips = [section(clip, csv, versions) for clip, csv, versions in STORY]
    n = max(len(c) for c in clips)
    width = max(c[0].shape[1] for c in clips)
    frames = []
    for t in range(n):  # all clips play at once, one per row; short clips hold their last frame
        rows = [c[min(t, len(c) - 1)] for c in clips]
        rows = [cv2.copyMakeBorder(r, 0, 2, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
        frames.append(np.vstack(rows))
    frames = frames * args.loops
    write_video(Path(args.out), frames, args.fps)
    print(f"wrote {args.out}: {len(frames)} frames, {len(frames) / args.fps:.0f} s")


if __name__ == "__main__":
    main()
