"""A short presentation video of what our iterations changed in the MASKS.

For a few clips, plays the true mask (green) next to two versions' masks (red), with each
panel's J&F for that clip read from the results CSVs. A title card before each clip says
what failed and what fixed it. Only existing run outputs are used; nothing is re-run.

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

# (title, explanation, clip, results csv, [(version, label), ...])
STORY = [
    ("1. A click is ambiguous: part or whole?",
     "B0 keeps SAM 2's most confident mask: often a part (the helmet).\nI8 keeps the largest of its 3 masks: the whole rider.",
     "real_mbike-trick", "results/real_results.csv", [("B0", "B0 click"), ("I8", "I8 click")]),
    ("2. Text prompts pick the wrong look-alike",
     "'man in black jacket': the detector's top box is the other man.\nI2c sees two close scores and asks which one.",
     "real_loading", "results/real_results.csv", [("I2", "I2 text"), ("I2c", "I2c asks")]),
    ("3. Still unsolved: confidently wrong text",
     "'woman in white': one confident wrong box, so I2c does not ask.\nA click still gets it right.",
     "real_india", "results/real_results.csv", [("I2c", "I2c text"), ("I8", "I8 click")]),
    ("4. Several objects: 'remove the pigs'",
     "Before, only the top box was kept: one pig.\nI7 keeps every box above 0.45 and tracks each.",
     "multi_pigs", "results/multi_results.csv", [("I2c", "I2c (one)"), ("I7", "I7 (all)")]),
    ("5. Several objects in a crowd: 'women'",
     "I7 finds all 3 named women among an unlabelled crowd.",
     "multi_india", "results/multi_results.csv", [("I2c", "I2c (one)"), ("I7", "I7 (all)")]),
]
H = 300  # panel height


def overlay(img, mask, color):
    out = img.copy()
    out[mask] = (0.45 * out[mask] + 0.55 * np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return cv2.resize(out, (round(out.shape[1] * H / out.shape[0]), H))


def label(panel, text):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(panel, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return panel


def title_card(width, title, text):
    card = np.full((H + 40, width, 3), 24, np.uint8)
    cv2.putText(card, title, (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(card, line, (40, 170 + 42 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
    cv2.putText(card, "green = true mask (DAVIS)   red = our mask   number = J&F for this clip",
                (40, H + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    return card


def section(clip, csv, versions, title, text, fps):
    scores = pd.read_csv(csv).set_index(["version", "clip"]).JF
    frames = list_frames(Path("data") / clip / "frames")
    rows = []
    for f in frames:
        img = read_image(f)
        panels = [label(overlay(img, read_mask(Path("data") / clip / "gt_masks" / f.name), (0, 200, 0)), "True mask")]
        for v, name in versions:
            m = read_mask(Path("runs") / v / clip / "masks" / f.name)
            panels.append(label(overlay(img, m, (0, 0, 255)), f"{name}: J&F {scores[(v, clip)]:.2f}"))
        row = np.hstack(panels)
        rows.append(np.vstack([row, np.full((40, row.shape[1], 3), 24, np.uint8)]))
    card = title_card(rows[0].shape[1], title, text)
    return [card] * int(fps * 4) + rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/B0.yaml", help="unused; kept for the common CLI")
    ap.add_argument("--out", default="runs/mask_story.mp4")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args(argv)
    sections = [section(clip, csv, versions, title, text, args.fps) for title, text, clip, csv, versions in STORY]
    width = max(s[0].shape[1] for s in sections)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(24, 24, 24))
            for s in sections for r in s]
    write_video(Path(args.out), rows, args.fps)
    print(f"wrote {args.out}: {len(rows)} frames, {len(rows) / args.fps:.0f} s")


if __name__ == "__main__":
    main()
