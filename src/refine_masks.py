"""Stage 3: clean up masks before inpainting.

B0 only binarizes the SAM 2 masks. SAM 2 masks hug the object tightly, so a thin
ring of object pixels (blurred edges, motion blur, attached shadow) survives and
gets smeared into the fill as a visible "halo".

I1 adds three morphology steps, each off by default so B0 stays reproducible:
  min_blob_area  drop connected components smaller than this many pixels
                 (stray specks SAM 2 sometimes marks far from the object)
  fill_holes     fill background pixels completely enclosed by the mask
                 (e.g. the gap between a dog's legs that SAM 2 left out by mistake
                 is NOT enclosed, so it stays; a speckle inside the body is filled)
  dilate_px      grow the mask outward by this many pixels with an elliptical kernel,
                 so the halo ring is inside the region that gets inpainted
Order matters: clean specks first (so they don't get dilated), then fill, then dilate.

Usage:
    python -m src.refine_masks --config configs/B0.yaml --clip dog_on_soapbox
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.utils import clip_paths, list_frames, load_config, read_mask, stage_done, timed, write_mask


def remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area  # label 0 is the background
    return keep[labels]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood-fill the background from the image border; anything not reached is a hole."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)  # 1-px pad so the border is all background
    padded[1:-1, 1:-1] = mask
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return mask | (flood[1:-1, 1:-1] == 0)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def refine(mask: np.ndarray, refine_cfg: dict) -> np.ndarray:
    """B0 config: identity (binarization already happened in read_mask)."""
    if refine_cfg.get("min_blob_area", 0) > 0:
        mask = remove_small_blobs(mask, refine_cfg["min_blob_area"])
    if refine_cfg.get("fill_holes", False):
        mask = fill_holes(mask)
    if refine_cfg.get("dilate_px", 0) > 0:
        mask = dilate(mask, refine_cfg["dilate_px"])
    return mask


def run(cfg: dict, clip: str, overwrite: bool = False) -> None:
    paths = clip_paths(cfg, clip)
    masks = list_frames(paths["masks"])
    out_dir = paths["masks_refined"]
    if not overwrite and stage_done(out_dir, len(masks)):
        print(f"  [refine]  {clip}: already done, skipping")
        return

    refine_cfg = cfg.get("refine", {})
    threshold = refine_cfg.get("threshold", 127)
    with timed(paths["timing"], "refine", len(masks)):
        for m in masks:
            write_mask(out_dir / m.name, refine(read_mask(m, threshold), refine_cfg))
    print(f"  [refine]  {clip}: {len(masks)} masks")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.clip, args.overwrite)


if __name__ == "__main__":
    main()
