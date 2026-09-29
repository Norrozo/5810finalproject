"""Stage 3: clean up masks before inpainting.

B0 only binarizes the SAM 2 masks. That is intentionally minimal: SAM 2 masks
hug the object tightly, so a thin ring of object pixels (and any attached shadow)
usually survives and gets smeared into the fill. Expect visible "halos".

Iteration I1 is where this stage gets real work, for example:
  - dilating the mask by a few pixels (cv2.dilate)
  - filling holes inside the mask (e.g. flood fill from the border, or cv2.morphologyEx CLOSE)
  - removing tiny disconnected blobs (cv2.connectedComponentsWithStats)
Add the new settings to configs/I1.yaml under `refine:` so B0 stays reproducible.

Usage:
    python -m src.refine_masks --config configs/B0.yaml --clip dog_on_soapbox
"""
from __future__ import annotations

import argparse

import numpy as np

from src.utils import clip_paths, list_frames, load_config, read_mask, stage_done, timed, write_mask


def refine(mask: np.ndarray, refine_cfg: dict) -> np.ndarray:
    # B0: identity (binarization already happened in read_mask).
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
