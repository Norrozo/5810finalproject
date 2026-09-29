"""B0 inpainting: fill each frame independently with OpenCV's classical inpainting.

cv2.inpaint fills the hole by propagating colors inward from its border
(Telea's fast-marching method, or Navier-Stokes). It's fast and needs no GPU, but:
  - it only sees the current frame, so the fill changes frame to frame -> flicker
  - it only knows the hole's border colors, so large holes turn into blurry smears
  - it can't recover real background that was visible in other frames

Those weaknesses are what iterations I3 (flow-guided, uses other frames) and
I4 (ProPainter) are meant to fix.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import list_frames, read_image, read_mask, write_image

METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def inpaint_frame(img: np.ndarray, mask: np.ndarray, radius: int, method: str) -> np.ndarray:
    if not mask.any():
        return img.copy()
    return cv2.inpaint(img, mask.astype(np.uint8) * 255, radius, METHODS[method])


def inpaint_clip(frames_dir: Path, masks_dir: Path, out_dir: Path, inpaint_cfg: dict) -> int:
    frames = list_frames(frames_dir)
    radius = inpaint_cfg.get("radius", 5)
    method = inpaint_cfg.get("opencv_method", "telea")
    for f in frames:
        filled = inpaint_frame(read_image(f), read_mask(masks_dir / f.name), radius, method)
        write_image(out_dir / f.name, filled)
    return len(frames)
