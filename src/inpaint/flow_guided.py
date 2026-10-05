"""I3: our own flow-guided video inpainting.

Idea: when the object or the camera moves, the background hidden in frame t is
usually visible in some other frame. Optical flow tells us where to look for it.

Steps (all per clip):
  1. Flow. Farneback dense optical flow between each pair of neighboring frames,
     in both directions: fwd[t] maps frame t -> t+1, bwd[t] maps frame t -> t-1.
     A flow vector at pixel p says "the thing at p is at p + flow(p) in the other frame".

  2. Flow completion. Inside the hole, the flow follows the *object*, which is the
     wrong motion: we want where the *background* behind the object went. So we
     erase the flow in the hole (dilated a little, because Farneback smooths over a
     window and the object's motion leaks just outside its mask) and fill it in from
     the surrounding background flow with cv2.inpaint. Background motion is mostly
     smooth (camera pan/zoom), so this works well; it fails for complex 3D scenes.

  3. Propagation. Forward sweep t = 1..N-1: for each hole pixel p in frame t, look up
     q = p + bwd[t](p) in frame t-1. If q is known there (real background, or already
     filled by this sweep), copy its color. Then a backward sweep t = N-2..0 using fwd[t].
     Each filled pixel remembers its "age": how many frames away its real source was.

  4. Merge. Where both sweeps found a value, keep the one with the smaller age (fewer
     hops -> less blur from repeated bilinear sampling); on a tie, average them.

  5. Fallback. Pixels that no frame ever saw (the object covered that spot for the
     whole clip) are filled with cv2.inpaint (Telea), as in B0.

  Optional hop cap (max_hops). Every hop adds a little flow error and a little blur
  from bilinear sampling, so a pixel copied through 10 frames is much worse than one
  copied from the next frame (measured: ~18.7 dB at 1 hop vs ~12 dB at 8+ hops on
  dog/camel/breakdance). With max_hops = K, a pixel only counts as filled if its real
  source is at most K frames away; the rest goes to the Telea fallback, which then has
  a much smaller hole bordered by recovered background. Tuned on the dev clips
  (configs/dev_clips.txt), never on the test set.

Why this helps vs. B0: copied pixels are real background, so the fill is sharp, and
the same background point is copied into every frame, so it doesn't flicker.
Limitations: wrong flow -> wrong pixels; repeated sampling blurs over many hops;
brightness changes between frames are copied as-is (no blending).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import list_frames, read_image, read_mask, write_image


# ---------------------------------------------------------------- flow

def compute_flow(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Dense flow from src to dst (H x W x 2, in pixels). Same Farneback settings as evaluate.py."""
    g_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(g_src, g_dst, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def complete_flow(flow: np.ndarray, mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Replace the flow inside the (dilated) hole with a smooth fill from the surrounding background flow."""
    if not mask.any():
        return flow
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    region = cv2.dilate(mask.astype(np.uint8), kernel)
    return np.dstack([cv2.inpaint(np.ascontiguousarray(flow[..., c]), region, 5, cv2.INPAINT_TELEA)
                      for c in range(2)])


# ---------------------------------------------------------------- propagation

def sample(img: np.ndarray, flow: np.ndarray, interp: int) -> np.ndarray:
    """out[p] = img[p + flow(p)]. Outside the image counts as 0 (= unknown for masks)."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], interp,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sweep(frames: list[np.ndarray], masks: list[np.ndarray], flows: list[np.ndarray | None],
          order: range, step: int, max_hops: int | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One propagation pass. Frame t pulls from frame t - step, which this pass already filled.
    Pixels whose source would be more than max_hops frames away stay unknown.

    Returns per-frame filled images and ages (age = inf where still unknown).
    """
    n = len(frames)
    imgs = [f.astype(np.float32) for f in frames]
    ages = [np.where(m, np.inf, 0).astype(np.float32) for m in masks]
    for t in order:
        hole = ~np.isfinite(ages[t])
        if not hole.any():
            continue
        prev = t - step
        known_prev = np.isfinite(ages[prev]).astype(np.float32)
        # Bilinear-sampled "known" is < 1 if any of the 4 neighbors is unknown: only accept fully-known spots.
        ok = hole & (sample(known_prev, flows[t], cv2.INTER_LINEAR) > 0.999)
        age = sample(np.where(np.isfinite(ages[prev]), ages[prev], 0), flows[t], cv2.INTER_NEAREST)
        if max_hops is not None:
            ok &= age + 1 <= max_hops
        if not ok.any():
            continue
        color = sample(imgs[prev], flows[t], cv2.INTER_LINEAR)
        imgs[t][ok] = color[ok]
        ages[t][ok] = age[ok] + 1
    assert len(imgs) == n
    return imgs, ages


def merge(img_f: np.ndarray, age_f: np.ndarray, img_b: np.ndarray, age_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine the forward and backward results; return the image and the still-unknown mask."""
    use_f = age_f < age_b
    use_b = age_b < age_f
    tie = (age_f == age_b) & np.isfinite(age_f)
    out = img_f.copy()
    out[use_b] = img_b[use_b]
    out[tie] = 0.5 * (img_f[tie] + img_b[tie])
    unknown = ~np.isfinite(np.minimum(age_f, age_b))
    return out, unknown


# ---------------------------------------------------------------- entry point

def inpaint_clip(frames_dir: Path, masks_dir: Path, out_dir: Path, inpaint_cfg: dict) -> int:
    frame_paths = list_frames(frames_dir)
    frames = [read_image(f) for f in frame_paths]
    masks = [read_mask(masks_dir / f.name) for f in frame_paths]
    n = len(frames)
    dilate_px = inpaint_cfg.get("flow_dilate_px", 15)

    # bwd[t]: t -> t-1 (used by the forward sweep); fwd[t]: t -> t+1 (used by the backward sweep).
    bwd = [None] + [complete_flow(compute_flow(frames[t], frames[t - 1]), masks[t], dilate_px) for t in range(1, n)]
    fwd = [complete_flow(compute_flow(frames[t], frames[t + 1]), masks[t], dilate_px) for t in range(n - 1)] + [None]

    max_hops = inpaint_cfg.get("max_hops")  # None = no cap (the original I3)
    imgs_f, ages_f = sweep(frames, masks, bwd, range(1, n), step=1, max_hops=max_hops)
    imgs_b, ages_b = sweep(frames, masks, fwd, range(n - 2, -1, -1), step=-1, max_hops=max_hops)

    radius = inpaint_cfg.get("radius", 5)
    for t, f in enumerate(frame_paths):
        filled, unknown = merge(imgs_f[t], ages_f[t], imgs_b[t], ages_b[t])
        filled = np.clip(filled, 0, 255).astype(np.uint8)
        if unknown.any():  # never seen in any frame: fall back to spatial inpainting
            filled = cv2.inpaint(filled, unknown.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
        write_image(out_dir / f.name, filled)
    return n
