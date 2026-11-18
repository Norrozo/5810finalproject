"""I3h (proposal name: I3b): global alignment + temporal median fill.

Idea: our flow fill (I3/I3b) copies a pixel frame by frame, so its error grows with
every hop. If the camera only pans/zooms over a distant background, one 3x3
homography maps a whole neighbor frame onto the current one, so we can copy from a
frame 10 steps away directly, in one step, with no accumulated error.

Steps (per clip):
  1. Features. ORB keypoints + descriptors in every frame, only on the background
     (the object's mask, dilated a little, is excluded), so the object's own motion
     never votes.
  2. Alignment. For frame t and each neighbor s with |s - t| <= window: match ORB
     descriptors (Lowe's ratio test), then fit a homography H (s -> t) with RANSAC,
     which ignores mismatches and anything that moves differently from the
     background. Pairs with fewer than min_inliers inliers are skipped (not aligned).
  3. Temporal median. Warp each aligned neighbor into frame t. For every hole pixel,
     take the per-channel median over the neighbors where that spot was background
     (not masked in s, inside s's image). The median ignores a few badly aligned or
     differently lit frames, and since frames t and t+1 use almost the same
     neighbors, the fill barely changes between frames -> little flicker.
  4. Fallback. Pixels no neighbor showed (the object covered them in every frame of
     the window) are filled with cv2.inpaint (Telea), as in B0.

Assumption: the background is far away or flat (a homography models a plane or a
rotating camera). Near, 3D backgrounds with parallax will be misaligned; the median
and the inlier check limit, but do not remove, that error.
Hyperparameter: window (frames on each side), tuned on the dev clips with src/sweep.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import list_frames, read_image, read_mask, write_image


def features(img: np.ndarray, mask: np.ndarray, orb, dilate_px: int):
    """ORB keypoints/descriptors on the background only."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    allowed = (cv2.dilate(mask.astype(np.uint8), k) == 0).astype(np.uint8) * 255
    return orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), allowed)


def homography(feat_src, feat_dst, matcher, min_inliers: int) -> np.ndarray | None:
    """Homography mapping src pixels onto dst pixels, or None if the match is unreliable."""
    (kp_s, des_s), (kp_d, des_d) = feat_src, feat_dst
    if des_s is None or des_d is None or len(kp_s) < min_inliers or len(kp_d) < min_inliers:
        return None
    pairs = matcher.knnMatch(des_s, des_d, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([kp_s[m.queryIdx].pt for m in good])
    dst = np.float32([kp_d[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inliers.sum() < min_inliers:
        return None
    return H


def inpaint_clip(frames_dir: Path, masks_dir: Path, out_dir: Path, inpaint_cfg: dict) -> int:
    frame_paths = list_frames(frames_dir)
    frames = [read_image(f) for f in frame_paths]
    masks = [read_mask(masks_dir / f.name) for f in frame_paths]
    n = len(frames)
    h, w = frames[0].shape[:2]
    window = inpaint_cfg.get("window", 10)
    window = n if window is None else window  # None = every frame of the clip
    min_inliers = inpaint_cfg.get("min_inliers", 30)
    dilate_px = inpaint_cfg.get("feature_dilate_px", 5)
    radius = inpaint_cfg.get("radius", 5)

    # fastThreshold 5 (OpenCV default 20) also finds corners in low-contrast texture (sky, water,
    # blurred walls); ORB still keeps only the 2000 strongest.
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=5)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    feats = [features(f, m, orb, dilate_px) for f, m in zip(frames, masks)]

    for t in range(n):
        hole = masks[t]
        filled = frames[t].copy()
        if hole.any():
            ys, xs = np.nonzero(hole)
            samples = []  # one (P, 3) array per aligned neighbor, NaN where it did not see the spot
            for s in range(max(0, t - window), min(n, t + window + 1)):
                if s == t:
                    continue
                H = homography(feats[s], feats[t], matcher, min_inliers)
                if H is None:
                    continue
                warped = cv2.warpPerspective(frames[s], H, (w, h), flags=cv2.INTER_LINEAR)
                # Valid = background in s and inside s's image. Warp "is background" with nearest
                # neighbor, after shrinking it by 1 px so bilinear colors never mix in the object.
                bg = cv2.erode((~masks[s]).astype(np.uint8), np.ones((3, 3), np.uint8))
                valid = cv2.warpPerspective(bg, H, (w, h), flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)[ys, xs] > 0
                vals = warped[ys, xs].astype(np.float32)
                vals[~valid] = np.nan
                samples.append(vals)
            unknown = hole.copy()
            if samples:
                stack = np.stack(samples)                       # (K, P, 3)
                seen = ~np.isnan(stack[..., 0]).all(axis=0)     # (P,)
                if seen.any():
                    med = np.nanmedian(stack[:, seen], axis=0)
                    filled[ys[seen], xs[seen]] = np.clip(med, 0, 255).astype(np.uint8)
                    unknown[ys[seen], xs[seen]] = False
            if unknown.any():  # never seen in the window: spatial fallback
                filled = cv2.inpaint(filled, unknown.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
        write_image(out_dir / frame_paths[t].name, filled)
    return n
