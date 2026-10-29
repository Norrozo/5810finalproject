"""Learned metrics for evaluate.py: LPIPS and RAFT-aligned flicker.

lpips_mask   LPIPS (Zhang et al., 2018) between the output and the clean video, on a
             crop around the object region. LPIPS compares deep-network features, so it
             tracks perceived similarity better than PSNR/SSIM and does not reward blur.
             It is computed on a crop because, like PSNR, a whole-frame score would be
             dominated by the pixels every method leaves untouched.            (lower = better)
flicker_raft The same flicker measure as evaluate.flicker_scores, but frames are aligned
             with RAFT (Teed & Deng, 2020) instead of Farneback. Our own inpainting method
             (I3b) uses Farneback flow, so a Farneback-aligned metric could favor it;
             RAFT is an independent, more accurate flow estimator.               (lower = better)

RAFT flow on the clean video is identical for every version, so it is computed once
per clip and cached in data/<clip>/raft_flow.npy (float16, git-ignored).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def _lpips_model():
    import lpips
    return lpips.LPIPS(net="alex", verbose=False).to(_device()).eval()


@lru_cache(maxsize=1)
def _raft_model():
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    return raft_large(weights=Raft_Large_Weights.DEFAULT).to(_device()).eval()


def unload() -> None:
    """Free the networks (keeps memory free for the next pipeline stage)."""
    import gc
    import torch
    _lpips_model.cache_clear()
    _raft_model.cache_clear()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_tensor(img_bgr: np.ndarray):
    """BGR uint8 -> 1x3xHxW RGB float in [-1, 1], the input range LPIPS and RAFT expect."""
    import torch
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].to(_device())


def region_crop(region: np.ndarray, pad: int = 16, min_size: int = 64) -> tuple[slice, slice] | None:
    """Bounding box of the region, padded, at least min_size on each side (LPIPS needs context)."""
    ys, xs = np.where(region)
    if len(ys) == 0:
        return None
    h, w = region.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    if y1 - y0 < min_size:
        c = (y0 + y1) // 2
        y0, y1 = max(0, c - min_size // 2), min(h, c + min_size // 2)
    if x1 - x0 < min_size:
        c = (x0 + x1) // 2
        x0, x1 = max(0, c - min_size // 2), min(w, c + min_size // 2)
    return slice(y0, y1), slice(x0, x1)


def lpips_mask(outputs: list[np.ndarray], gts: list[np.ndarray], regions: list[np.ndarray]) -> float:
    import torch
    model = _lpips_model()
    scores = []
    with torch.inference_mode():
        for out, gt, region in zip(outputs, gts, regions):
            crop = region_crop(region)
            if crop is None:
                continue
            scores.append(float(model(_to_tensor(out[crop]), _to_tensor(gt[crop])).item()))
    return float(np.mean(scores)) if scores else float("nan")


def raft_flows(gts: list[np.ndarray], cache: Path | None = None) -> np.ndarray:
    """Flow from frame t to frame t-1 for t = 1..n-1 (index t-1), shape (n-1, H, W, 2)."""
    if cache is not None and cache.exists():
        flows = np.load(cache)
        if len(flows) == len(gts) - 1:
            return flows.astype(np.float32)
    import torch
    import torch.nn.functional as F
    model = _raft_model()
    h, w = gts[0].shape[:2]
    ph, pw = (-h) % 8, (-w) % 8  # RAFT needs sizes divisible by 8
    flows = []
    with torch.inference_mode():
        for t in range(1, len(gts)):
            a = F.pad(_to_tensor(gts[t]), (0, pw, 0, ph), mode="replicate")
            b = F.pad(_to_tensor(gts[t - 1]), (0, pw, 0, ph), mode="replicate")
            flow = model(a, b)[-1][0, :, :h, :w].permute(1, 2, 0).cpu().numpy()
            flows.append(flow.astype(np.float16))
    flows = np.stack(flows)
    if cache is not None:
        np.save(cache, flows)
    return flows.astype(np.float32)


def flicker_raft(outputs: list[np.ndarray], gts: list[np.ndarray], regions: list[np.ndarray],
                 cache: Path | None = None) -> tuple[float, float]:
    """Flicker for the output and for the clean video (reference), RAFT-aligned."""
    from src.evaluate import warp_prev_to_curr
    flows = raft_flows(gts, cache)
    out_scores, ref_scores = [], []
    for t in range(1, len(outputs)):
        region = regions[t]
        if not region.any():
            continue
        for seq, scores in ((outputs, out_scores), (gts, ref_scores)):
            warped = warp_prev_to_curr(seq[t - 1], flows[t - 1])
            diff = np.abs(seq[t].astype(np.float32) - warped.astype(np.float32))
            scores.append(float(diff[region].mean()))
    mean = lambda s: float(np.mean(s)) if s else float("nan")
    return mean(out_scores), mean(ref_scores)
