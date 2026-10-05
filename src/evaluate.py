"""Stage 5: score a pipeline version and write one row per clip to results/results.csv.

Metrics (synthetic benchmark clips, which have ground truth):
  J          region similarity: IoU between predicted and GT masks            (higher = better)
  F          boundary accuracy: F-measure of mask edges within a tolerance    (higher = better)
  JF         mean of J and F, the standard DAVIS summary number
  psnr       PSNR of the output vs. the clean background, whole frame         (higher = better)
  psnr_mask  PSNR inside the (slightly dilated) object region only            (higher = better)
             -> more sensitive, since most of the frame is untouched
  ssim       structural similarity, whole frame                               (higher = better)
  flicker    temporal inconsistency inside the region: mean |out_t - warp(out_{t-1})|,
             warping with optical flow computed on the clean GT video          (lower = better)
  flicker_ref the same measure on the GT video itself: the floor you can approach
Runtime (all clips, including custom videos without GT):
  spf_<stage> seconds per frame for each stage, plus spf_total

J and F score the raw SAM 2 masks (stage 2), not the refined ones: refinement
deliberately grows the mask past the object (I1), which would lower J&F without
tracking getting any worse. Refinement is judged by the inpainting metrics.

Note: J and F here are our own implementation following the DAVIS definitions
(boundary tolerance = 0.008 x image diagonal). They closely track, but are not
byte-identical to, the official DAVIS toolkit numbers.

Usage:
    python -m src.evaluate --config configs/B0.yaml            # all eval clips
    python -m src.evaluate --config configs/B0.yaml --clips dog_on_soapbox
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from src.utils import clip_paths, list_frames, load_config, read_clip_list, read_image, read_mask

COLUMNS = ["version", "clip", "n_frames", "J", "F", "JF", "psnr", "psnr_mask", "ssim",
           "flicker", "flicker_ref", "spf_segment", "spf_refine", "spf_inpaint", "spf_total", "timestamp"]
PSNR_CAP = 100.0  # identical images give infinite PSNR; cap so averages stay finite


# ---------------------------------------------------------------- segmentation metrics

def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def boundary_f(pred: np.ndarray, gt: np.ndarray, tol_frac: float) -> float:
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() and not bg.any():
        return 1.0
    if not bp.any() or not bg.any():
        return 0.0
    r = max(1, round(tol_frac * math.hypot(*pred.shape)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bg_near = cv2.dilate(bg.astype(np.uint8), disk) > 0
    bp_near = cv2.dilate(bp.astype(np.uint8), disk) > 0
    precision = (bp & bg_near).sum() / bp.sum()
    recall = (bg & bp_near).sum() / bg.sum()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------- image metrics

def psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    if region is not None:
        if not region.any():
            return PSNR_CAP
        diff = diff[region]
    mse = np.mean(diff ** 2)
    return PSNR_CAP if mse == 0 else min(PSNR_CAP, 10 * math.log10(255.0 ** 2 / mse))


def warp_prev_to_curr(prev_img: np.ndarray, flow_curr_to_prev: np.ndarray) -> np.ndarray:
    h, w = flow_curr_to_prev.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(prev_img, gx + flow_curr_to_prev[..., 0], gy + flow_curr_to_prev[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def flicker_scores(outputs: list[np.ndarray], gts: list[np.ndarray], regions: list[np.ndarray]) -> tuple[float, float]:
    """Temporal inconsistency inside the region, for the output and for the GT (reference)."""
    out_scores, ref_scores = [], []
    for t in range(1, len(outputs)):
        region = regions[t]
        if not region.any():
            continue
        g_prev = cv2.cvtColor(gts[t - 1], cv2.COLOR_BGR2GRAY)
        g_curr = cv2.cvtColor(gts[t], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g_curr, g_prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for seq, scores in ((outputs, out_scores), (gts, ref_scores)):
            warped = warp_prev_to_curr(seq[t - 1], flow)
            diff = np.abs(seq[t].astype(np.float32) - warped.astype(np.float32))
            scores.append(float(diff[region].mean()))
    mean = lambda s: float(np.mean(s)) if s else float("nan")
    return mean(out_scores), mean(ref_scores)


# ---------------------------------------------------------------- per clip

def evaluate_clip(cfg: dict, clip: str) -> dict:
    paths = clip_paths(cfg, clip)
    ev = cfg.get("evaluation", {})
    frames = list_frames(paths["frames"])
    row = {"version": cfg["version"], "clip": clip, "n_frames": len(frames)}

    if paths["timing"].exists():
        timing = json.loads(paths["timing"].read_text())
        total = 0.0
        for stage in ("segment", "refine", "inpaint"):
            if stage in timing:
                row[f"spf_{stage}"] = timing[stage]["sec_per_frame"]
                total += timing[stage]["sec_per_frame"]
        row["spf_total"] = round(total, 4)

    if not paths["gt_masks"].exists():
        return row  # custom video: runtime only

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ev.get("region_dilation", 5) + 1,) * 2)
    Js, Fs, psnrs, psnr_masks, ssims = [], [], [], [], []
    outputs, gts, regions = [], [], []
    for f in frames:
        gt_mask = read_mask(paths["gt_masks"] / f.name)
        pred_mask = read_mask(paths["masks"] / f.name)  # raw tracker output, before refinement
        Js.append(jaccard(pred_mask, gt_mask))
        Fs.append(boundary_f(pred_mask, gt_mask, ev.get("boundary_tolerance", 0.008)))

        out = read_image(paths["inpainted"] / f.name)
        gt = read_image(paths["gt_clean"] / f.name)
        region = cv2.dilate(gt_mask.astype(np.uint8), kernel) > 0
        psnrs.append(psnr(out, gt))
        psnr_masks.append(psnr(out, gt, region))
        ssims.append(structural_similarity(gt, out, channel_axis=2, data_range=255))
        outputs.append(out); gts.append(gt); regions.append(region)

    flicker, flicker_ref = flicker_scores(outputs, gts, regions)
    row.update(J=np.mean(Js), F=np.mean(Fs), JF=(np.mean(Js) + np.mean(Fs)) / 2,
               psnr=np.mean(psnrs), psnr_mask=np.mean(psnr_masks), ssim=np.mean(ssims),
               flicker=flicker, flicker_ref=flicker_ref)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}


# ---------------------------------------------------------------- results CSV

def upsert_rows(csv_path: Path, new_rows: list[dict]) -> None:
    """Add rows, replacing any existing row for the same (version, clip)."""
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    keys = {(r["version"], r["clip"]) for r in new_rows}
    rows = [r for r in rows if (r["version"], r["clip"]) not in keys]
    stamp = datetime.now().isoformat(timespec="seconds")
    rows += [{**r, "timestamp": stamp} for r in new_rows]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(cfg: dict, clips: list[str]) -> list[dict]:
    rows = []
    for clip in clips:
        row = evaluate_clip(cfg, clip)
        rows.append(row)
        if "JF" in row:
            print(f"  [eval]    {clip}: J&F={row['JF']:.3f}  PSNR={row['psnr']:.2f}  "
                  f"PSNR(mask)={row['psnr_mask']:.2f}  SSIM={row['ssim']:.3f}  "
                  f"flicker={row['flicker']:.2f} (ref {row['flicker_ref']:.2f})")
        else:
            print(f"  [eval]    {clip}: no ground truth, runtime only")
    upsert_rows(Path(cfg["results_csv"]), rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", nargs="*", help="Default: all clips in eval_clips")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    clips = args.clips or [e["name"] for e in read_clip_list(cfg["eval_clips"])]
    run(cfg, clips)


if __name__ == "__main__":
    main()
