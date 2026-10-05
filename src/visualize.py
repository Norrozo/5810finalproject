"""Make report-ready visuals for one clip of one version.

Writes to runs/<version>/<clip>/:
    side_by_side.mp4  Input | Mask | Result (| Ground truth, if available)
    stills.png        the same panels for the first, middle, and last frame

Usage:
    python -m src.visualize --config configs/B0.yaml --clip dog_on_soapbox
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.utils import clip_paths, list_frames, load_config, read_image, read_mask

PANEL_HEIGHT = 240


def overlay_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    red = np.zeros_like(img); red[..., 2] = 255
    out[mask] = (0.45 * img[mask] + 0.55 * red[mask]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (8 + 9 * len(text), 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def panel_row(paths: dict, name: str) -> np.ndarray:
    frame = read_image(paths["frames"] / name)
    mask = read_mask(paths["masks_refined"] / name)
    panels = [("Input", frame), ("Mask", overlay_mask(frame, mask)),
              ("Result", read_image(paths["inpainted"] / name))]
    if (paths["gt_clean"] / name).exists():
        panels.append(("Ground truth", read_image(paths["gt_clean"] / name)))
    h, w = frame.shape[:2]
    size = (int(w * PANEL_HEIGHT / h), PANEL_HEIGHT)
    return np.hstack([label(cv2.resize(img, size, interpolation=cv2.INTER_AREA), t) for t, img in panels])


def write_video(path: Path, rows: list[np.ndarray], fps: int) -> None:
    """Write an mp4. OpenCV can only write MPEG-4 Part 2 ("mp4v"), which QuickTime and
    browsers won't play, so re-encode to H.264 with ffmpeg when it's installed."""
    h, w = rows[0].shape[:2]
    w, h = w - w % 2, h - h % 2  # H.264 needs even dimensions
    tmp = path.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for r in rows:
        writer.write(r[:h, :w])
    writer.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(path)], check=True)
        tmp.unlink()
    else:
        tmp.replace(path)


def run(cfg: dict, clip: str, fps: int = 12) -> None:
    paths = clip_paths(cfg, clip)
    names = [f.name for f in list_frames(paths["frames"])]
    rows = [panel_row(paths, n) for n in names]

    write_video(paths["run"] / "side_by_side.mp4", rows, fps)

    picks = sorted({0, len(rows) // 2, len(rows) - 1})
    cv2.imwrite(str(paths["run"] / "stills.png"), np.vstack([rows[i] for i in picks]))
    print(f"  [viz]     {clip}: side_by_side.mp4, stills.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args(argv)
    run(load_config(args.config), args.clip, args.fps)


if __name__ == "__main__":
    main()
