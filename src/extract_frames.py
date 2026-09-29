"""Turn a video file into numbered PNG frames under data/<name>/frames/.

Frames are resized so height <= max_height and capped at max_frames (from the config),
which keeps SAM 2 and inpainting within Colab's GPU memory.

Usage:
    python -m src.extract_frames --config configs/B0.yaml --video my_clip.mp4 --name my_clip
"""
from __future__ import annotations

import argparse

import cv2

from src.utils import clip_paths, load_config, write_image


def extract(video_path: str, out_dir, max_frames: int, max_height: int, stride: int = 1) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    written, idx = 0, 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            if h > max_height:
                scale = max_height / h
                new_w = int(round(w * scale / 2) * 2)  # keep width even for video encoders
                frame = cv2.resize(frame, (new_w, max_height), interpolation=cv2.INTER_AREA)
            write_image(out_dir / f"{written:05d}.png", frame)
            written += 1
        idx += 1
    cap.release()
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True, help="Clip name used for data/<name>/")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = clip_paths(cfg, args.name)["frames"]
    n = extract(args.video, out, cfg["max_frames"], cfg["max_height"], args.stride)
    print(f"Wrote {n} frames to {out}")


if __name__ == "__main__":
    main()
