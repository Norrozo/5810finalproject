"""Build the REAL test set: unedited DAVIS videos with their own object masks.

The synthetic benchmark (make_synthetic.py) pastes objects with hard edges and no
shadows, which makes segmentation easier than in real footage. This set scores
tracking (J&F) on real videos against DAVIS's annotation of one object. There is no
clean background, so fill quality is judged visually on these clips.

For each line of configs/real_clips.txt (name, davis_clip, object_id, prompt) writes:
    data/<name>/frames/    first max_frames frames
    data/<name>/gt_masks/  DAVIS mask of object_id

Multi-object lists (configs/multi_clips.txt, I7) give several ids ("1,2,3"): the mask is
their union, and data/<name>/gt_boxes.json holds each object's frame-0 box, so evaluation
can count how many of the objects the text prompt found.

Usage:
    python -m src.make_real --config configs/B0.yaml --davis-root DAVIS
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.make_synthetic import load_davis_clip, load_object_mask
from src.evaluate import mask_to_box
from src.utils import clip_paths, load_config, write_image, write_mask


def read_real_list(path: str | Path) -> list[dict]:
    entries = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, clip, obj, prompt = line.split(maxsplit=3)
            entries.append({"name": name, "clip": clip, "object_ids": [int(i) for i in obj.split(",")],
                            "prompt": prompt})
    return entries


def build(davis_root: Path, entry: dict, cfg: dict) -> int:
    paths = clip_paths(cfg, entry["name"])
    frames = load_davis_clip(davis_root, entry["clip"])[: cfg["max_frames"]]
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        objects = [load_object_mask(davis_root, entry["clip"], f.stem, k) for k in entry["object_ids"]]
        h = cfg["max_height"]
        if img.shape[0] > h:
            w = round(img.shape[1] * h / img.shape[0])
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            objects = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                       for m in objects]
        write_image(paths["frames"] / f"{i:05d}.png", img)
        write_mask(paths["gt_masks"] / f"{i:05d}.png", np.logical_or.reduce(objects))
        if i == 0 and len(objects) > 1:
            boxes = [mask_to_box(m) for m in objects if m.any()]
            (Path(cfg["data_root"]) / entry["name"] / "gt_boxes.json").write_text(json.dumps(boxes))
    return len(frames)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--davis-root", required=True)
    ap.add_argument("--clips-file", default="configs/real_clips.txt")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for e in read_real_list(args.clips_file):
        print(f"  {e['name']}: {build(Path(args.davis_root), e, cfg)} frames")


if __name__ == "__main__":
    main()
