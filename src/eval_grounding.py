"""Score grounding alone: does the chosen box land on the right object in frame 0?

Much faster than the full pipeline (no SAM 2), so grounding ideas can be compared on
the real DEV set (configs/real_dev_clips.txt) before one is run on the test sets.
A hit = box IoU > 0.5 with the true object's bounding box. Writes results/grounding_dev.csv.

Usage:
    python -m src.eval_grounding --configs configs/I2.yaml configs/I2b.yaml --clips-file configs/real_dev_clips.txt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluate import box_iou, mask_to_box
from src.grounding import find, read_prompts_file, unload_model
from src.make_real import read_real_list
from src.utils import clip_paths, list_frames, load_config, read_image, read_mask


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--clips-file", default="configs/real_dev_clips.txt")
    ap.add_argument("--out", default="results/grounding_dev.csv")
    args = ap.parse_args(argv)

    clips = [e["name"] for e in read_real_list(args.clips_file)]
    prompts = read_prompts_file(args.clips_file)
    rows = []
    for c in args.configs:
        cfg = load_config(c)
        hits = []
        for clip in clips:
            paths = clip_paths(cfg, clip)
            first = list_frames(paths["frames"])[0]
            gt = mask_to_box(read_mask(paths["gt_masks"] / first.name))
            dets = find(read_image(first), prompts[clip], cfg["grounding"])
            iou = box_iou(dets[0]["box"], gt) if dets else 0.0
            hits.append(iou > 0.5)
            rows.append({"variant": cfg["version"], "clip": clip, "prompt": prompts[clip],
                         "iou": round(iou, 3), "hit": int(iou > 0.5), "n_candidates": len(dets)})
        unload_model()
        print(f"{cfg['version']:10s} hit rate {sum(hits)}/{len(hits)}  "
              + " ".join("x" if not h else "." for h in hits))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
