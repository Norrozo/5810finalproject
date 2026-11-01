"""Score grounding alone: does the chosen box land on the right object in frame 0?

Much faster than the full pipeline (no SAM 2), so grounding ideas can be compared on
the real DEV set (configs/real_dev_clips.txt) before one is run on the test sets.
A hit = box IoU > 0.5 with the true object's bounding box. Writes results/grounding_dev.csv.

Multi-object mode (I7): with --multi-min-scores, detect once per clip of a multi-object
list and, for each min_score, count objects found and extra boxes (src.evaluate.object_counts).
Writes results/multi_grounding_dev.csv. Used to choose grounding.multi.min_score on dev.

Usage:
    python -m src.eval_grounding --configs configs/I2.yaml configs/I2b.yaml --clips-file configs/real_dev_clips.txt
    python -m src.eval_grounding --configs configs/I7.yaml --clips-file configs/multi_dev_clips.txt \
        --multi-min-scores 0.2 0.25 0.3 0.35 0.4 --out results/multi_grounding_dev.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluate import box_iou, mask_to_box, object_counts
from src.grounding import find, read_prompts_file, select_all, unload_model
from src.make_real import read_real_list
from src.utils import clip_paths, list_frames, load_config, read_image, read_mask


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--clips-file", default="configs/real_dev_clips.txt")
    ap.add_argument("--out", default="results/grounding_dev.csv")
    ap.add_argument("--multi-min-scores", type=float, nargs="+", help="multi-object mode (I7)")
    args = ap.parse_args(argv)

    clips = [e["name"] for e in read_real_list(args.clips_file)]
    prompts = read_prompts_file(args.clips_file)
    rows = multi_rows(args, clips, prompts) if args.multi_min_scores else single_rows(args, clips, prompts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def multi_rows(args, clips, prompts) -> list[dict]:
    cfg = load_config(args.configs[0])
    m_cfg = cfg["grounding"].get("multi", {})
    dets = {}
    for clip in clips:
        first = list_frames(clip_paths(cfg, clip)["frames"])[0]
        dets[clip] = find(read_image(first), prompts[clip], cfg["grounding"])
    unload_model()
    rows = []
    for t in args.multi_min_scores:
        for clip in clips:
            gt = json.loads((Path(cfg["data_root"]) / clip / "gt_boxes.json").read_text())
            chosen = [d["box"] for d in select_all(dets[clip], {**m_cfg, "min_score": t})]
            rows.append({"min_score": t, "clip": clip, "prompt": prompts[clip], **object_counts(chosen, gt)})
        r = [x for x in rows if x["min_score"] == t]
        found, n, extra = (sum(x[k] for x in r) for k in ("objects_found", "n_objects", "objects_extra"))
        print(f"min_score {t:.2f}: found {found}/{n} objects, {extra} extra boxes  (found - extra = {found - extra})")
    return rows


def single_rows(args, clips, prompts) -> list[dict]:
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
    return rows


if __name__ == "__main__":
    main()
