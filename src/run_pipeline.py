"""Run the full pipeline for one version: segment -> refine -> inpaint -> visualize -> evaluate.

Benchmark (all clips in configs/eval_clips.txt, auto-click from ground truth):
    python -m src.run_pipeline --config configs/B0.yaml

Your own video (needs a click on the object in the first frame):
    python -m src.run_pipeline --config configs/B0.yaml --video my.mp4 --name my_clip --click 320,240

Stages skip work that is already on disk, so re-running after a Colab disconnect resumes.
Use --overwrite to force everything to re-run.
"""
from __future__ import annotations

import argparse

from src import evaluate, extract_frames, inpaint, refine_masks, segment, visualize
from src.utils import clip_paths, load_config, read_clip_list


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", nargs="*", help="Subset of eval clips (default: all)")
    ap.add_argument("--video", help="Run on your own video instead of the benchmark")
    ap.add_argument("--name", help="Clip name for --video")
    ap.add_argument("--click", help="X,Y on the object in frame 0 (for --video)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-viz", action="store_true", help="Skip side-by-side videos (faster)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    click = [int(v) for v in args.click.split(",")] if args.click else None

    if args.video:
        if not args.name or not click:
            ap.error("--video needs --name and --click")
        n = extract_frames.extract(args.video, clip_paths(cfg, args.name)["frames"],
                                   cfg["max_frames"], cfg["max_height"])
        print(f"Extracted {n} frames")
        clips = [args.name]
    else:
        clips = args.clips or [e["name"] for e in read_clip_list(cfg["eval_clips"])]

    print(f"== {cfg['version']} on {len(clips)} clip(s) ==")
    for clip in clips:
        segment.run(cfg, clip, click=click, overwrite=args.overwrite)
        refine_masks.run(cfg, clip, overwrite=args.overwrite)
        inpaint.run(cfg, clip, overwrite=args.overwrite)
        if not args.no_viz:
            visualize.run(cfg, clip)

    rows = evaluate.run(cfg, clips)
    scored = [r for r in rows if "JF" in r]
    if scored:
        mean = lambda k: sum(r[k] for r in scored) / len(scored)
        print(f"== {cfg['version']} mean over {len(scored)} clips: J&F={mean('JF'):.3f}  "
              f"PSNR={mean('psnr'):.2f}  PSNR(mask)={mean('psnr_mask'):.2f}  "
              f"SSIM={mean('ssim'):.3f}  flicker={mean('flicker'):.2f} ==")
    print(f"Results written to {cfg['results_csv']}")


if __name__ == "__main__":
    main()
