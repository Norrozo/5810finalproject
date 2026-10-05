"""Stage 2: segment the object on frame 0 from a click and track it with SAM 2.

Prompt sources, in order of priority:
  1. --click X,Y on the command line
  2. data/<clip>/prompt.json  ({"points": [[x, y], ...]})
  3. auto-click from the ground-truth mask on frame 0 (synthetic benchmark only),
     so evaluation runs need no manual clicking

B0 limitations (these are future iterations, not bugs):
  - one positive click on frame 0 only, no text prompt (I2)
  - no re-detection if the object leaves and comes back (I5)
  - single object (I7)

Usage:
    python -m src.segment --config configs/B0.yaml --clip dog_on_soapbox
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.utils import (clip_paths, list_frames, load_config, read_image, read_mask,
                       record_timing, stage_done, timed, write_mask)


# ---------------------------------------------------------------- prompts

def auto_click_from_mask(mask: np.ndarray) -> list[int]:
    """Point deepest inside the mask (max distance from its edge), like a careful human click."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return [int(x), int(y)]


def resolve_prompt(paths: dict, click: list[int] | None = None) -> dict:
    if click is not None:
        points, source = [click], "cli"
    elif paths["prompt"].exists():
        points, source = json.loads(paths["prompt"].read_text())["points"], "prompt.json"
    elif paths["gt_masks"].exists():
        first = list_frames(paths["gt_masks"])[0]
        points, source = [auto_click_from_mask(read_mask(first))], "auto_from_gt"
    else:
        raise ValueError("No prompt: pass --click X,Y or create data/<clip>/prompt.json")
    return {"frame": 0, "points": points, "labels": [1] * len(points), "source": source}


# ---------------------------------------------------------------- methods

def segment_sam2(cfg: dict, frames: list[Path], prompt: dict, out_dir: Path) -> None:
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    seg_cfg = cfg["segmentation"]
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():  # Apple Silicon; set PYTORCH_ENABLE_MPS_FALLBACK=1
        device = "mps"
    else:
        device = "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(seg_cfg["hf_model"], device=device)

    # bfloat16 only helps on Ampere+ GPUs (A100, L4). Colab's T4 runs in float32.
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    # SAM 2's video loader expects a folder of JPEGs named 00000.jpg, 00001.jpg, ...
    tmp = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"{i:05d}.jpg"), read_image(f), [cv2.IMWRITE_JPEG_QUALITY, 95])

        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(tmp),
                                         offload_video_to_cpu=seg_cfg.get("offload_video_to_cpu", True))
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=prompt["frame"], obj_id=1,
                points=np.array(prompt["points"], dtype=np.float32),
                labels=np.array(prompt["labels"], dtype=np.int32),
            )
            written = set()
            for idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                mask = (mask_logits[0] > 0.0).squeeze().cpu().numpy()
                write_mask(out_dir / frames[idx].name, mask)
                written.add(idx)

        # Safety net: any frame SAM 2 skipped gets an empty mask so later stages don't crash.
        h, w = read_image(frames[0]).shape[:2]
        for i, f in enumerate(frames):
            if i not in written:
                write_mask(out_dir / f.name, np.zeros((h, w), bool))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def segment_reuse(cfg: dict, clip: str, frames: list[Path], out_dir: Path) -> dict:
    """Copy masks from an earlier version that used identical segmentation settings.

    Iterations that only change later stages (I1, I3, I4) reuse B0's SAM 2 masks so
    every version is compared on exactly the same masks, and nobody waits for SAM 2 twice.
    Returns the earlier version's segment timing, so s/frame stays a fair comparison.
    """
    src = clip_paths({**cfg, "version": cfg["segmentation"]["reuse_from"]}, clip)
    if not stage_done(src["masks"], len(frames)):
        raise FileNotFoundError(f"reuse_from: no masks in {src['masks']}; run that version first")
    for f in frames:
        shutil.copy(src["masks"] / f.name, out_dir / f.name)
    if src["used_prompt"].exists():
        shutil.copy(src["used_prompt"], clip_paths(cfg, clip)["used_prompt"])
    return json.loads(src["timing"].read_text())["segment"]


def segment_oracle(paths: dict, frames: list[Path], out_dir: Path) -> None:
    """Copy ground-truth masks. Isolates inpainting quality from segmentation errors."""
    if not paths["gt_masks"].exists():
        raise ValueError("Oracle segmentation needs data/<clip>/gt_masks (synthetic clips only)")
    for f in frames:
        write_mask(out_dir / f.name, read_mask(paths["gt_masks"] / f.name))


# ---------------------------------------------------------------- entry point

def run(cfg: dict, clip: str, click: list[int] | None = None, overwrite: bool = False) -> None:
    paths = clip_paths(cfg, clip)
    frames = list_frames(paths["frames"])
    if not frames:
        raise FileNotFoundError(f"No frames in {paths['frames']}")
    out_dir = paths["masks"]
    if not overwrite and stage_done(out_dir, len(frames)):
        print(f"  [segment] {clip}: already done, skipping")
        return

    method = cfg["segmentation"]["method"]
    paths["run"].mkdir(parents=True, exist_ok=True)
    reuse = cfg["segmentation"].get("reuse_from")
    if reuse:
        out_dir.mkdir(parents=True, exist_ok=True)
        t = segment_reuse(cfg, clip, frames, out_dir)
        record_timing(paths["timing"], "segment", t["seconds"], t["frames"])
        print(f"  [segment] {clip}: {len(frames)} masks (reused from {reuse})")
        return

    prompt = resolve_prompt(paths, click) if method != "oracle" else {"source": "oracle"}
    paths["used_prompt"].write_text(json.dumps(prompt, indent=2))

    with timed(paths["timing"], "segment", len(frames)):
        if method == "sam2":
            segment_sam2(cfg, frames, prompt, out_dir)
        elif method == "oracle":
            segment_oracle(paths, frames, out_dir)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    print(f"  [segment] {clip}: {len(frames)} masks ({method}, prompt from {prompt['source']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--click", help="X,Y pixel on frame 0")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    click = [int(v) for v in args.click.split(",")] if args.click else None
    run(load_config(args.config), args.clip, click, args.overwrite)


if __name__ == "__main__":
    main()
