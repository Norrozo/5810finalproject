"""Stage 2: segment the object on frame 0 from a click and track it with SAM 2.

Prompt sources, in order of priority:
  0. if the config has a `grounding` section (I2+): the box from stage 1 (boxes.json)
  1. --click X,Y on the command line
  2. data/<clip>/prompt.json  ({"points": [[x, y], ...]})
  3. auto-click from the ground-truth mask on frame 0 (synthetic benchmark only),
     so evaluation runs need no manual clicking

B0 limitations (these are future iterations, not bugs):
  - one positive click on frame 0 only, no text prompt (I2)
  - no re-detection if the object leaves and comes back (I5)
  - single object (I7 adds plural text prompts: one SAM 2 object per box, masks merged)

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
                       record_timing, stage_done, timed, uses_grounding, write_mask)


# ---------------------------------------------------------------- prompts

def auto_click_from_mask(mask: np.ndarray) -> list[int]:
    """Point deepest inside the mask (max distance from its edge), like a careful human click."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return [int(x), int(y)]


def resolve_prompt(paths: dict, click: list[int] | None = None, use_box: bool = False) -> dict:
    if use_box:
        boxes = json.loads(paths["boxes"].read_text())
        prompt = {"frame": 0, "box": boxes["box"], "source": f"text: {boxes['text']}"}
        if boxes.get("multi"):  # I7: several objects, tracked separately, merged into one mask
            prompt["boxes"] = boxes["boxes"]
        return prompt
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

def whole_object_mask(predictor, frame: Path, prompt: dict) -> np.ndarray:
    """I8: one click is ambiguous (a shirt, the person, or the person and their bike?).
    SAM 2 answers with three candidate masks, roughly part / larger part / whole object,
    and by default keeps the one it is most confident about, which is often a part
    (real dev: a boxer's helmet, an upside-down person's legs). Here we keep the LARGEST
    candidate, i.e. the whole object, and start tracking from that mask instead of the click.
    Uses the video model's own weights through SAM 2's single-image interface.
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    image_predictor = SAM2ImagePredictor(predictor)
    image_predictor.set_image(cv2.cvtColor(read_image(frame), cv2.COLOR_BGR2RGB))
    candidates, _scores, _ = image_predictor.predict(
        point_coords=np.array(prompt["points"], dtype=np.float32),
        point_labels=np.array(prompt["labels"], dtype=np.int32), multimask_output=True)
    candidates = candidates > 0
    return candidates[int(np.argmax(candidates.sum(axis=(1, 2))))]


def segment_sam2(cfg: dict, frames: list[Path], prompts: list[dict]) -> list[np.ndarray]:
    """Track one object with SAM 2 and return a boolean mask per frame.

    `prompts` = one or more {"frame": k, "box": [x0, y0, x1, y1]} or
    {"frame": k, "points": [[x, y]], "labels": [1]}. B0/I2 pass one prompt on frame 0;
    I5 adds correction boxes on later frames, which SAM 2 treats as extra conditioning frames.
    I7: {"frame": 0, "boxes": [...]} gives each box its own SAM 2 object id, so the
    objects are tracked separately; the returned mask is their union.
    """
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

    h, w = read_image(frames[0]).shape[:2]
    masks = [np.zeros((h, w), bool) for _ in frames]  # frames SAM 2 skips stay empty
    # SAM 2's video loader expects a folder of JPEGs named 00000.jpg, 00001.jpg, ...
    tmp = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"{i:05d}.jpg"), read_image(f), [cv2.IMWRITE_JPEG_QUALITY, 95])

        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(tmp),
                                         offload_video_to_cpu=seg_cfg.get("offload_video_to_cpu", True))
            for p in prompts:
                if "boxes" in p:
                    for obj_id, box in enumerate(p["boxes"], start=1):
                        predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=p["frame"], obj_id=obj_id,
                            box=np.array(box, dtype=np.float32))
                elif "box" in p:
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=p["frame"], obj_id=1,
                        box=np.array(p["box"], dtype=np.float32))
                elif seg_cfg.get("click_mask") == "largest":
                    predictor.add_new_mask(inference_state=state, frame_idx=p["frame"], obj_id=1,
                                           mask=whole_object_mask(predictor, frames[p["frame"]], p))
                else:
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=p["frame"], obj_id=1,
                        points=np.array(p["points"], dtype=np.float32),
                        labels=np.array(p["labels"], dtype=np.int32),
                    )
            for idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                # union over objects (one object before I7)
                masks[idx] = (mask_logits > 0.0).any(dim=0).squeeze().cpu().numpy()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        # Free the model before the next clip; otherwise memory builds up across clips
        # and an MPS (16 GB Mac) run slows from ~2 s/frame to >10 s/frame through swapping.
        predictor = state = None  # drop the model and its per-video state (frame features)
        import gc
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    return masks


def redetect_corrections(masks: list[np.ndarray], keyframes: dict, rd_cfg: dict,
                         anchor_box: list[float] | None = None) -> list[dict]:
    """I5: decide where the tracker has lost the object, using detections on every N-th frame.

    On each keyframe k with a confident detection (score >= min_score), the track is
    corrected with the best detection if either
      - the tracked mask is (nearly) empty: the object was lost, e.g. after leaving the
        frame or a long occlusion, or
      - the tracked mask overlaps no detection (box IoU < match_iou): it drifted onto
        something that does not match the prompt. With match_all_candidates (I5b) every
        candidate counts here, not only the confident ones.
    A track that still sits on one of the detections is left alone, so a correct track
    is never switched to a look-alike.

    The correction is the detection nearest the object's last confirmed position (the
    frame-0 box, then each keyframe where the track matched a detection), not simply the
    top-scoring one: with several look-alikes (I2c asks the user which one), the top
    detection may be a different object than the one being removed.
    """
    from src.evaluate import box_iou, mask_to_box
    def nearest(dets, box):
        if box is None:
            return dets[0]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        dist = lambda d: ((d["box"][0] + d["box"][2]) / 2 - cx) ** 2 + ((d["box"][1] + d["box"][3]) / 2 - cy) ** 2
        return max(dets, key=lambda d: (box_iou(box, d["box"]), -dist(d)))

    corrections = []
    for k_str, cands in sorted(keyframes.items(), key=lambda kv: int(kv[0])):
        k = int(k_str)
        dets = [c for c in cands if c["score"] >= rd_cfg.get("min_score", 0.35)]
        if not dets:
            continue
        m = masks[k]
        # I5b: the track is confirmed if it sits on ANY candidate, even a low-scoring one.
        # (I5 only checked confident ones, so a correct track whose object scored low on
        # that frame was "corrected" onto a confident look-alike.)
        confirm = cands if rd_cfg.get("match_all_candidates") else dets
        if m.mean() < rd_cfg.get("min_area_frac", 0.001):
            reason = "lost"
        elif max(box_iou(mask_to_box(m), d["box"]) for d in confirm) < rd_cfg.get("match_iou", 0.3):
            reason = "drifted"
        else:
            anchor_box = max(confirm, key=lambda d: box_iou(mask_to_box(m), d["box"]))["box"]
            continue
        best = nearest(dets, anchor_box)
        anchor_box = best["box"]
        corrections.append({"frame": k, "box": best["box"], "score": best["score"], "reason": reason})
    return corrections


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

    use_box = uses_grounding(cfg)
    prompt = resolve_prompt(paths, click, use_box) if method != "oracle" else {"source": "oracle"}
    paths["used_prompt"].write_text(json.dumps(prompt, indent=2))

    with timed(paths["timing"], "segment", len(frames)):
        if method == "sam2" and use_box and prompt["box"] is None:
            # Grounding found nothing: remove nothing (counts as a grounding miss in evaluation)
            h, w = read_image(frames[0]).shape[:2]
            for f in frames:
                write_mask(out_dir / f.name, np.zeros((h, w), bool))
        elif method == "sam2":
            masks = segment_sam2(cfg, frames, [prompt])
            keyframes = json.loads(paths["boxes"].read_text()).get("keyframes") if use_box else None
            if keyframes:
                corrections = redetect_corrections(masks, keyframes, cfg["grounding"].get("redetect", {}),
                                                   anchor_box=prompt["box"])
                prompt["corrections"] = corrections
                paths["used_prompt"].write_text(json.dumps(prompt, indent=2))
                if corrections:
                    print(f"  [segment] {clip}: re-detection corrected frames "
                          f"{[c['frame'] for c in corrections]}, re-tracking")
                    masks = segment_sam2(cfg, frames, [prompt] + corrections)
            for f, m in zip(frames, masks):
                write_mask(out_dir / f.name, m)
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
