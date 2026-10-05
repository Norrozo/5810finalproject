"""Build the synthetic removal benchmark from DAVIS 2017.

Real videos have no "what's behind the object" ground truth. So we make our own:
take an object (using its DAVIS annotation mask) from one video and paste it onto
a different video. Removing it should give back the untouched background video.

For each clip in configs/eval_clips.txt this writes:
    data/<name>/frames/    composite frames (background + pasted object)  -> pipeline input
    data/<name>/gt_masks/  exact paste masks                              -> segmentation GT
    data/<name>/gt_clean/  original background frames                     -> removal GT

Usage:
    python -m src.make_synthetic --config configs/B0.yaml --davis-root DAVIS
    python -m src.make_synthetic --config configs/B0.yaml --davis-root DAVIS --clips-file configs/dev_clips.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils import clip_paths, load_config, read_clip_list, write_image, write_mask


def load_davis_clip(davis_root: Path, clip: str) -> list[Path]:
    frames = sorted((davis_root / "JPEGImages" / "480p" / clip).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames for DAVIS clip '{clip}' under {davis_root}")
    return frames


def load_object_mask(davis_root: Path, clip: str, frame_name: str, object_id: int) -> np.ndarray:
    # DAVIS annotations are palette PNGs: pixel value = object id. Read with PIL, not cv2,
    # because cv2 would convert the palette to colors.
    ann = np.array(Image.open(davis_root / "Annotations" / "480p" / clip / f"{frame_name}.png"))
    return ann == object_id


def build_clip(davis_root: Path, entry: dict, cfg: dict) -> int:
    obj_frames = load_davis_clip(davis_root, entry["object_clip"])
    bg_frames = load_davis_clip(davis_root, entry["background_clip"])
    n = min(len(obj_frames), len(bg_frames), cfg["max_frames"])
    paths = clip_paths(cfg, entry["name"])

    for i in range(n):
        bg = cv2.imread(str(bg_frames[i]))
        obj = cv2.imread(str(obj_frames[i]))
        mask = load_object_mask(davis_root, entry["object_clip"], obj_frames[i].stem, entry["object_id"])

        h, w = bg.shape[:2]
        if obj.shape[:2] != (h, w):
            obj = cv2.resize(obj, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        # Hard paste (no feathering) so the GT mask is exact.
        composite = np.where(mask[..., None], obj, bg)

        name = f"{i:05d}.png"
        write_image(paths["frames"] / name, composite)
        write_image(paths["gt_clean"] / name, bg)
        write_mask(paths["gt_masks"] / name, mask)

        if i == 0 and mask.sum() == 0:
            print(f"  WARNING: object not visible in frame 0 of {entry['name']}; "
                  f"the auto-click prompt will fail. Pick a different pair.")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--davis-root", required=True, help="Folder containing JPEGImages/ and Annotations/")
    ap.add_argument("--clips-file", help="Clip list to build (default: eval_clips from the config)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    davis_root = Path(args.davis_root)
    entries = [e for e in read_clip_list(args.clips_file or cfg["eval_clips"]) if "object_clip" in e]
    for entry in tqdm(entries, desc="Building synthetic clips"):
        n = build_clip(davis_root, entry, cfg)
        tqdm.write(f"  {entry['name']}: {n} frames")


if __name__ == "__main__":
    main()
