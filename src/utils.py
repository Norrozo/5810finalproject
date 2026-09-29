"""Shared helpers: config loading, per-clip paths, image/mask I/O, and stage timing."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------- config / paths

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def clip_paths(cfg: dict, clip: str) -> dict[str, Path]:
    """All paths for one clip.

    data/<clip>/           inputs shared by every version (frames, ground truth)
    runs/<version>/<clip>/ outputs of one pipeline version (never shared)
    """
    data = Path(cfg["data_root"]) / clip
    run = Path(cfg["runs_root"]) / cfg["version"] / clip
    return {
        "frames": data / "frames",
        "gt_masks": data / "gt_masks",      # only for synthetic benchmark clips
        "gt_clean": data / "gt_clean",      # only for synthetic benchmark clips
        "prompt": data / "prompt.json",     # optional manual click for custom videos
        "run": run,
        "masks": run / "masks",
        "masks_refined": run / "masks_refined",
        "inpainted": run / "inpainted",
        "timing": run / "timing.json",
        "used_prompt": run / "prompt.json",
    }


def read_clip_list(path: str | Path) -> list[dict]:
    """Parse configs/eval_clips.txt (name, object_clip, object_id, background_clip)."""
    clips = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        entry = {"name": parts[0]}
        if len(parts) >= 4:
            entry.update(object_clip=parts[1], object_id=int(parts[2]), background_clip=parts[3])
        clips.append(entry)
    return clips


# ---------------------------------------------------------------- image / mask I/O
# Images stay in OpenCV's BGR order everywhere in this repo.

def list_frames(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).glob("*.png"))


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Load a mask PNG as a boolean array (True = remove)."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > threshold


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def stage_done(out_dir: Path, n_frames: int) -> bool:
    """True if a stage already wrote all its frames (lets interrupted Colab runs resume)."""
    return out_dir.exists() and len(list_frames(out_dir)) >= n_frames


# ---------------------------------------------------------------- timing

def record_timing(path: Path, stage: str, seconds: float, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[stage] = {
        "seconds": round(seconds, 3),
        "frames": n_frames,
        "sec_per_frame": round(seconds / max(n_frames, 1), 4),
    }
    path.write_text(json.dumps(data, indent=2))


@contextmanager
def timed(path: Path, stage: str, n_frames: int):
    start = time.perf_counter()
    yield
    record_timing(path, stage, time.perf_counter() - start, n_frames)
