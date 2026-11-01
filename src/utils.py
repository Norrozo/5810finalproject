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

def _merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        base[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config(path: str | Path, overlays: list[str | Path] | None = None) -> dict:
    """Load a version config, then apply overlays (e.g. configs/real_set.yaml) on top.

    Overlays change WHERE a version runs (clip list, results file, prompts), or swap in
    ground-truth masks to isolate one stage (configs/fill_only.yaml). They never change
    the method being measured, and always write to their own results CSV.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for o in overlays or []:
        with open(o) as f:
            _merge(cfg, yaml.safe_load(f))
    cfg["_config_path"] = str(path)
    return cfg


def uses_grounding(cfg: dict) -> bool:
    """Text prompts are on only when the config names a grounding method (I2 and later).
    An overlay may add grounding.prompts_file to any config without switching it on."""
    return bool(cfg.get("grounding", {}).get("method"))


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
        "boxes": run / "boxes.json",        # stage 1 output (text prompt -> box), I2+
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
            # object_id may be a list ("1,2,3") in the multi-object sets (I7); kept as text there
            obj = int(parts[2]) if parts[2].isdigit() else parts[2]
            entry.update(object_clip=parts[1], object_id=obj, background_clip=parts[3])
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
