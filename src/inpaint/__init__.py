"""Stage 4: inpainting. Dispatches to the method named in the config.

Usage:
    python -m src.inpaint --config configs/B0.yaml --clip dog_on_soapbox
"""
from __future__ import annotations

from src.utils import clip_paths, list_frames, stage_done, timed


def run(cfg: dict, clip: str, overwrite: bool = False) -> None:
    paths = clip_paths(cfg, clip)
    n = len(list_frames(paths["frames"]))
    out_dir = paths["inpainted"]
    if not overwrite and stage_done(out_dir, n):
        print(f"  [inpaint] {clip}: already done, skipping")
        return

    inpaint_cfg = cfg["inpaint"]
    method = inpaint_cfg["method"]
    with timed(paths["timing"], "inpaint", n):
        if method == "opencv":
            from src.inpaint.opencv import inpaint_clip
            inpaint_clip(paths["frames"], paths["masks_refined"], out_dir, inpaint_cfg)
        elif method == "flow_guided":
            raise NotImplementedError("Flow-guided inpainting is iteration I3: implement src/inpaint/flow_guided.py")
        elif method == "propainter":
            raise NotImplementedError("ProPainter is iteration I4: add a wrapper in src/inpaint/propainter.py")
        else:
            raise ValueError(f"Unknown inpaint method: {method}")
    print(f"  [inpaint] {clip}: {n} frames ({method})")
