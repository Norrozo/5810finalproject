"""I4: ProPainter (Zhou et al., ICCV 2023), a learned video inpainter.

ProPainter does what our flow-guided method (I3) does, but with learned parts:
  1. RAFT optical flow (learned, more accurate than Farneback)
  2. a recurrent network that completes the flow inside the hole
  3. propagation of real pixels along the completed flow, in feature space and image space
  4. a transformer that hallucinates plausible content for regions no frame ever saw,
     looking at nearby frames and a sparse set of distant reference frames
Step 4 is what our method lacks (it falls back to Telea), so this should remove most
of the remaining blur.

ProPainter's dependencies can conflict with SAM 2's, so it runs in its own virtualenv
(envs/propainter.txt, default .venv-propainter/) as a subprocess. The two only share files.
Weights (~200 MB) download automatically to weights/ on the first run.

ProPainter processes frames at a size divisible by 8 and resizes back, which slightly
blurs the whole frame. To keep pixels outside the hole untouched (as every other method
does), we paste ProPainter's output back only inside its dilated mask.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.utils import list_frames, read_image, read_mask, write_image

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "third_party" / "ProPainter" / "inference_propainter.py"


def inpaint_clip(frames_dir: Path, masks_dir: Path, out_dir: Path, inpaint_cfg: dict) -> int:
    frames = list_frames(frames_dir)
    python = REPO / inpaint_cfg.get("propainter_python", ".venv-propainter/bin/python")
    dilation = inpaint_cfg.get("mask_dilation", 4)
    if not SCRIPT.exists():
        raise FileNotFoundError(f"{SCRIPT} missing: run `git submodule update --init`")

    tmp = Path(tempfile.mkdtemp(prefix="propainter_"))
    try:
        cmd = [str(python), str(SCRIPT),
               "--video", str(Path(frames_dir).resolve()),
               "--mask", str(Path(masks_dir).resolve()),
               "--output", str(tmp),
               "--mask_dilation", str(dilation),
               "--neighbor_length", str(inpaint_cfg.get("neighbor_length", 10)),
               "--ref_stride", str(inpaint_cfg.get("ref_stride", 10)),
               "--subvideo_length", str(inpaint_cfg.get("subvideo_length", 80)),
               "--save_frames"]
        if inpaint_cfg.get("fp16", False):
            cmd.append("--fp16")
        env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
        subprocess.run(cmd, cwd=REPO, env=env, check=True)  # cwd=REPO -> weights go to ./weights

        # ProPainter names outputs 0000.png, 0001.png, ... under <output>/<input folder name>/frames/
        results = sorted((tmp / Path(frames_dir).name / "frames").glob("*.png"))
        if len(results) != len(frames):
            raise RuntimeError(f"ProPainter wrote {len(results)} frames, expected {len(frames)}")

        # Same dilation ProPainter applied (scipy binary_dilation = 3x3 cross, `dilation` times)
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for f, r in zip(frames, results):
            img = read_image(f)
            region = cv2.dilate(read_mask(Path(masks_dir) / f.name).astype(np.uint8), cross,
                                iterations=dilation) > 0
            out = img.copy()
            out[region] = read_image(r)[region]
            write_image(out_dir / f.name, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return len(frames)
