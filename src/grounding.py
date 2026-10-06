"""Stage 1 (I2): find the object named by a text prompt on the first frame.

Grounding DINO (Liu et al., 2023) is an open-vocabulary detector: given an image and
free text ("dog", "black swan"), it returns boxes for things matching the text, with a
confidence score. We keep the single highest-scoring box (one object, see I7) and pass
it to SAM 2 as a box prompt instead of a click.

Text sources, in order of priority:
  1. --text on the command line ("remove the dog" works; "remove" is stripped)
  2. data/<clip>/prompt.json  ({"text": "dog"})
  3. the clip's line in grounding.prompts_file (configs/eval_prompts.txt for the benchmark)

Writes runs/<version>/<clip>/boxes.json:
  {"text": "dog", "box": [x0, y0, x1, y1] or null, "score": 0.83, "candidates": [...]}

Usage:
    python -m src.grounding --config configs/I2.yaml --clip dog_on_soapbox
    python -m src.grounding --config configs/I2.yaml --clip my_clip --text "remove the dog"
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.utils import clip_paths, list_frames, load_config, read_image, timed


def clean_prompt(text: str) -> str:
    """'Remove the dog!' -> 'dog.' (Grounding DINO expects lowercase phrases ending in '.')"""
    t = text.lower().strip()
    t = re.sub(r"^(please\s+)?(remove|erase|delete|get rid of)\s+", "", t)
    t = re.sub(r"^(the|a|an|all)\s+", "", t)
    t = t.strip(" .!?")
    return t + "."


def read_prompts_file(path: str | Path) -> dict[str, str]:
    prompts = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, text = line.split(maxsplit=1)
            prompts[name] = text
    return prompts


def resolve_text(cfg: dict, clip: str, text: str | None) -> str:
    if text:
        return text
    paths = clip_paths(cfg, clip)
    if paths["prompt"].exists():
        saved = json.loads(paths["prompt"].read_text())
        if "text" in saved:
            return saved["text"]
    prompts_file = cfg["grounding"].get("prompts_file")
    if prompts_file and clip in (prompts := read_prompts_file(prompts_file)):
        return prompts[clip]
    raise ValueError(f"No text prompt for {clip}: pass --text or add it to data/{clip}/prompt.json")


@lru_cache(maxsize=1)
def load_model(hf_model: str):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(hf_model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_model).to(device).eval()
    return processor, model, device


def unload_model() -> None:
    """Free Grounding DINO before SAM 2 runs (both together swap on a 16 GB Mac)."""
    import gc
    import torch
    load_model.cache_clear()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def detect(img_bgr: np.ndarray, text: str, g_cfg: dict) -> list[dict]:
    """All boxes matching `text`, best first: [{"box": [x0,y0,x1,y1], "score": s, "label": l}, ...]"""
    import torch
    from PIL import Image
    processor, model, device = load_model(g_cfg["hf_model"])
    image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=clean_prompt(text), return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=g_cfg.get("box_threshold", 0.35),
        text_threshold=g_cfg.get("text_threshold", 0.25), target_sizes=[image.size[::-1]])[0]
    dets = [{"box": [round(float(v), 1) for v in b], "score": round(float(s), 4), "label": str(l)}
            for b, s, l in zip(res["boxes"].cpu(), res["scores"].cpu(), res.get("text_labels", res["labels"]))]
    return sorted(dets, key=lambda d: -d["score"])


def run(cfg: dict, clip: str, text: str | None = None, overwrite: bool = False) -> dict:
    paths = clip_paths(cfg, clip)
    if not overwrite and paths["boxes"].exists():
        print(f"  [ground]  {clip}: already done, skipping")
        return json.loads(paths["boxes"].read_text())

    text = resolve_text(cfg, clip, text)
    first = read_image(list_frames(paths["frames"])[0])
    paths["run"].mkdir(parents=True, exist_ok=True)
    with timed(paths["timing"], "ground", len(list_frames(paths["frames"]))):
        dets = detect(first, text, cfg["grounding"])
    best = dets[0] if dets else None
    result = {"text": text, "box": best["box"] if best else None,
              "score": best["score"] if best else None, "candidates": dets[:5]}
    paths["boxes"].write_text(json.dumps(result, indent=2))
    if best:
        print(f"  [ground]  {clip}: '{text}' -> box {best['box']} (score {best['score']:.2f})")
    else:
        print(f"  [ground]  {clip}: '{text}' -> NO DETECTION")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--text")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.clip, args.text, args.overwrite)


if __name__ == "__main__":
    main()
