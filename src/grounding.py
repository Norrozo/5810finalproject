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

from src.utils import clip_paths, list_frames, load_config, read_image, read_mask, timed


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
            parts = line.split()
            # "name prompt words" (eval_prompts.txt) or "name davis_clip object_id prompt words" (real_clips.txt)
            prompts[parts[0]] = " ".join(parts[3:] if len(parts) >= 4 and parts[2].isdigit() else parts[1:])
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


@lru_cache(maxsize=1)
def load_clip(hf_model: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return CLIPProcessor.from_pretrained(hf_model), CLIPModel.from_pretrained(hf_model).to(device).eval(), device


def rerank_clip(img_bgr: np.ndarray, text: str, dets: list[dict], rr_cfg: dict) -> list[dict]:
    """I2b: re-rank detector candidates by how well each crop matches the WHOLE prompt.

    Grounding DINO (tiny) scores boxes mostly by the noun, so "man in black jacket"
    can pick the wrong man. CLIP was trained to match images to full captions, so it
    is better at attributes ("in black jacket") and at preferring one piglet over a
    box around the whole litter. Each box is cropped with 10% padding and scored by
    CLIP image-text similarity.

    The final score multiplies the detector's confidence by CLIP's probability over the
    candidates (softmax of the similarities), treating them as two independent pieces of
    evidence. When CLIP can't tell candidates apart (a generic prompt like "person" gives
    near-equal similarities), the detector's order wins; when CLIP is confident (an
    attribute match), CLIP wins. Pure CLIP ranking was tried first and failed on a dev
    clip for exactly that reason (two people, similarity 25.80 vs 25.72).
    """
    import torch
    from PIL import Image
    if len(dets) <= 1:
        return dets
    processor, model, device = load_clip(rr_cfg.get("hf_model", "openai/clip-vit-base-patch16"))
    h, w = img_bgr.shape[:2]
    crops = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        px, py = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
        x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
        x1, y1 = min(w, int(x1 + px) + 1), min(h, int(y1 + py) + 1)
        crops.append(Image.fromarray(cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
    phrase = clean_prompt(text).rstrip(".")
    inputs = processor(text=[f"a photo of a {phrase}"], images=crops, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        sims = model(**inputs).logits_per_image[:, 0].float().cpu().numpy()
    probs = np.exp(sims - sims.max())
    probs /= probs.sum()
    for d, s, p in zip(dets, sims, probs):
        d["clip"], d["clip_prob"] = round(float(s), 3), round(float(p), 4)
        d["final"] = round(d["score"] * float(p), 4)
    return sorted(dets, key=lambda d: -d["final"])


def is_ambiguous(dets: list[dict], ratio: float) -> bool:
    """I2c: the prompt is ambiguous when the runner-up box scores within `ratio` of the best.

    On the real dev set every top-1 miss had a close runner-up (e.g. 0.42 vs 0.44), and
    the right object was always among the top 3. ratio = 0.7 was chosen there.
    """
    return len(dets) > 1 and dets[1]["score"] >= ratio * dets[0]["score"]


def simulated_choice(dets: list[dict], gt_mask: np.ndarray) -> int:
    """Benchmark stand-in for the user's click: the candidate that overlaps the true object
    most. Only used when a clip is flagged ambiguous; reported as simulated."""
    from src.evaluate import box_iou, mask_to_box
    gt = mask_to_box(gt_mask)
    return int(np.argmax([box_iou(d["box"], gt) for d in dets]))


def unload_model() -> None:
    """Free Grounding DINO (and CLIP) before SAM 2 runs (together they swap on a 16 GB Mac)."""
    import gc
    import torch
    load_model.cache_clear()
    load_clip.cache_clear()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def find(img_bgr: np.ndarray, text: str, g_cfg: dict) -> list[dict]:
    """Candidates for the prompt, best first. With `rerank` (I2b), detect with a lower
    threshold to widen the pool, then let CLIP order it; otherwise as in I2."""
    rr = g_cfg.get("rerank")
    if not rr:
        return detect(img_bgr, text, g_cfg)
    pool = detect(img_bgr, text, {**g_cfg, "box_threshold": rr.get("pool_threshold", 0.2)})
    return rerank_clip(img_bgr, text, pool[: rr.get("pool_size", 10)], rr)


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
    frames = list_frames(paths["frames"])
    every = cfg["grounding"].get("redetect", {}).get("every")  # I5: also detect on every N-th frame
    paths["run"].mkdir(parents=True, exist_ok=True)
    with timed(paths["timing"], "ground", len(frames)):
        dets = find(read_image(frames[0]), text, cfg["grounding"])
        keyframes = {str(k): find(read_image(frames[k]), text, cfg["grounding"])[:5]
                     for k in range(every, len(frames), every)} if every else None
    best = dets[0] if dets else None
    result = {"text": text, "box": best["box"] if best else None,
              "score": best["score"] if best else None, "candidates": dets[:10]}
    dis = cfg["grounding"].get("disambiguate")  # I2c: ask the user when the prompt is ambiguous
    if dis and dets:
        choices = dets[: dis.get("max_choices", 3)]
        result["ambiguous"] = is_ambiguous(choices, dis.get("ratio", 0.7))
        result["choices"] = choices
        if result["ambiguous"] and dis.get("simulate_user") and paths["gt_masks"].exists():
            k = simulated_choice(choices, read_mask(paths["gt_masks"] / frames[0].name))
            result.update(box=choices[k]["box"], score=choices[k]["score"], user_choice=k, simulated=True)
    if keyframes is not None:
        result["keyframes"] = keyframes
    paths["boxes"].write_text(json.dumps(result, indent=2))
    if result.get("ambiguous"):
        print(f"  [ground]  {clip}: '{text}' AMBIGUOUS ({len(result['choices'])} close candidates)"
              + (f" -> simulated user picks #{result['user_choice'] + 1}" if "user_choice" in result else ""))
    elif best:
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
