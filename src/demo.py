"""Stage 6: Gradio demo. Upload a video, say what to remove (text or a click), get the result.

It runs the same stage scripts as the benchmark, with the settings of an existing
config (so the demo shows exactly what the reports measure). Outputs go to
runs/demo/<clip>/ and are overwritten on every run.

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.demo          # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src import extract_frames, grounding, inpaint, refine_masks, segment, visualize
from src.utils import clip_paths, list_frames, load_config, read_image, read_mask

# Fill methods offered in the demo -> the config whose inpainting settings they use.
METHODS = {
    "Flow-guided, our method (fast)": "configs/I3b.yaml",
    "ProPainter (best, ~2 min)": "configs/I4.yaml",
    "OpenCV Telea (baseline)": "configs/B0.yaml",
}
GROUNDING_CONFIG = "configs/I7.yaml"   # text prompts: asks "which one?" when ambiguous (I2c);
                                       # plural prompts ("the pigs") remove all of them (I7)
CLICK_CONFIG = "configs/I8.yaml"       # clicks select the whole object, not a part
EXAMPLES_DIR = Path("showcase")  # pre-rendered before/after videos (git-ignored), shown instantly in the demo


def demo_config(method: str, use_text: bool) -> dict:
    cfg = copy.deepcopy(load_config(METHODS[method]))
    cfg["version"] = "demo"
    cfg["segmentation"].pop("reuse_from", None)
    cfg["segmentation"]["method"] = "sam2"
    if use_text:
        cfg["grounding"] = load_config(GROUNDING_CONFIG)["grounding"]
    else:
        cfg.pop("grounding", None)
        cfg["segmentation"]["click_mask"] = load_config(CLICK_CONFIG)["segmentation"]["click_mask"]
    return cfg


def clip_name(video_path: str) -> str:
    return "demo_" + hashlib.md5(Path(video_path).read_bytes()).hexdigest()[:8]


def on_upload(video_path: str | None):
    """Extract frames once per video and show frame 0 for clicking."""
    if not video_path:
        return None, None, None
    cfg = load_config("configs/B0.yaml")
    name = clip_name(video_path)
    frames_dir = clip_paths(cfg, name)["frames"]
    if not list_frames(frames_dir):
        extract_frames.extract(video_path, frames_dir, cfg["max_frames"], cfg["max_height"])
    first = cv2.cvtColor(read_image(list_frames(frames_dir)[0]), cv2.COLOR_BGR2RGB)
    return first, name, None


def first_frame(name: str) -> np.ndarray:
    return read_image(list_frames(clip_paths(load_config("configs/B0.yaml"), name)["frames"])[0])


def draw_choices(name: str, choices: list[dict], picked: int | None = None) -> np.ndarray:
    """Frame 0 with numbered candidate boxes; the picked one in green."""
    img = first_frame(name)
    for i, c in enumerate(choices):
        x0, y0, x1, y1 = (int(v) for v in c["box"])
        color = (0, 220, 0) if i == picked else (0, 165, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
        cv2.rectangle(img, (x0, y0), (x0 + 30, y0 + 30), color, -1)
        cv2.putText(img, str(i + 1), (x0 + 8, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def on_click(name: str | None, choices: list[dict] | None, evt: gr.SelectData):
    """A click either answers "which one?" (picks the candidate box under it)
    or, with no question pending, marks the object to remove."""
    if not name:
        return None, None, None, ""
    x, y = evt.index
    if choices:
        inside = [i for i, c in enumerate(choices)
                  if c["box"][0] <= x <= c["box"][2] and c["box"][1] <= y <= c["box"][3]]
        if not inside:
            return draw_choices(name, choices), None, None, "Click inside one of the numbered boxes."
        area = lambda i: (choices[i]["box"][2] - choices[i]["box"][0]) * (choices[i]["box"][3] - choices[i]["box"][1])
        k = min(inside, key=area)  # the smallest box under the click
        return (draw_choices(name, choices, k), None, choices[k]["box"],
                f"Picked #{k + 1}. Press **Remove it** to continue.")
    first = first_frame(name)
    cv2.circle(first, (x, y), 8, (0, 255, 0), -1)
    cv2.circle(first, (x, y), 9, (255, 255, 255), 2)
    return cv2.cvtColor(first, cv2.COLOR_BGR2RGB), [int(x), int(y)], None, ""


def run_removal(name: str | None, text: str, click: list[int] | None, method: str,
                choices: list[dict] | None, chosen: list[float] | None):
    """Returns (result video, side-by-side, mask view, status, frame-0 image, choices, chosen)."""
    if not name:
        raise gr.Error("Upload a video first.")
    text = (text or "").strip()
    if not text and not click:
        raise gr.Error("Type what to remove, or click on it in the first frame.")
    cfg = demo_config(method, use_text=bool(text))
    paths = clip_paths(cfg, name)

    note = ""
    if text and chosen:
        # The user answered "which one?": keep the earlier grounding, use their box.
        res = json.loads(paths["boxes"].read_text())
        res.update(box=chosen, user_choice=[c["box"] for c in res["choices"]].index(chosen))
        for sub in ("masks", "masks_refined", "inpainted"):
            shutil.rmtree(paths[sub], ignore_errors=True)
        paths["boxes"].write_text(json.dumps(res, indent=2))
        note = f"Removing the {grounding.clean_prompt(text).rstrip('.')} you picked (#{res['user_choice'] + 1}). "
    elif text:
        shutil.rmtree(paths["run"], ignore_errors=True)  # fresh outputs every run
        res = grounding.run(cfg, name, text=text, overwrite=True)
        grounding.unload_model()
        if res["box"] is None:
            raise gr.Error(f"Couldn't find '{text}' in the first frame. Try another word, or click on it.")
        if res.get("ambiguous"):
            msg = (f"**{len(res['choices'])} things match '{text}'.** Click the one to remove "
                   "on the image at the left, then press **Remove it** again.")
            return None, None, None, msg, draw_choices(name, res["choices"]), res["choices"], None
        if res.get("multi"):
            note = f"Found {len(res['boxes'])} × {grounding.clean_prompt(text).rstrip('.')}; removing all of them. "
        else:
            note = f"Found the {grounding.clean_prompt(text).rstrip('.')} (confidence {res['score']:.2f}). "
    else:
        shutil.rmtree(paths["run"], ignore_errors=True)
    segment.run(cfg, name, click=None if text else click, overwrite=True)
    refine_masks.run(cfg, name, overwrite=True)
    inpaint.run(cfg, name, overwrite=True)
    visualize.run(cfg, name)

    # Result-only video (H.264) next to the side-by-side one
    rows = [read_image(f) for f in list_frames(paths["inpainted"])]
    visualize.write_video(paths["run"] / "result.mp4", rows, fps=12)

    first_mask = read_mask(list_frames(paths["masks"])[0])
    first = read_image(list_frames(paths["frames"])[0])
    overlay = cv2.cvtColor(visualize.overlay_mask(first, first_mask), cv2.COLOR_BGR2RGB)
    covered = 100 * first_mask.mean()
    note += f"Masked {covered:.1f}% of frame 0. Fill: {method}."
    return (str(paths["run"] / "result.mp4"), str(paths["run"] / "side_by_side.mp4"), overlay, note,
            gr.update(), None, None)


def build_app():
    with gr.Blocks(title="Magic Eraser") as app:
        gr.Markdown("# Magic Eraser\nRemove an object from a video. Type what to remove "
                    "(\"remove the dog\") **or** click on it in the first frame. "
                    "Clips are cut to the first 60 frames at 480p.")
        name = gr.State(None)
        click = gr.State(None)
        choices = gr.State(None)  # candidate boxes when a text prompt is ambiguous
        chosen = gr.State(None)   # the box the user picked among them
        examples = sorted(EXAMPLES_DIR.glob("[0-9]*.mp4")) if EXAMPLES_DIR.exists() else []
        if examples:
            # Pre-rendered results play instantly, so a presentation never waits on a live run.
            with gr.Tab("Examples"):
                for i in range(0, len(examples), 2):
                    with gr.Row():
                        for f in examples[i:i + 2]:
                            gr.Video(str(f), label=f.stem.split("_", 1)[-1], show_label=False, autoplay=False)
        with gr.Tab("Try it"), gr.Row():
            with gr.Column():
                video = gr.Video(label="1. Upload a video", sources=["upload"])
                first = gr.Image(label="2. ...or click the object here (optional)", interactive=False)
                text = gr.Textbox(label="2. What to remove", placeholder="remove the dog")
                method = gr.Radio(list(METHODS), value=list(METHODS)[0], label="3. Fill method")
                go = gr.Button("Remove it", variant="primary")
            with gr.Column():
                result = gr.Video(label="Result")
                status = gr.Markdown()
                mask_view = gr.Image(label="What got selected (frame 0)")
                side = gr.Video(label="Input | Mask | Result")

        reset = lambda: (None, None)
        video.change(on_upload, video, [first, name, click]).then(reset, None, [choices, chosen])
        text.change(reset, None, [choices, chosen])
        first.select(on_click, [name, choices], [first, click, chosen, status])
        go.click(run_removal, [name, text, click, method, choices, chosen],
                 [result, side, mask_view, status, first, choices, chosen])
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Public gradio.live link (e.g. from Colab)")
    args = ap.parse_args(argv)
    build_app().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
