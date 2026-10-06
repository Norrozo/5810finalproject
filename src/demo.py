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
import shutil
from pathlib import Path

import cv2
import gradio as gr

from src import extract_frames, grounding, inpaint, refine_masks, segment, visualize
from src.utils import clip_paths, list_frames, load_config, read_image, read_mask

# Fill methods offered in the demo -> the config whose inpainting settings they use.
METHODS = {
    "Flow-guided, our method (fast)": "configs/I3b.yaml",
    "ProPainter (best, ~2 min)": "configs/I4.yaml",
    "OpenCV Telea (baseline)": "configs/B0.yaml",
}
GROUNDING_CONFIG = "configs/I2.yaml"


def demo_config(method: str, use_text: bool) -> dict:
    cfg = copy.deepcopy(load_config(METHODS[method]))
    cfg["version"] = "demo"
    cfg["segmentation"].pop("reuse_from", None)
    cfg["segmentation"]["method"] = "sam2"
    if use_text:
        cfg["grounding"] = load_config(GROUNDING_CONFIG)["grounding"]
    else:
        cfg.pop("grounding", None)
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


def on_click(name: str | None, evt: gr.SelectData):
    """Mark the clicked point on frame 0 and remember it."""
    if not name:
        return None, None
    x, y = evt.index
    first = read_image(list_frames(clip_paths(load_config("configs/B0.yaml"), name)["frames"])[0])
    cv2.circle(first, (x, y), 8, (0, 255, 0), -1)
    cv2.circle(first, (x, y), 9, (255, 255, 255), 2)
    return cv2.cvtColor(first, cv2.COLOR_BGR2RGB), [int(x), int(y)]


def run_removal(name: str | None, text: str, click: list[int] | None, method: str):
    if not name:
        raise gr.Error("Upload a video first.")
    text = (text or "").strip()
    if not text and not click:
        raise gr.Error("Type what to remove, or click on it in the first frame.")
    cfg = demo_config(method, use_text=bool(text))
    paths = clip_paths(cfg, name)
    shutil.rmtree(paths["run"], ignore_errors=True)  # fresh outputs every run

    note = ""
    if text:
        res = grounding.run(cfg, name, text=text, overwrite=True)
        if res["box"] is None:
            raise gr.Error(f"Couldn't find '{text}' in the first frame. Try another word, or click on it.")
        note = f"Found '{text}' (confidence {res['score']:.2f}). "
        grounding.unload_model()
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
    return str(paths["run"] / "result.mp4"), str(paths["run"] / "side_by_side.mp4"), overlay, note


def build_app():
    with gr.Blocks(title="Magic Eraser") as app:
        gr.Markdown("# Magic Eraser\nRemove an object from a video. Type what to remove "
                    "(\"remove the dog\") **or** click on it in the first frame. "
                    "Clips are cut to the first 60 frames at 480p.")
        name = gr.State(None)
        click = gr.State(None)
        with gr.Row():
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

        video.change(on_upload, video, [first, name, click])
        first.select(on_click, [name], [first, click])
        go.click(run_removal, [name, text, click, method], [result, side, mask_view, status])
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Public gradio.live link (e.g. from Colab)")
    args = ap.parse_args(argv)
    build_app().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
