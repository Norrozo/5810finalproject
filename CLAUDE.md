# CLAUDE.md

Instructions for Claude when working in this repository.

## Project

**Magic Eraser: Text-Driven Object Removal in Video**, a CIS 5810 (Fall 2026) final project, Option 1 (assembling existing modules into an application).

The user gives a video and a text prompt ("remove the dog"). The pipeline finds the object, segments and tracks it through every frame, and fills in the background so the object disappears without flicker.

Grading emphasizes **pipeline design, baseline results, and documented iterations**. Every change should be measurable against the baseline. A clean record of experiments matters as much as the final output.

## Pipeline

Each stage is a separate script that reads from disk and writes to disk. Do not merge stages into one process.

| Stage | Tool | Input | Output |
|---|---|---|---|
| 1. Grounding | Grounding DINO | first frame + text prompt | `boxes.json` |
| 2. Segmentation + tracking | SAM 2 video predictor | frames + box or click | `masks/*.png` |
| 3. Mask refinement | OpenCV morphology | raw masks | `masks_refined/*.png` |
| 4. Inpainting | `cv2.inpaint`, our flow-guided method, or ProPainter | frames + refined masks | `inpainted/*.png` |
| 5. Evaluation | our scripts | outputs + ground truth | row in `results/results.csv` |
| 6. Demo | Gradio | video upload + prompt | result video |

### File conventions

- Inputs shared by every version live in `data/<clip_name>/`:
  - `frames/00000.png, 00001.png, ...` are the pipeline input, zero-padded to 5 digits
  - `gt_masks/`, `gt_clean/` are the ground truth (synthetic benchmark clips only)
  - `prompt.json` is an optional manual click for custom videos
- Outputs of one version live in `runs/<version>/<clip_name>/`: `masks/`, `masks_refined/`, `inpainted/`, `timing.json`, `prompt.json`, `side_by_side.mp4`, `stills.png`.
- Masks are single-channel PNGs with values 0 (keep) or 255 (remove), named to match the frames.
- Images are BGR (OpenCV order) everywhere.
- **Never overwrite another version's outputs.** Each iteration writes to its own folder so before/after comparisons stay valid.
- All paths come from `src/utils.py: clip_paths()`. Don't build paths by hand.

### Testing

- `python -m tests.smoke_test` runs the whole pipeline on fake data with no GPU (oracle segmentation). Run it after any change to `src/`, and extend it when adding a stage or method.
- `segmentation.method: oracle` uses ground-truth masks, to evaluate inpainting separately from segmentation.

## Iterations

| Tag | Change |
|---|---|
| B0 | Click prompt + SAM 2 + `cv2.inpaint` (Telea), per frame |
| I1 | Mask dilation + hole filling |
| I2 | Text prompt via Grounding DINO |
| I3 | Our own flow-guided inpainting (Farneback optical flow + temporal pixel propagation, `cv2.inpaint` fallback) |
| I3b | I3 + hop cap (`max_hops: 5`, tuned on the dev set) |
| I4 | ProPainter |
| I5 | Re-detection every N frames for objects that leave and re-enter |
| I6 | Speed: smaller SAM 2 model, lower resolution, chunking |
| I7 | Multi-object prompts + failure-case study |

Rules:
- One iteration changes **one thing**. If a task mixes several changes, point this out and suggest splitting it.
- Iteration settings live in `configs/<tag>.yaml`, not in code edits to earlier versions. Older versions must stay reproducible.
- An iteration that doesn't improve results is still a valid result. Log it; don't delete it.

## Evaluation

- Fixed test set: the clip names in `configs/eval_clips.txt` (10–15 DAVIS 2017 validation clips). Do not change this list without being asked, since that invalidates comparisons.
- **Never tune on the test set.** Pick hyperparameters on `configs/dev_clips.txt` (DAVIS train videos) with `python -m src.sweep`, which writes to `results/tuning.csv`; then run the chosen value once on the test set.
- J&F scores the raw SAM 2 masks (`masks/`), not `masks_refined/`.
- Metrics, all written to `results/results.csv` as one row per (version, clip) by `src/evaluate.py`:
  - **J&F**: region IoU and boundary F-measure, implemented in `src/evaluate.py` following the DAVIS definitions.
  - **Grounding hit rate**: prompt box IoU > 0.5 with the ground-truth box on frame 0.
  - **PSNR / PSNR (mask) / SSIM** on the synthetic benchmark: a DAVIS object pasted onto another DAVIS clip, removed, then compared to the untouched clip. PSNR (mask) scores only the object region.
  - **Flicker**: mean absolute difference between consecutive filled frames inside the region, after flow alignment. `flicker_ref` is the same measure on the clean video.
  - **Runtime**: seconds per frame for each stage.
- Results come only from running the scripts. **Never write, estimate, or edit numbers in `results/` by hand**, and never put made-up numbers in reports or docs.

## Environment

- Python. Runs on Google Colab with a free-tier GPU (about 16 GB VRAM) or a local GPU.
- SAM 2 and ProPainter may need conflicting dependency versions. Keep them in **separate environments** (`envs/sam2.txt`, `envs/propainter.txt`) and connect them only through files on disk. Don't try to resolve conflicts by installing both into one environment.
- Keep clips short (5–10 s) at about 480p. ProPainter runs out of memory on long or high-resolution clips.
- On a 16 GB Mac (MPS): run one heavy model at a time. SAM 2, Grounding DINO and ProPainter together swap to disk and slow down 5–10x. Stages free their models when done; don't run two pipelines at once.
- Colab sessions time out, so scripts should skip frames or clips whose outputs already exist, allowing an interrupted run to resume.

## Repository layout

```
configs/        per-iteration YAML configs, eval_clips.txt
src/
  grounding.py      stage 1
  segment.py        stage 2
  refine_masks.py   stage 3
  inpaint/
    opencv.py       B0 baseline
    flow_guided.py  our own method (I3)
    propainter.py   wrapper around third_party/ProPainter (I4)
  evaluate.py       all metrics
  visualize.py      side-by-side videos and stills
  run_pipeline.py   runs all stages for one config
  summarize.py      per-version Markdown table
  make_synthetic.py builds the synthetic benchmark
  extract_frames.py video file to frames
  demo.py           Gradio app (python -m src.demo)
tests/          smoke_test.py
notebooks/      Colab notebooks that call src/ scripts (no core logic here)
third_party/    external repos as git submodules, unmodified
data/, runs/, weights/   git-ignored
results/        results.csv, figures (committed)
docs/           experiment log, report drafts
AI_USAGE.md     generative-AI disclosure log
```

## Code style

- Every script has an `argparse` CLI and a `--config` option. No hard-coded paths.
- Short docstrings on functions; type hints where they help.
- Set random seeds wherever randomness is involved.
- Don't commit data, model weights, or videos. Keep `.gitignore` up to date.
- Third-party code stays in `third_party/`, is credited in `README.md`, and is wrapped rather than edited. If an edit is unavoidable, note it in the README.

## Working with the students

This is a class project, and the students must be able to explain every part of it to TAs and on Project Day.

- **Explain as you go.** When writing non-trivial code, especially `src/inpaint/flow_guided.py`, briefly explain the approach and any CV concepts involved (optical flow, warping, morphology, the metrics).
- Prefer clear code over clever code. The flow-guided method in particular should be readable enough to present in a report.
- Ask before large refactors or before changing the pipeline's file conventions.
- **AI disclosure is required by the course.** After any session where you write or substantially change code, add an entry to `AI_USAGE.md` with the date, which files changed, what you did, and what was taken directly from your output. Keep entries factual and brief.
- After running an experiment, add a short dated entry to `docs/experiment_log.md`: what changed, the command run, the results, and one line on what was learned. The reports are written from this log.

## Key deadlines

- 10/9: Proposal (B0 + I1)
- 11/20: Midterm report (I2–I4)
- 12/6: Project Day demo
- 12/10: Final report, code, demo video, README (I5–I7)
