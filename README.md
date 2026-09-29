# Magic Eraser: Object Removal in Video

CIS 5810 Final Project, Fall 2026 · Option 1

Point at an object in the first frame of a video, and the pipeline tracks it through every frame and fills in the background so it disappears.

This is the **B0 baseline**: a working MVP that is deliberately simple, so each iteration has a clear, measurable improvement to make.

## Pipeline

```
video frames ──► segment ──► refine masks ──► inpaint ──► visualize + evaluate
                 SAM 2        (binarize)       cv2.inpaint
                 from a click                  per frame
```

Each stage reads its input from disk and writes its output to disk (`runs/<version>/<clip>/`). A stage can therefore be swapped or re-run without recomputing the others, and interrupted Colab runs pick up where they left off.

## Quick start (Colab)

Open `notebooks/colab_B0.ipynb` in Colab with a T4 GPU and run the cells in order. The notebook installs everything, downloads DAVIS, builds the benchmark, runs B0, and prints the results table.

## Quick start (local, with an NVIDIA GPU)

```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"

# 1. Download DAVIS 2017 TrainVal 480p from davischallenge.org and unzip it to ./DAVIS
# 2. Build the synthetic benchmark
python -m src.make_synthetic --config configs/B0.yaml --davis-root DAVIS
# 3. Run the baseline on all benchmark clips
python -m src.run_pipeline --config configs/B0.yaml
# 4. Print the results table
python -m src.summarize
```

Test that everything except SAM 2 works, with no GPU and no downloads:

```bash
python -m tests.smoke_test
```

### Your own video

```bash
python -m src.run_pipeline --config configs/B0.yaml --video my_clip.mp4 --name my_clip --click 320,240
```

`--click` is an (x, y) pixel on the object in the first frame. Clips are cut to `max_frames` and resized to `max_height` from the config, to fit in Colab's GPU memory.

## Evaluation

Real videos have no ground truth for what's behind an object, so we build a **synthetic benchmark**. We take an object from one DAVIS video (using its annotation mask) and paste it onto a different DAVIS video. A perfect removal gives back the untouched background video. The ten test clips are listed in `configs/eval_clips.txt` and **stay fixed across all iterations**.

| Metric | Measures | Better |
|---|---|---|
| J, F, J&F | Mask overlap (IoU) and boundary accuracy vs. the paste mask | higher |
| PSNR | Output vs. clean background, whole frame | higher |
| PSNR (mask) | Same, but only inside the object region, where the hard part is | higher |
| SSIM | Structural similarity, whole frame | higher |
| Flicker | Change inside the region between consecutive frames, after motion alignment | lower |
| Flicker ref | Same measure on the clean video: the lowest flicker you can expect | — |
| s/frame | Runtime per frame, per stage and total | lower |

For benchmark clips, the click is placed automatically at the center of the object's ground-truth mask on frame 0, so evaluation is repeatable and needs no manual input.

Setting `segmentation.method: oracle` in a config uses ground-truth masks instead of SAM 2. This measures inpainting quality on its own, separate from segmentation errors. It's useful for comparing inpainting methods fairly.

Results accumulate in `results/results.csv`, with one row per (version, clip). Re-running a version replaces its rows.

## Known limitations of B0 (the iteration roadmap)

B0 works end to end, but it has clear weaknesses. Each one maps to a planned iteration:

| Weakness | Where it shows up | Planned fix |
|---|---|---|
| Masks hug the object tightly, so edge pixels and attached shadows survive | Visible halos; lower PSNR (mask) | **I1**: mask dilation and hole filling in `src/refine_masks.py` |
| Needs a manual click | Can't say "remove the dog" | **I2**: text prompts with Grounding DINO |
| Each frame is filled on its own | High flicker compared with the reference | **I3**: our own flow-guided inpainting |
| Fill only uses border colors, so large holes blur | Low PSNR (mask); smeared stills | **I4**: ProPainter |
| Prompt is on frame 0 only | Loses objects that leave and re-enter | **I5**: periodic re-detection |
| SAM 2 large model at full resolution | Seconds per frame | **I6**: smaller model, lower resolution, chunking |
| One object at a time | Can't remove "all the people" | **I7**: multiple objects + failure study |

The benchmark also has its own limitation: pasted objects have hard edges and no shadows, so they're easier to segment than real objects. Evaluating segmentation on the original DAVIS videos as well would be a sensible addition.

## Adding an iteration

1. Copy `configs/B0.yaml` to `configs/I1.yaml`, set `version: I1`, and change **one thing**.
2. Put new code behind a config option, so older versions still reproduce exactly.
3. Run `python -m src.run_pipeline --config configs/I1.yaml`, then `python -m src.summarize`.
4. Record the change, the command, the results, and what you learned in `docs/experiment_log.md`.

## Repository layout

```
configs/        B0.yaml (one file per iteration), eval_clips.txt (frozen test set)
src/
  make_synthetic.py   builds the benchmark from DAVIS
  extract_frames.py   video file → frames
  segment.py          SAM 2 tracking from a click (or oracle GT masks)
  refine_masks.py     mask cleanup (identity in B0)
  inpaint/            opencv.py (B0); flow_guided.py and propainter.py come later
  evaluate.py         metrics → results/results.csv
  visualize.py        side_by_side.mp4 + stills.png per clip
  run_pipeline.py     runs all stages
  summarize.py        per-version Markdown table for reports
notebooks/      Colab notebook (calls src/, no core logic)
tests/          smoke test on fake data
docs/           experiment log
results/        results.csv (committed)
data/, runs/    generated; git-ignored
```

## Attribution

- **SAM 2**: Ravi et al., *SAM 2: Segment Anything in Images and Videos*, Meta AI, 2024. https://github.com/facebookresearch/sam2 (Apache 2.0)
- **DAVIS 2017**: Pont-Tuset et al., *The 2017 DAVIS Challenge on Video Object Segmentation*, 2017. https://davischallenge.org
- **OpenCV inpainting**: Telea, *An Image Inpainting Technique Based on the Fast Marching Method*, 2004; Bertalmio et al., *Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting*, 2001.
- Evaluation metric definitions follow the DAVIS benchmark (J and F).

## Generative AI use

See `AI_USAGE.md`, as required by the course.
