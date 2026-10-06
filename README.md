# Magic Eraser: Object Removal in Video

CIS 5810 Final Project, Fall 2026 · Option 1

Point at an object in the first frame of a video, and the pipeline tracks it through every frame and fills in the background so it disappears.

Status: **B0** (baseline), **I1** (mask refinement), **I2** (text prompts), **I3/I3b** (our flow-guided inpainting), **I4** (ProPainter), and the **demo** are done. See `docs/experiment_log.md` for every run.

| Version | Ground hit ↑ | J&F ↑ | PSNR ↑ | PSNR mask ↑ | SSIM ↑ | Flicker ↓ | s/frame ↓ |
|---|---|---|---|---|---|---|---|
| B0: click + SAM 2 + Telea | – | 0.943 | 26.955 | 16.388 | 0.947 | 18.753 | 1.379 |
| I1: + mask dilation | – | 0.943 | 26.052 | 15.507 | 0.942 | 18.918 | 1.380 |
| I2: text prompt (Grounding DINO) | 9/10 | 0.890 | 26.143 | 15.940 | 0.943 | 19.175 | 2.158* |
| I3: flow-guided fill | – | 0.943 | 26.544 | 15.977 | 0.942 | 16.301 | 1.503 |
| I3b: flow-guided, max 5 hops | – | 0.943 | 27.068 | 16.500 | 0.945 | 16.313 | 1.512 |
| I4: ProPainter | – | 0.943 | 31.706 | 21.141 | 0.969 | 9.215 | 5.429 |

\* run slowed by memory swapping; see the log.

(10 test clips; regenerate with `python -m src.summarize`.)

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

On a Mac with Apple Silicon, SAM 2 runs on the GPU via MPS. Prefix commands with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

Test that everything except SAM 2 works, with no GPU and no downloads:

```bash
python -m tests.smoke_test
```

### Demo: remove things by typing

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.demo     # open http://127.0.0.1:7860
```

Upload a video, type what to remove ("remove the dog") or click it in the first frame, and pick a fill method. Use `--share` on Colab for a public link. ProPainter needs its own environment: `python3 -m venv .venv-propainter && .venv-propainter/bin/pip install -r envs/propainter.txt`, plus `git submodule update --init`.

### Your own video

```bash
python -m src.run_pipeline --config configs/I2.yaml --video my_clip.mp4 --name my_clip --text "remove the dog"
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

J&F scores the raw SAM 2 masks; refinement (I1) is judged by the inpainting metrics only.

**Tuning uses a separate dev set.** `configs/dev_clips.txt` holds 5 pairs built from DAVIS *train* videos (the test set uses only val videos). `python -m src.sweep` tries several values of one config key on the dev set, with ground-truth masks, and writes to `results/tuning.csv`. Values are picked there and then run once on the test set.

Iterations that don't change segmentation set `segmentation.reuse_from: B0`, so every version is compared on the same SAM 2 masks.

## Known limitations of B0 (the iteration roadmap)

B0 works end to end, but it has clear weaknesses. Each one maps to a planned iteration:

| Weakness | Where it shows up | Planned fix |
|---|---|---|
| Masks hug the object tightly, so edge pixels and attached shadows survive | Visible halos; lower PSNR (mask) | **I1**: mask dilation and hole filling in `src/refine_masks.py` (done: hurt on this benchmark, which has no halos) |
| Needs a manual click | Can't say "remove the dog" | **I2**: text prompts with Grounding DINO (done: 9/10 found) |
| Each frame is filled on its own | High flicker compared with the reference | **I3**: our own flow-guided inpainting (done: −13% flicker) |
| Fill only uses border colors, so large holes blur | Low PSNR (mask); smeared stills | **I4**: ProPainter (done: +4.75 dB, flicker halved) |
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
configs/        one YAML per iteration, eval_clips.txt (frozen test set), dev_clips.txt (tuning set)
src/
  make_synthetic.py   builds the benchmark from DAVIS
  extract_frames.py   video file → frames
  segment.py          SAM 2 tracking from a click (or oracle GT masks)
  refine_masks.py     mask cleanup (identity in B0)
  inpaint/            opencv.py (B0), flow_guided.py (I3); propainter.py comes later
  sweep.py            tune one config value on the dev set
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
- **ProPainter**: Zhou et al., *ProPainter: Improving Propagation and Transformer for Video Inpainting*, ICCV 2023. https://github.com/sczhou/ProPainter (S-Lab License 1.0, non-commercial), used unmodified as a git submodule in `third_party/ProPainter`.
- **Grounding DINO**: Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*, 2023. Model `IDEA-Research/grounding-dino-tiny` via Hugging Face Transformers (Apache 2.0).
- **Gradio** for the demo UI.
- **DAVIS 2017**: Pont-Tuset et al., *The 2017 DAVIS Challenge on Video Object Segmentation*, 2017. https://davischallenge.org
- **Optical flow**: Farnebäck, *Two-Frame Motion Estimation Based on Polynomial Expansion*, 2003 (OpenCV implementation). The flow-guided method follows the general idea of flow-guided video inpainting (Xu et al., *Deep Flow-Guided Video Inpainting*, CVPR 2019): complete the flow, then propagate pixels along it. Our version is classical, with no learned components.
- **OpenCV inpainting**: Telea, *An Image Inpainting Technique Based on the Fast Marching Method*, 2004; Bertalmio et al., *Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting*, 2001.
- Evaluation metric definitions follow the DAVIS benchmark (J and F).

## Generative AI use

See `AI_USAGE.md`, as required by the course.
