# Experiment Log

One entry per experiment, including failed ones. The reports are written from this log.

Template:

```
## YYYY-MM-DD: <version>, <one-line description>
- Change: what is different from the previous version (one thing)
- Hypothesis: what we expect to improve, and why
- Command: python -m src.run_pipeline --config configs/<version>.yaml
- Results: paste `python -m src.summarize --versions <prev> <version>`
- Observations: what the stills/videos show; where it failed
- Takeaway: one line on what we learned and what to try next
```

---

## 2026-10-05: B0, baseline

- Change: n/a (first version). Click prompt → SAM 2 (`sam2.1-hiera-large`) → binarize → cv2.inpaint (Telea, radius 5), per frame. Click placed automatically at the deepest point of the frame-0 GT mask.
- Hardware: Apple M5 laptop, SAM 2 on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`), torch 2.14.1. Runtime numbers are for this machine, not Colab T4.
- Command: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.run_pipeline --config configs/B0.yaml`
- Results:

| Version | Clips | J&F ↑ | PSNR ↑ | PSNR mask ↑ | SSIM ↑ | Flicker ↓ | s/frame ↓ |
|---|---|---|---|---|---|---|---|
| B0 | 10 | 0.943 | 26.955 | 16.388 | 0.947 | 18.753 | 1.379 |

  Per clip (from `results/results.csv`): J&F is 0.96–0.99 on 9/10 clips; **libby_on_scooter-black = 0.595**. PSNR (mask) ranges 12.1 (libby) to 20.5 (goat). Flicker is 1.6–4.4× `flicker_ref` on every clip. Segmentation is ~97% of runtime (inpaint ≈ 0.04 s/frame).
- Observations:
  - Segmentation is mostly solved on this benchmark (pasted objects have hard edges), with one exception: in libby the pasted dog is split into two disconnected pieces by an occluder in its source mask, and a single click only tracks one piece.
  - Inpainting is the bottleneck. On large holes (dog_on_soapbox, camel) Telea smears border colors into streaks; the hidden background (the soapbox cart) can't be recovered from one frame. PSNR (mask) ≈ 14–15 on these.
  - Fill is computed per frame, so streaks change frame to frame → flicker well above the clean reference.
- Takeaway: B0 works end to end. Biggest gaps are inpainting quality on large holes (I3/I4) and temporal consistency (I3); single-click prompts fail on disconnected objects (motivates multi-click/I7). Next: I1 (mask dilation) for edge halos.

**Metric fix (2026-10-05, before I1):** `src/evaluate.py` now scores J&F on the raw SAM 2 masks (`masks/`) instead of `masks_refined/`. Refinement deliberately grows the mask past the object, which would lower J&F without tracking getting worse. B0's refine step is the identity, so B0's numbers are unchanged.

## 2026-10-05: I1, mask refinement (drop specks, fill holes, dilate 5 px)

- Change: `refine.min_blob_area: 100`, `fill_holes: true`, `dilate_px: 5`. SAM 2 masks reused from B0 (`segmentation.reuse_from: B0`), so B0 and I1 are compared on identical masks. Settings chosen up front (5 px ≈ ProPainter's default mask dilation), not tuned.
- Hypothesis: tight masks leave a halo ring of object pixels that gets smeared into the fill; dilating moves the halo inside the hole.
- Command: `python -m src.run_pipeline --config configs/I1.yaml`
- Results:

| Version | Clips | J&F ↑ | PSNR ↑ | PSNR mask ↑ | SSIM ↑ | Flicker ↓ | s/frame ↓ |
|---|---|---|---|---|---|---|---|
| B0 | 10 | 0.943 | 26.955 | 16.388 | 0.947 | 18.753 | 1.379 |
| I1 | 10 | 0.943 | 26.052 | 15.507 | 0.942 | 18.918 | 1.380 |

- Observations: worse on every inpainting metric, on every clip. The benchmark pastes objects with hard edges and no shadows, so there is no halo to remove; dilation only makes the hole bigger for Telea to blur.
- Takeaway: negative result, and evidence for the known benchmark limitation (no soft edges/shadows). Dilation should be re-checked on real (non-pasted) videos, where halos do exist. I3 builds on B0, not I1.

## 2026-10-05: I3, flow-guided inpainting (our own method)

- Change: `inpaint.method: flow_guided` (`src/inpaint/flow_guided.py`): Farneback flow in both directions → flow completion inside the hole (Telea on the flow field) → forward and backward pixel propagation → keep the source with fewer hops → Telea fallback for never-seen pixels. Masks reused from B0.
- Hypothesis: background hidden in frame t is visible in other frames; copying real pixels should be sharper than Telea and consistent over time (less flicker).
- Command: `python -m src.run_pipeline --config configs/I3.yaml`
- Results:

| Version | Clips | J&F ↑ | PSNR ↑ | PSNR mask ↑ | SSIM ↑ | Flicker ↓ | s/frame ↓ |
|---|---|---|---|---|---|---|---|
| B0 | 10 | 0.943 | 26.955 | 16.388 | 0.947 | 18.753 | 1.379 |
| I3 | 10 | 0.943 | 26.544 | 15.977 | 0.942 | 16.301 | 1.503 |

- Observations: flicker −13%, as hoped. PSNR (mask) slightly lower. Stills show real hidden background recovered (the cart's red barrel in dog_on_soapbox, people in breakdance_on_india) but also tearing and misplaced fragments.
- Diagnosis (scratch script, numbers on B0 masks): propagation fills 58–100% of the hole, but its PSNR on those pixels depends strongly on how many frames each pixel was copied through: dog_on_soapbox 18.8 dB at 1 hop → 12.6 dB at 8–16 hops; camel 15.8 → 11.9; breakdance 18.6 → 12.3. Pixels at the hole's edge were the most accurate, so edge contamination is not the main problem. Each hop adds flow error + bilinear blur.
- Takeaway: chained propagation drifts. Cap the number of hops (I3b).

## 2026-10-05: I3 hop-cap tuning on the DEV set

- To avoid tuning on the test set, built a separate dev set: `configs/dev_clips.txt`, 5 pairs from DAVIS 2017 *train* videos (the test set uses only val videos). Ground-truth (oracle) masks, so only the inpainting is being tuned.
- Command: `python -m src.make_synthetic --config configs/I3.yaml --davis-root DAVIS --clips-file configs/dev_clips.txt` then `python -m src.sweep --config configs/I3.yaml --key inpaint.max_hops --values 1 2 3 5 8 none` and `--key inpaint.method --values opencv`
- Results (`results/tuning.csv`):

| Version | Clips | J&F ↑ | PSNR ↑ | PSNR mask ↑ | SSIM ↑ | Flicker ↓ | s/frame ↓ |
|---|---|---|---|---|---|---|---|
| Telea (B0 inpainting) | 5 | 1.000 | 29.601 | 20.128 | 0.945 | 8.051 | 0.037 |
| max_hops 1 | 5 | 1.000 | 29.953 | 20.481 | 0.946 | 7.647 | 0.185 |
| max_hops 2 | 5 | 1.000 | 30.073 | 20.601 | 0.946 | 7.437 | 0.182 |
| max_hops 3 | 5 | 1.000 | 30.120 | 20.648 | 0.946 | 7.292 | 0.182 |
| max_hops 5 | 5 | 1.000 | 30.122 | 20.650 | 0.946 | 7.132 | 0.179 |
| max_hops 8 | 5 | 1.000 | 30.005 | 20.532 | 0.945 | 7.024 | 0.184 |
| no cap (I3) | 5 | 1.000 | 28.933 | 19.461 | 0.944 | 6.832 | 0.179 |

- Observations: same pattern as the test set: uncapped I3 loses to Telea on PSNR (mask). Capping wins on both PSNR and flicker. More hops → less flicker but lower accuracy. 3 and 5 tie on PSNR; 5 has lower flicker.
- Takeaway: choose max_hops = 5, then run once on the test set (I3b).

## 2026-10-05: I3b, flow-guided with max_hops = 5

- Change: I3 + `inpaint.max_hops: 5` (chosen on the dev set above, not the test set).
- Command: `python -m src.run_pipeline --config configs/I3b.yaml`
- Results:

| Version | Clips | J&F ↑ | PSNR ↑ | PSNR mask ↑ | SSIM ↑ | Flicker ↓ | s/frame ↓ |
|---|---|---|---|---|---|---|---|
| B0 | 10 | 0.943 | 26.955 | 16.388 | 0.947 | 18.753 | 1.379 |
| I1 | 10 | 0.943 | 26.052 | 15.507 | 0.942 | 18.918 | 1.380 |
| I3 | 10 | 0.943 | 26.544 | 15.977 | 0.942 | 16.301 | 1.503 |
| I3b | 10 | 0.943 | 27.068 | 16.500 | 0.945 | 16.313 | 1.512 |

- Observations: best version so far on PSNR, PSNR (mask) and flicker (−13% vs B0); SSIM 0.002 below B0. The PSNR gain over B0 is small (+0.1 dB). PSNR rewards blur (a smooth average is "close on average"), while a sharp fill that's off by a few pixels is punished, so visual stills matter here: I3b recovers real texture where B0 shows a smeared blob, but still has seams and misplaced fragments. Inpainting adds ~0.13 s/frame; SAM 2 still dominates runtime.
- Takeaway: flow-guided propagation helps, but large holes over moving, non-planar backgrounds need a learned model → I4 (ProPainter). Worth adding a perceptual metric (e.g. LPIPS) so the report isn't PSNR-only.

## 2026-10-05: I4, ProPainter

- Change: `inpaint.method: propainter` (Zhou et al., ICCV 2023; `third_party/ProPainter` submodule, own env `.venv-propainter` from `envs/propainter.txt`, called by `src/inpaint/propainter.py`). Masks reused from B0. ProPainter defaults except two settings forced by the 16 GB Mac: `fp16: true` (4x faster; PSNR mask 20.38 vs 20.44 dB fp32 on dog_on_soapbox frame 5) and `subvideo_length: 20` (default 80 swapped to disk, >10 min/clip; 20 = ~2 min/clip). On Colab's T4 the defaults should fit. Its own `mask_dilation: 4` is kept as part of the published method. Output is pasted back only inside ProPainter's dilated mask, so pixels outside the hole are untouched as in every other version.
- Hypothesis: learned flow completion + a transformer that hallucinates never-seen regions should remove the remaining blur that I3b falls back to Telea for.
- Command: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.run_pipeline --config configs/I4.yaml`
- Results:

| Version | Clips | J&F ↑ | PSNR ↑ | PSNR mask ↑ | SSIM ↑ | Flicker ↓ | s/frame ↓ |
|---|---|---|---|---|---|---|---|
| B0 | 10 | 0.943 | 26.955 | 16.388 | 0.947 | 18.753 | 1.379 |
| I3b | 10 | 0.943 | 27.068 | 16.500 | 0.945 | 16.313 | 1.512 |
| I4 | 10 | 0.943 | 31.706 | 21.141 | 0.969 | 9.215 | 5.429 |

- Observations: by far the biggest gain of any iteration: +4.75 dB PSNR (mask) over B0, flicker halved, and better on every clip. Flicker is at or below `flicker_ref` on 4 clips (blackswan, goat, cows, breakdance): ProPainter's fill is slightly smoother over time than the real video. Visually the fog is gone: dog_on_soapbox shows a clean ramp, road and cart. It cannot recover what no frame ever shows (the barrel behind the dog), so it fills that with plausible background. libby stays worst (13.25 dB) because its masks miss half the dog (J&F 0.595): no inpainter can fix a segmentation miss. Runtime ~4x B0 (inpainting ~1.5 s/frame on MPS fp16).
- Takeaway: ProPainter is the inpainting backbone for the final system. Our flow-guided method (I3b) remains the fast option and the "own method" contribution. Remaining errors are now dominated by segmentation (libby) → I2/I5/I7.

## 2026-10-05: I2, text prompts with Grounding DINO

- Change: stage 1 `src/grounding.py`. Grounding DINO (`IDEA-Research/grounding-dino-tiny`, Hugging Face default thresholds 0.35/0.25, not tuned) finds the object named in `configs/eval_prompts.txt` on frame 0; its top box replaces B0's click as SAM 2's prompt. Prompts were written once from each object's DAVIS video name ("dog", "black swan", "person", ...), before running. Inpainting identical to B0 (Telea), so differences come only from the prompt.
- New metrics: `grounding_iou` (box vs. GT mask's bounding box on frame 0) and `grounding_hit` (IoU > 0.5).
- Command: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.run_pipeline --config configs/I2.yaml`
- Results:

| Version | Clips | Ground hit ↑ | J&F ↑ | PSNR ↑ | PSNR mask ↑ | SSIM ↑ | Flicker ↓ | s/frame ↓ |
|---|---|---|---|---|---|---|---|---|
| B0 (click) | 10 | – | 0.943 | 26.955 | 16.388 | 0.947 | 18.753 | 1.379 |
| I2 (text) | 10 | 0.900 | 0.890 | 26.143 | 15.940 | 0.943 | 19.175 | 2.158 |

  Grounding IoU is 0.94–0.99 on 9 clips. **parkour_on_bike-packing: IoU 0.12, J&F 0.11**: "person" matched the cyclist who is already in the background video, not the pasted runner. **libby_on_scooter-black: J&F 0.595 → 0.953**: the box covers both disconnected halves of the dog, where B0's single click only selected one.
- Observations: text prompts work as well as a click when the noun is unambiguous, and better for objects split into pieces. They fail when the background contains another instance of the same class (two people). Runtime is inflated: this run swapped on the 16 GB Mac (SAM 2 slowed from ~2 to >10 s/frame over the run). Fixed afterwards by freeing Grounding DINO before SAM 2 and freeing SAM 2 after each clip (`grounding.unload_model`, end of `segment_sam2`); the s/frame number for I2 is not a fair speed comparison.
- Takeaway: text prompting is viable (9/10). The miss motivates I5 (re-detection) and I7 (multiple objects / more specific prompts, e.g. "person running"). Text and click can be combined in the final system (text first, click to correct).

## 2026-10-05: Demo (stage 6) and a real-video check

- Built `src/demo.py` (Gradio): upload a video, type "remove the bear" or click the object, choose the fill (I3b flow-guided, I4 ProPainter, or B0 Telea). Uses the same stage scripts and configs as the benchmark.
- Real DAVIS video `bear` (train split, not pasted), prompt "remove the bear": grounding score 0.96, SAM 2 mask clean on every frame.
  - Flow-guided fill: a blurry bear-shaped smudge. The camera follows the bear, so the background behind it is hidden in every frame; there is nothing to propagate and it falls back to Telea. A faint outline of the fur also survives (soft real edges, unlike the pasted benchmark objects).
  - ProPainter fill: better (rock and dirt texture continue into the hole) but still a soft smudge, because it must hallucinate a large region (~12% of the frame) that no frame shows.
- Takeaway: two things the synthetic benchmark hides: (1) objects tracked by the camera leave a permanently hidden region, the hardest case for every method; (2) real soft edges leave halos, so I1's dilation should be re-tested on real clips. Good material for the I7 failure-case study.

## 2026-10-29: Evaluation additions: LPIPS and RAFT-aligned flicker

- Change: two metrics added to `src/evaluate.py` via `src/metrics_deep.py`, promised in the proposal. **LPIPS mask** (AlexNet LPIPS on a crop around the object region, padded 16 px) because PSNR rewards blur. **Flicker RAFT**: the same flicker measure aligned with RAFT-large flow instead of Farneback, because our own method (I3b) uses Farneback and a Farneback-aligned metric could favor it. RAFT flow on the clean video is cached per clip (`data/<clip>/raft_flow.npy`). No pipeline was re-run; all versions were re-scored. Every pre-existing value in `results/results.csv` was checked unchanged (60/60 rows).
- Command: `python -m src.evaluate --config configs/<v>.yaml` for B0, I1, I2, I3, I3b, I4
- Results:

| Version | PSNR mask ↑ | LPIPS mask ↓ | Flicker (Farneback) ↓ | Flicker RAFT ↓ |
|---|---|---|---|---|
| B0 | 16.388 | 0.303 | 18.753 | 18.993 |
| I1 | 15.507 | 0.313 | 18.918 | 19.246 |
| I2 | 15.940 | 0.313 | 19.175 | 19.575 |
| I3 | 15.977 | 0.298 | 16.301 | 17.536 |
| I3b | 16.500 | 0.294 | 16.313 | 17.313 |
| I4 | 21.141 | 0.142 | 9.215 | 8.464 |

- Observations: (1) LPIPS confirms the "PSNR rewards blur" concern: uncapped I3 scores *lower* PSNR than B0 but *better* LPIPS (0.298 vs 0.303), i.e. it looks closer to the truth despite the PSNR drop. (2) The Farneback flicker metric did slightly favor our Farneback-based method: I3b's flicker reduction vs B0 is −13% with Farneback alignment but −9% with RAFT. The gain is real but smaller. (3) ProPainter halves LPIPS (0.142) and is the only method below B0's flicker under both aligners.
- Takeaway: report LPIPS and RAFT flicker as the primary perceptual/temporal metrics from now on; keep PSNR mask and Farneback flicker for continuity with B0–I4.

## 2026-10-29: Real-video test set (tracking only): B0 and I2

- Change: new test set `configs/real_clips.txt` (built by `src/make_real.py`): 10 unedited DAVIS 2017 val videos, scored by J&F against DAVIS's own mask of object 1, with a text prompt per clip. Chosen and prompted before any results, mixing easy cases with crowds, look-alikes and fast motion. No clean background exists, so no fill metrics. Run with the overlay `configs/real_set.yaml`, which only changes the clip list, prompts and results file (`results/real_results.csv`), never method settings.
- Command: `python -m src.run_pipeline --config configs/{B0,I2}.yaml --overlay configs/real_set.yaml`
- Results:

| Clip | Prompt | B0 J&F (click) | I2 J&F (text) | I2 box IoU |
|---|---|---|---|---|
| bike-packing | bicycle | 0.826 | 0.795 | 0.99 |
| dance-twirl | woman dancing | 0.938 | 0.941 | 0.96 |
| drift-chicane | car | 0.955 | 0.955 | 0.93 |
| india | woman in white | 0.918 | **0.000** | 0.00 |
| judo | judo fighter | 0.933 | 0.933 | 0.96 |
| loading | man in black jacket | 0.991 | **0.036** | 0.00 |
| mbike-trick | motorcycle | **0.235** | 0.304 | 0.48 |
| pigs | piglet | 0.982 | **0.016** | 0.00 |
| shooting | man | 0.893 | **0.090** | 0.37 |
| soapbox | soapbox cart | **0.312** | **0.746** | 0.88 |
| **Mean** | | **0.798** | **0.482** (hit 5/10) | |

- Observations: (1) The synthetic benchmark overstates tracking: B0's J&F drops from 0.943 (pasted) to 0.798 (real). (2) B0's two collapses (mbike-trick, soapbox) are *part-vs-whole* errors: the click lands on the rider and SAM 2 outlines only that part (predicted mask ≈ 20–30% of the true one, almost entirely inside it). A box conveys the object's extent, so I2 fixes soapbox (0.31 → 0.75). (3) Text grounding drops from 9/10 hits (pasted) to 5/10 (real): with several people or pigs in view, Grounding DINO-tiny ranks by the noun and ignores attributes ("in white", "in black jacket") or boxes a whole group ("piglet"). The pasted benchmark hid this because the pasted object was usually the only one of its kind. On 2 of 5 misses the correct box was the detector's 2nd candidate; on the other 3 it fell below the 0.35 threshold.
- Takeaway: grounding ambiguity, not tracking, is the main weakness on real video. Periodic re-detection (I5) cannot fix a wrong initial pick, so the next iteration is I2b: candidate re-ranking.

## 2026-10-29: I2b design (dev set only)

- Idea: widen the candidate pool (detector threshold 0.35 → 0.2, up to 10 boxes) and re-rank with CLIP (ViT-B/16), which matches whole captions and is better at attributes.
- First version (pure CLIP ranking) failed on the dev clip lucia_on_hockey: prompt "person", two people with CLIP similarity 25.80 vs 25.72, so CLIP overrode the detector's correct pick (IoU 0.76 → 0.00).
- Final version (fixed before any test run): score = detector confidence × CLIP softmax probability over the candidates. A near-tie in CLIP leaves the detector's order; a confident CLIP match overrides it. Dev set: 5/5 correct picks.

## 2026-10-29: Real DEV set for grounding; I2b rejected; I2c design

- Problem: the synthetic dev set has one ambiguous clip, too few to design grounding fixes on. New **real dev set** `configs/real_dev_clips.txt`: 14 DAVIS *train* videos (no overlap with either test set) with several objects of the same kind, prompted the way a user would describe object 1. `src/eval_grounding.py` scores frame-0 grounding alone (hit = box IoU > 0.5), seconds per variant, no SAM 2.
- Results (`results/grounding_dev*.csv`), one change from I2 each:

| Variant | Real dev hits |
|---|---|
| I2 (Grounding DINO tiny, threshold 0.35) | 11/14 |
| I2b (wider pool + detector × CLIP re-rank) | **6/14** |
| A: Grounding DINO *base* | 11/14 |
| B: CLIP re-rank, normal pool | 7/14 |
| C: wider pool (0.2), no re-rank | 11/14 |

- I2b on the real test set: J&F 0.482 → 0.483, hits 5/10 → 5/10 (fixed pigs 0.016 → 0.982, broke drift-chicane 0.955 → 0.000 by preferring a different car). Together with the dev result: **CLIP ViT-B/16 re-ranking is unreliable for attribute prompts; rejected** (logged, configs kept for reproducibility).
- Key observation: on the real dev set the correct object is in the detector's **top 3 on 14/14 clips** (top-1: 11/14), and every top-1 miss had a close runner-up (0.47/0.33, 0.44/0.42, 0.49/0.36). The detector knows when it is unsure.
- I2c design: keep the top 3 (pool threshold 0.2); if the runner-up scores ≥ ratio × best, the prompt is ambiguous and the demo asks the user to click one of the numbered boxes. Ratio sweep on real dev: 0.5 → asks 9/14, correct 14/14; 0.6 → 8/14, 14/14; **0.7 → 7/14, 14/14**; 0.8 → 5/14, 12/14; 0.9 → 4/14, 12/14. Chose 0.7 (fewest questions with all correct). On the benchmarks no person can click, so a *simulated user* picks the choice on the true object; results report how many clips needed a question.
- Demo note: grounding scores shift with video compression (the same lindy-hop frame from an MP4 vs. the PNG: mean pixel difference 2.9, top score 0.44 → 0.62), so in the demo the same prompt can be ambiguous on one encoding and not another.

## 2026-10-29: I2c on the real test set

- Command: `python -m src.run_pipeline --config configs/I2c.yaml --overlay configs/real_set.yaml`
- Results (`results/real_results.csv`):

| Version | Grounding hits | J&F | Clips that asked the user |
|---|---|---|---|
| B0 (click) | – | 0.798 | – |
| I2 (text) | 5/10 | 0.482 | – |
| I2b (CLIP re-rank) | 5/10 | 0.483 | – |
| **I2c (ask when ambiguous, simulated user)** | **7/10** | **0.674** | 3/10 |

- Asked on drift-chicane, loading, pigs; the simulated pick fixed loading (0.036 → 0.991) and pigs (0.016 → 0.982) and kept drift-chicane correct. Not asked on india and shooting: the detector was *confidently* wrong (no close runner-up), so asking cannot help there. mbike-trick stays low because "motorcycle" does not cover the rider that DAVIS annotates as object 1.
- Takeaway: asking only when the detector is unsure recovers 2 of 5 grounding misses for 3 questions in 10 clips. Clicking (B0) is still the most reliable prompt on real video (0.798 vs 0.674); text is the convenient default with a click as fallback. Remaining misses need either better attribute grounding or a click.

## 2026-10-29: I2c on the synthetic test set; I5 status

- I2c (synthetic): hits 9/10, J&F 0.890, identical to I2. It asked on 3 clips (libby, breakdance, car-shadow), where the top pick was already correct, so the simulated user changed nothing. The parkour miss is not flagged (the detector is confidently wrong). I2b on the synthetic set: hits **6/10**, J&F 0.681, confirming its rejection.
- I5 (periodic re-detection) is implemented (`configs/I5.yaml`, `redetect_corrections` in `src/segment.py`, unit-tested in the smoke test) but **not yet run**. It targets objects that are lost mid-clip; the real-set failures found today are wrong initial picks, which re-detection by design does not change. Next: run I5 on both sets, building on I2c.

## 2026-10-30: I5 (periodic re-detection), built on I2c

- Change: Grounding DINO also runs on frames 10, 20, …; if the SAM 2 track is empty ("lost") or overlaps no confident detection (score ≥ 0.35, box IoU < 0.3: "drifted"), the detection nearest the object's last confirmed position is added as a SAM 2 prompt and the clip is re-tracked. Before running, the correction was changed from "top-scoring detection" to "nearest detection", so with look-alikes it cannot jump to a different object than the one the user chose (unit test added). Settings fixed in advance, not tuned.
- Commands: `python -m src.run_pipeline --config configs/I5.yaml` and the same with `--overlay configs/real_set.yaml`.
- Synthetic: no corrections on any clip; identical to I2c (hits 9/10, J&F 0.890). Pasted objects never leave the frame, so there is nothing to re-detect.
- Real test: **J&F 0.674 → 0.600 (worse)**. Corrections on 2 clips: india (0.000 → 0.023, was already a grounding miss) and loading (**0.991 → 0.235**).
- Why loading broke: the track was correct, but on frame 10 the right man was detected with score 0.32, below the 0.35 cutoff, so only another man counted as a detection. The correct track matched no *confident* detection, was labelled "drifted" and re-seeded on the other man. The bug is using the confidence cutoff for the "is the track still right?" check: a weak detection is not trusted enough to *move* the track, but it is enough to *confirm* it.
- Takeaway: re-detection can only help when it is very sure the track is wrong. Next: I5b (confirm against all candidates), checked on the real **dev** set before the test set, since this failure was found on the test set.

## 2026-10-30: Evaluating each stage on its own; checking the metrics

- Problem: end-to-end numbers mix stages. A grounding miss makes J&F, PSNR and LPIPS all bad even if tracking and filling worked, so a table of end-to-end means cannot say which stage to improve.
- New: `python -m src.stage_report` writes `results/stage_report.md` with one section per stage: (1) grounding hit rate, (2) tracking J&F from a correct click, and for text versions on the clips grounding got right only, (3) fill quality with every fill method given the **true** masks (`--overlay configs/fill_only.yaml`, results in `results/fill_results.csv`), (4) end to end.
- First finding from the split: on real video, text versions track well once grounding is right (J&F on grounding hits 0.87–0.91, vs. 0.48–0.67 overall). Nearly all of the text-prompt loss is grounding.
- New: `python -m src.metric_check` scores fake outputs of known quality on the 10 synthetic test clips with the exact evaluation code (no tuning involved). Means:

| Case | PSNR mask ↑ | SSIM ↑ | LPIPS mask ↓ | Flicker ↓ | Flicker RAFT ↓ |
|---|---|---|---|---|---|
| perfect (clean video) | 100.0 | 1.000 | 0.000 | 7.63 | 7.31 |
| not removed | 10.45 | 0.922 | 0.404 | 34.11 | 36.34 |
| gray box | 12.03 | 0.929 | 0.439 | 7.76 | 9.70 |
| blurred clean video | 20.51 | 0.949 | 0.305 | 5.55 | 5.93 |
| clean video shifted 4 px | 17.68 | 0.933 | 0.087 | 10.07 | 10.13 |
| clean + per-frame brightness flicker | 37.27 | 0.998 | 0.007 | 11.71 | 11.56 |

- What it shows:
  - PSNR mask prefers the blurred fill to the sharp, slightly shifted one (20.5 vs 17.7); LPIPS says the opposite (0.305 vs 0.087), which matches how they look. This is the measured reason LPIPS is a primary metric.
  - Flicker only measures stability, not correctness: a flat gray box is as stable as the clean video (7.76 vs 7.63), and blur is *more* stable than the clean video (5.55). Flicker must be read next to the clean-video reference and next to LPIPS, never alone.
  - Flicker is the only metric that catches the flickering case (7.6 → 11.7) while per-frame metrics call it near-perfect (PSNR 37, LPIPS 0.007). The temporal metric is needed.
  - SSIM is computed on the whole frame, so it barely separates "object not removed" (0.922) from perfect (1.000). It is the weakest metric here; kept for continuity.
- Agreement (Spearman over all (version, clip) rows of results.csv, `results/metric_agreement.csv`): Farneback vs RAFT flicker 0.98 (they rank outputs the same; RAFT stays primary because it is independent of our method's flow), PSNR mask vs LPIPS 0.38 (they measure different things, so both are reported).

## 2026-10-30: Fill stage on its own (true masks)

- Command: `python -m src.run_pipeline --config configs/<B0|I1|I3|I3b|I4>.yaml --overlay configs/fill_only.yaml` (results in `results/fill_results.csv`).

| Fill (true masks) | PSNR mask ↑ | LPIPS mask ↓ | Flicker RAFT ↓ | Fill s/frame |
|---|---|---|---|---|
| B0 Telea | 17.33 | 0.249 | 16.96 | 0.03 |
| I1 Telea + 5 px dilation | 15.77 | 0.309 | 18.96 | 0.03 |
| I3 flow-guided | 16.58 | 0.248 | 16.13 | 0.16 |
| I3b flow-guided, hop cap | 17.46 | 0.239 | 15.65 | 0.16 |
| I4 ProPainter | **22.26** | **0.125** | **6.99** | 4.37 |
| (clean video) | – | – | 7.31 | – |

- Same ranking as end to end, so the fill differences reported before were fill differences, not mask side effects. Perfect masks raise every method by about 1 dB PSNR mask (B0 16.39 → 17.33, I4 21.14 → 22.26) and lower LPIPS (B0 0.303 → 0.249): that gap is what tracking errors cost the fill.
- I1 is still worst with true masks, so its loss comes from the bigger hole, not from SAM 2.
- ProPainter's flicker on true masks (6.99) is at the clean video's level (7.31): its remaining error is per-frame detail (LPIPS 0.125), not temporal stability.

## 2026-10-30: I5b on the real dev set; re-detection rejected

- Change (I5b): a track counts as confirmed if it matches *any* keyframe candidate, not only confident ones. Checked on the real **dev** set (14 DAVIS train clips with look-alikes, full masks; new overlay `configs/real_dev_set.yaml`), because the I5 failure was found on the test set.
- Commands: `python -m src.run_pipeline --config configs/<I2c|I5|I5b>.yaml --overlay configs/real_dev_set.yaml` (`results/real_dev_results.csv`).

| Real dev (14 clips) | Grounding hits | J&F | Clips re-tracked |
|---|---|---|---|
| I2c (no re-detection) | 14/14 | **0.912** | – |
| I5 | 14/14 | 0.799 | 7 |
| I5b | 14/14 | 0.866 | 3 |

- I5b removed most false corrections (crossing 0.364 → 0.948, lindy-hop, schoolgirls) but is still below no re-detection. All 4 remaining corrections were false alarms: the track was right (IoU with the true mask 0.93–0.95), but the detector did not find the object on that keyframe at all, so the track matched nothing and was moved to another object (dancing 0.938 → 0.640, upside-down 0.901 → 0.580).
- Over 24 real clips (dev + test) re-detection made **no** correct correction. SAM 2's memory never lost the object on these clips, while the detector misses objects in single frames (unusual poses, upside-down). The detector is the less reliable signal, so it should not override the tracker.
- Decision: **re-detection rejected** (I5 and I5b kept as configs for reproducibility). I5b is not run on the test set: it was rejected on dev. The final system keeps I2c grounding + SAM 2 tracking. Re-detection would only make sense for objects that really leave the frame, which our clips do not test.

## 2026-10-31: I6 speed: smaller SAM 2 models (dev only); rejected

- Question: SAM 2 is 97% of B0's runtime. Can a smaller SAM 2.1 model keep tracking quality?
- Command (real dev, click prompts so grounding is out of the picture): `python -m src.sweep --config configs/B0.yaml --key segmentation.hf_model --segmentation sam2 --clips-file configs/real_dev_clips.txt --values facebook/sam2.1-hiera-tiny facebook/sam2.1-hiera-small facebook/sam2.1-hiera-base-plus facebook/sam2.1-hiera-large` (results in `results/tuning.csv`; sweep.py now names model-id versions by their last path part).

| SAM 2.1 model | Real dev J&F | Segment s/frame (M5, MPS) |
|---|---|---|
| hiera-tiny | 0.708 | 0.70 |
| hiera-small | 0.710 | 0.76 |
| hiera-base-plus | 0.604 | 0.98 |
| hiera-large (B0) | **0.791** | 1.18 |

- Smaller models are at most 1.7× faster on this Mac and lose 0.08–0.19 J&F, with whole-clip failures from the same click (kid-football: large 0.99, small 0.02; hockey and skate-park: base-plus 0.30/0.34). Not worth it: **keep hiera-large**. Speed is not run on the test set, since no candidate passed dev.
- Side finding that drives I8: with the large model, the click prompt still loses badly on some clips where the text box works (boxing 0.12 vs 0.95, upside-down 0.08 vs 0.90). The click lands on the object, but SAM 2 segments only a part of it (12% of the boxer, 12% of the upside-down person).

## 2026-10-31: Plain-language scorecard

- Why: the raw metrics (J&F, LPIPS, RAFT flicker) are exact but hard to read for anyone outside the project. `python -m src.scorecard` writes `results/scorecard.md`, which turns them into five everyday numbers. Nothing new is measured; all numbers come from the result CSVs.
  - **Right object**: clips where the text prompt picked the right thing (grounding hits, e.g. 7/10).
  - **Mask accuracy**: overlap with the true object in % (J&F × 100).
  - **Fill score** (0–100): 0 = object still there, 100 = identical to the real background. Per clip: 100 × (1 − LPIPS of the fill / LPIPS of leaving the object in), using the "not removed" case from `src/metric_check.py`.
  - **Steadiness**: flicker relative to the real video (RAFT flicker / clean-video RAFT flicker; 1.0× = as steady as real footage).
  - **Speed**: seconds per frame on the M5 MacBook.
- Reading the current results this way: baseline fill 24/100 and 2.8× the real video's flicker; ProPainter 63/100 and 1.1×; with perfect masks ProPainter reaches 68/100 and 1.0×. On real videos, mask accuracy is 80% with a click and 67% with text (I2c).

## 2026-10-31: I8, a click selects the whole object

- Change: from one click SAM 2 predicts three candidate masks (roughly part / larger part / whole object) and B0 keeps the one it is most confident about, which is often a part. I8 keeps the **largest** candidate (`segmentation.click_mask: largest`, `whole_object_mask` in `src/segment.py`) and starts tracking from that mask. One change vs B0.
- Rule chosen on dev frame 0 (real dev + synthetic dev, 19 clips, IoU with the true mask): most-confident 0.757 → largest 0.879 on real dev, best-possible 0.914; synthetic dev 0.972 for every rule.
- Commands: `python -m src.run_pipeline --config configs/I8.yaml` with `--overlay configs/real_dev_set.yaml`, then `--overlay configs/real_set.yaml`, then no overlay.

| J&F | Real dev (14) | Real test (10) | Synthetic test (10) |
|---|---|---|---|
| B0 (most confident mask) | 0.791 | 0.798 | 0.943 |
| **I8 (largest mask)** | **0.903** | **0.858** | 0.941 |

- Fixed: boxing 0.12 → 0.95, upside-down 0.08 → 0.90 (dev); mbike-trick 0.24 → 0.92, soapbox 0.31 → 0.74 (test), the part-vs-whole failures noted for B0 on 2026-10-29.
- Broke: shooting 0.89 → 0.39 (test) and bmx-bumps 0.53 → 0.45 (dev). In both, DAVIS's object 1 is *less* than the whole thing: on shooting it covers only part of the man, on bmx-bumps the rider without the bike. I8 selects the whole man / rider + bike. For removal that is arguably what a user wants, but the metric counts it as wrong. Click ambiguity cannot be fully solved by a fixed rule; the demo could show the three masks and let the user pick (as I2c does for boxes).
- Synthetic test unchanged (pasted objects are whole by construction). Fill metrics within noise (PSNR mask 16.39 → 16.34).
- Decision: **kept** for click prompts; the demo now uses it.
