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
