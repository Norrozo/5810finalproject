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
