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

## (date): B0, baseline

- Change: n/a (first version). Click prompt → SAM 2 → binarize → cv2.inpaint (Telea, radius 5), per frame.
- Command: `python -m src.run_pipeline --config configs/B0.yaml`
- Results: _(paste summary table after running)_
- Observations: _(expect halos around object edges, blurry fill in large holes, flicker well above flicker_ref)_
- Takeaway:
