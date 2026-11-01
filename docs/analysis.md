# Magic Eraser: what our metrics say about each iteration

Group 35 (Daniel Zhang, William Huang). Every number below comes from a results file written
by our scripts (`results/*.csv`, `results/stage_report.md`, `results/scorecard.md`; charts of
every metric, J and F included, in `results/figures/*.png` from `python -m src.plot_metrics`); the
full record with commands is `docs/experiment_log.md`. Nothing here is estimated by hand.

## 1. How we test: one stage at a time

The pipeline has three stages that can fail: **finding** the object (text prompt → box),
**masking/tracking** it (SAM 2), and **filling** the hole. If we only scored the final video,
a wrong box would make every number bad and we could not tell which stage to fix. So each
stage is scored on its own, with the stages before it replaced by the correct answer:

| Stage | Question | How it is isolated | Score (plain words) |
|---|---|---|---|
| Finding | Did the text prompt pick the right object? | frame 0 only, box vs. true object | **Right object**: x/10 clips (box overlap > 50%) |
| Masking + tracking | Does the mask cover the object in every frame? | start from a correct click, or keep only clips where finding was right | **Mask accuracy** % (J&F × 100) |
| Filling | Does the hole look like the real background, without flicker? | every fill method gets the **true** masks (`configs/fill_only.yaml`) | **Fill score** /100 and **Steadiness** × |
| End to end | What does the user get? | full pipeline | all of the above + **Speed** s/frame |

Test sets (all chosen before results; settings are tuned only on separate dev sets):

- **Pasted benchmark** (10 clips): a DAVIS object pasted onto another DAVIS video, so the true
  background is known and the fill can be scored. Weakness: pasted objects have hard edges and
  are usually the only one of their kind, so it is easy for finding and tracking.
- **Real videos** (10 DAVIS val clips): unedited footage with look-alikes and crowds; masks only.
- **Multi-object videos** (5 DAVIS val clips, 15 objects): plural prompts ("pigs"); the truth is all of them.
- **Dev sets** for choosing settings: 5 pasted clips, 14 real clips (DAVIS train, look-alikes),
  7 multi-object clips.

### The fill score and steadiness, explained

- **Fill score** = 100 × (1 − LPIPS of our fill / LPIPS of leaving the object in). 0 means it
  looks as wrong as not removing the object, 100 means identical to the true background.
  LPIPS is a learned "how different does this look to a person" distance.
- **Steadiness** = flicker of our fill / flicker of the real video in the same spot. 1.0× means
  as steady as real footage; 2.8× means almost three times as jumpy.

### Can we trust the metrics? (`python -m src.metric_check`)

We scored fake outputs whose quality we know in advance (10 pasted clips, means):

| Fake output | PSNR mask ↑ | LPIPS ↓ | Flicker RAFT ↓ | What it tells us |
|---|---|---|---|---|
| perfect (true background) | 100 | 0.000 | 7.31 | the floor; real video flickers 7.31 |
| object not removed | 10.45 | 0.404 | 36.34 | every metric says bad ✓ |
| flat gray box | 12.03 | 0.439 | 9.70 | steady but wrong → flicker alone is not enough |
| blurred true background | 20.51 | 0.305 | 5.93 | PSNR likes blur |
| sharp, shifted 4 px | 17.68 | 0.087 | 10.13 | LPIPS prefers sharp-but-shifted, as a person does |
| brightness flicker per frame | 37.27 | 0.007 | 11.56 | only flicker catches it |

Conclusions: PSNR rewards blur (blurred 20.5 dB beats shifted 17.7 dB, though it looks worse),
so **LPIPS is our main fill metric**. Flicker only measures stability (a gray box is steady), so
**steadiness is always read next to the fill score**. Per-frame metrics miss flicker entirely,
so the temporal metric is needed. SSIM barely separates "not removed" (0.922) from perfect
(1.000) and is kept only for continuity.

## 2. Iterations: what changed, what the numbers said, what we decided

One change per iteration. "Pasted" = pasted benchmark, "real" = real videos.

| Tag | Change | Target number | Before → after | Decision and why |
|---|---|---|---|---|
| **B0** | click → SAM 2 → OpenCV Telea fill | – | pasted: mask 94%, fill 24/100, 2.8×, 1.38 s/frame; real mask 80% | baseline. Masks good on pasted; **fill is the weak stage** |
| **I1** | grow mask 5 px, fill holes | fill score | 24 → 21/100 (true masks: 38 → 22) | **rejected**: pasted objects have no halo; a bigger hole is just harder to fill |
| **I2** | text prompt (Grounding DINO) instead of click | right object | pasted 9/10; real **5/10**, mask 80% → 48% | kept as the text option; real video shows **finding** is the weak stage for text |
| **I3** | our flow-guided fill (copy background from other frames) | steadiness | flicker −13% (RAFT −9%), LPIPS 0.303 → 0.298 | improvement in steadiness, but long copy chains drift |
| **I3b** | I3 + cap copy chain at 5 hops (tuned on dev) | fill score | 24 → 26/100, 2.8× → 2.5× | **kept** as our fast fill (0.16 s/frame for the fill step) |
| **I4** | ProPainter fill | fill score | 24 → **63/100**, 2.8× → **1.1×**; 5.43 s/frame | **kept**: biggest gain of any iteration; cost is 4× slower |
| **I2b** | rank text candidates with CLIP | right object | real 5/10 → 5/10; real dev 11 → **6**/14 | **rejected**: CLIP ignores attributes ("in white") as badly as the detector |
| **I2c** | ask the user when the top two boxes score close | right object | real 5/10 → **7/10**, mask 48% → **67%**; asked on 3/10 clips | **kept**: the detector knows when it is unsure (dev: right object in top 3 on 14/14) |
| **I5** | re-detect every 10 frames, re-seed lost tracks | real mask | real 67% → **60%** | **rejected**: moved a correct track onto another man (loading 99% → 24%) |
| **I5b** | I5 with a fixed confirm check | real dev mask | real dev 91% → 87% (I5: 80%) | **rejected**: 0 correct corrections in 24 real clips; SAM 2 never lost the object, the detector missed it |
| **I6** | smaller SAM 2 models (speed) | speed at same mask | real dev 79% → 60–71%, at most 1.7× faster | **rejected**: speed gain too small for the accuracy lost |
| **I8** | a click keeps SAM 2's *largest* mask (whole object, not a part) | real mask | real 80% → **86%**, real dev 79% → **90%** | **kept**: fixes "clicked the rider, got only the helmet" (mbike-trick 24% → 92%) |
| **I7** | plural prompt ("the pigs") keeps every box ≥ 0.45, tracks each | objects found | multi-object test: 5/15 → **14/15**, mask 52% → **87%**; 2 extra; 1.26 → 2.17 s/frame | **kept**: better on all 12 multi-object clips (dev + test) |

On the questions in our task list:

- **Masking/tracking for I4:** I4 only changes the fill; it reuses B0's SAM 2 masks (94%), so
  its gain is purely the fill. With **true** masks ProPainter goes from 63 to 68/100: tracking
  errors cost it 5 points.
- **Masking/tracking for I6:** tested directly (same clicks, four model sizes on the real dev
  set). The large model is the only one without whole-clip failures (kid-football: large 99%,
  small 2%), so tracking stays on hiera-large.

## 3. Where the pipeline struggles (ranked)

1. **Filling (biggest gap overall).** Even the best fill, ProPainter, scores 63/100 end to end
   and 68/100 with perfect masks, so better masks cannot close the gap; the fill method is the
   limit. Fill-only results (true masks): Telea 38, our flow fill 41, ProPainter 68 /100.
   Hardest case: an object the camera follows (e.g. the bear): the background behind it is never
   visible in any frame, so every method has to invent it. Our flow fill falls back to Telea
   there and leaves a smudge.
2. **Finding the right object from text on real video.** Text finds the right object 7/10
   (I2c) vs 9/10 on pasted clips. The 2 remaining misses (india "woman in white", shooting
   "man") are *confident* wrong picks, so asking does not trigger. When the box is right,
   tracking is good: mask accuracy on correctly found clips is 87–91% (stage report), so
   **nearly all of the text-prompt loss is finding, not tracking**.
3. **Masking: part vs. whole object.** A click is ambiguous (the glasses or the man?). I8 fixed
   most cases (80% → 86%) but loses 2 clips where DAVIS labels only part of the thing
   (shooting 89% → 39%; bmx rider without the bike). A fixed rule can't always be right; the
   next step is letting the user pick among SAM 2's three masks.
4. **Speed.** ProPainter makes the full pipeline 5.4 s/frame (a 60-frame clip ≈ 5.5 min on the
   M5). SAM 2 is 97% of the baseline's time and smaller models were not worth it (I6).
5. **Several objects cost time.** Plural prompts now work (I7: 14/15 objects found), but SAM 2
   tracks each object separately, so 7 people take ~3× as long as one. "People" also catches
   people the user may not have meant (2 extra on the test set).
6. **Tracking itself is not the problem.** From a correct start SAM 2 holds the object: 94%
   pasted, 86–90% real, and in 24 real clips it never lost the object (I5/I5b).

## 4. In one paragraph

Scoring each stage separately shows that the system's quality is limited by **filling** (best
63/100, steadiness 1.1×, with ProPainter) and, for text prompts on real video, by **finding the
right object** (7/10). Tracking is reliable once it starts on the right object. The iterations
that helped attacked exactly those stages: ProPainter for the fill (+39 points), asking when
unsure for finding (+2 clips), whole-object clicks for masking (+6 points on real video), and
plural prompts for several objects (objects found 5/15 → 14/15).
The rejected ones (I1, I2b, I5, I5b, I6) are rejected because a measured number got worse, and
each is logged with the reason.
