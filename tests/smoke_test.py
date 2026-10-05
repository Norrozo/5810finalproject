"""End-to-end smoke test on tiny fake data. No GPU, no SAM 2, no DAVIS download.

Builds a fake DAVIS folder (textured panning background + a moving blob object),
then runs make_synthetic -> pipeline (oracle segmentation) -> evaluate -> summarize.
Checks that every stage produces output and the metrics are sane.

Run from the repo root:
    python -m tests.smoke_test
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from src import extract_frames, make_synthetic, run_pipeline, summarize
from src.evaluate import boundary_f, jaccard

H, W, N = 120, 200, 10


def fake_background(i: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (H, W * 2, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 6)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    return np.ascontiguousarray(base[:, 3 * i: 3 * i + W])  # camera pans 3 px/frame


def make_fake_davis(root: Path) -> None:
    for clip, seed in (("bgclip", 1), ("objclip", 2)):
        d = root / "JPEGImages" / "480p" / clip
        d.mkdir(parents=True)
        for i in range(N):
            img = fake_background(i, seed)
            if clip == "objclip":
                cv2.ellipse(img, (50 + 8 * i, 60), (22, 30), 0, 0, 360, (40, 200, 240), -1)
            cv2.imwrite(str(d / f"{i:05d}.jpg"), img)
    ann = root / "Annotations" / "480p" / "objclip"
    ann.mkdir(parents=True)
    for i in range(N):
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (50 + 8 * i, 60), (22, 30), 0, 0, 360, 1, -1)
        im = Image.fromarray(m, mode="P")
        im.putpalette([0, 0, 0, 128, 0, 0] + [0] * 762)
        im.save(ann / f"{i:05d}.png")


def test_metrics():
    a = np.zeros((50, 50), bool); a[10:30, 10:30] = True
    b = np.zeros((50, 50), bool); b[20:40, 10:30] = True
    assert jaccard(a, a) == 1.0 and boundary_f(a, a, 0.008) == 1.0
    assert abs(jaccard(a, b) - 1 / 3) < 1e-9
    assert jaccard(np.zeros_like(a), np.zeros_like(a)) == 1.0


def main():
    test_metrics()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_fake_davis(tmp / "DAVIS")
        (tmp / "clips.txt").write_text("# test\nobj_on_bg objclip 1 bgclip\n")

        cfg = yaml.safe_load(Path("configs/B0.yaml").read_text())
        cfg.update(version="SMOKE", data_root=str(tmp / "data"), runs_root=str(tmp / "runs"),
                   results_csv=str(tmp / "results.csv"), eval_clips=str(tmp / "clips.txt"))
        cfg["segmentation"]["method"] = "oracle"
        cfg_path = tmp / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        make_synthetic.main(["--config", str(cfg_path), "--davis-root", str(tmp / "DAVIS")])
        run_pipeline.main(["--config", str(cfg_path)])
        run_pipeline.main(["--config", str(cfg_path)])  # second run should skip (resume)

        run_dir = tmp / "runs" / "SMOKE" / "obj_on_bg"
        for sub in ("masks", "masks_refined", "inpainted"):
            assert len(list((run_dir / sub).glob("*.png"))) == N, sub
        assert (run_dir / "side_by_side.mp4").stat().st_size > 0
        assert (run_dir / "stills.png").exists()

        rows = list(csv.DictReader(open(tmp / "results.csv")))
        assert len(rows) == 1, "upsert should keep one row per (version, clip)"
        r = rows[0]
        assert float(r["JF"]) == 1.0, "oracle masks must score perfectly"
        assert 0 < float(r["psnr_mask"]) < float(r["psnr"]), "fill region should be the hard part"
        assert float(r["flicker"]) >= 0

        # I3 flow-guided inpainting on the same clip. The fake camera pans, so the hidden
        # background is visible in other frames and must beat per-frame Telea.
        cfg.update(version="SMOKE_FLOW")
        cfg["inpaint"]["method"] = "flow_guided"
        cfg["refine"].update(fill_holes=True, dilate_px=1, min_blob_area=5)  # exercise I1 refine code too
        cfg_path.write_text(yaml.safe_dump(cfg))
        run_pipeline.main(["--config", str(cfg_path)])
        rows = {r["version"]: r for r in csv.DictReader(open(tmp / "results.csv"))}
        assert float(rows["SMOKE_FLOW"]["psnr_mask"]) > float(rows["SMOKE"]["psnr_mask"]) + 3, \
            "flow-guided fill should clearly beat per-frame Telea on a panning clip"

        # Custom-video path: frame extraction
        vid = tmp / "v.mp4"
        wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W * 4, H * 4))
        for i in range(N):
            wr.write(cv2.resize(fake_background(i, 3), (W * 4, H * 4)))
        wr.release()
        n = extract_frames.extract(str(vid), tmp / "data" / "vid" / "frames", max_frames=5, max_height=240)
        assert n == 5 and cv2.imread(str(tmp / "data/vid/frames/00000.png")).shape[0] == 240

        summarize.main(["--csv", str(tmp / "results.csv")])
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
