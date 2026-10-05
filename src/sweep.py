"""Sweep one config value on the DEV clips and print a comparison table.

Tuning on the test set would make the test numbers optimistic, so sweeps run on
configs/dev_clips.txt and write to results/tuning.csv, never results/results.csv.
Each value gets its own version folder: runs/<version>_<key>-<value>/.

Usage:
    python -m src.make_synthetic --config configs/I3.yaml --davis-root DAVIS --clips-file configs/dev_clips.txt
    python -m src.sweep --config configs/I3.yaml --key inpaint.max_hops --values 1 2 3 5 8 none
"""
from __future__ import annotations

import argparse
import copy

import yaml

from src import run_pipeline, summarize
from src.utils import load_config


def parse_value(v: str):
    return None if v.lower() == "none" else yaml.safe_load(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--key", required=True, help="Dotted config key, e.g. inpaint.max_hops")
    ap.add_argument("--values", nargs="+", required=True, help="Values to try ('none' = unset)")
    ap.add_argument("--clips-file", default="configs/dev_clips.txt")
    ap.add_argument("--segmentation", default="oracle",
                    help="oracle (default) isolates the tuned stage from SAM 2 errors and needs no GPU")
    args = ap.parse_args(argv)

    base = load_config(args.config)
    versions = []
    for raw in args.values:
        cfg = copy.deepcopy(base)
        *parents, leaf = args.key.split(".")
        node = cfg
        for k in parents:
            node = node[k]
        node[leaf] = parse_value(raw)
        cfg["segmentation"]["method"] = args.segmentation
        cfg["segmentation"].pop("reuse_from", None)
        cfg.update(version=f"{base['version']}_{leaf}-{raw}", eval_clips=args.clips_file,
                   results_csv="results/tuning.csv")
        cfg.pop("_config_path", None)
        path = f"runs/{cfg['version']}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        run_pipeline.main(["--config", path, "--no-viz"])
        versions.append(cfg["version"])

    print()
    summarize.main(["--csv", "results/tuning.csv", "--versions", *versions])


if __name__ == "__main__":
    main()
