import argparse

from src.inpaint import run
from src.utils import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4: inpaint masked regions")
    ap.add_argument("--config", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.clip, args.overwrite)


if __name__ == "__main__":
    main()
