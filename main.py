#!/usr/bin/env python3
"""
main.py
-------
End-to-end pipeline for the driving-scene BLIP captioner project.
Run this from the repo root:

    python main.py --config configs/config.yaml

Expects your data already in place (no download step):
    data/images/       frame images
    data/annotations/  optional single JSON file with weather/timeofday/scene labels

Steps:
    1. Visualize dataset (sanity-check figures: overview + sample grid)
    2. Caption every image with BLIP-2 and render an image-left /
       caption-right figure for each one
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import visualize_dataset      # noqa: E402
import caption_and_visualize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--grid-size", type=int, default=12, help="Sample-grid figure size (visualize step)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override BLIP caption length cap")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("== 1/2: Visualizing dataset (overview + sample grid) ==")
    visualize_dataset.run(cfg, grid_size=args.grid_size)

    print("\n== 2/2: Captioning every image with BLIP and rendering figures ==")
    caption_and_visualize.run(cfg, max_new_tokens=args.max_new_tokens)

    print("\nDone. See outputs/figures/ and outputs/captions/figures/ for everything.")


if __name__ == "__main__":
    main()
