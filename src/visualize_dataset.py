"""
visualize_dataset.py
---------------------
Produces two figures so you can sanity-check the sample before spending any
GPU time on it:

  1. outputs/figures/dataset_overview.png
     Bar charts of weather / time-of-day / scene distribution across the
     sampled frames.

  2. outputs/figures/sample_grid.png
     A contact-sheet grid of N random frames, useful to eyeball diversity
     (rain vs clear, day vs night, highway vs city street, etc.)

Usage:
    python src/visualize_dataset.py --config configs/config.yaml
"""
import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from dataset import DrivingSceneDataset


def plot_overview(ds: DrivingSceneDataset, out_path: Path):
    summary = ds.summary()
    fields = ["weather", "timeofday", "scene"]

    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4))
    if len(fields) == 1:
        axes = [axes]

    for ax, field in zip(axes, fields):
        counts = summary[field]
        labels = list(counts.keys())
        values = [counts[k] for k in labels]
        order = sorted(range(len(values)), key=lambda i: -values[i])
        labels = [labels[i] for i in order]
        values = [values[i] for i in order]

        ax.bar(labels, values, color="#4C72B0")
        ax.set_title(f"{field} distribution (n={len(ds)})")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("count")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sample_grid(ds: DrivingSceneDataset, out_path: Path, n: int = 12, seed: int = 42):
    random.seed(seed)
    n = min(n, len(ds))
    indices = random.sample(range(len(ds)), n)

    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, idx in zip(axes, indices):
        frame = ds[idx]
        img = frame.load()
        ax.imshow(img)
        title = f"{frame.weather or '?'} / {frame.timeofday or '?'} / {frame.scene or '?'}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run(cfg: dict, grid_size: int = 12) -> "DrivingSceneDataset":
    ds = DrivingSceneDataset(
        images_dir=cfg["data"]["images_dir"],
        annotations_dir=cfg["data"]["annotations_dir"],
    )
    print(f"Loaded {len(ds)} frames")

    figures_dir = Path("outputs/figures")
    plot_overview(ds, figures_dir / "dataset_overview.png")
    plot_sample_grid(ds, figures_dir / "sample_grid.png", n=grid_size, seed=cfg["project"]["seed"])
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--grid-size", type=int, default=12)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg, grid_size=args.grid_size)


if __name__ == "__main__":
    main()
