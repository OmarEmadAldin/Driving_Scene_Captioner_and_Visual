"""
caption_and_visualize.py
-------------------------
For every image in data/images/:
  1. Runs BLIP-2 to generate a caption.
  2. Renders a figure with the image on the left and the caption text on
     the right, saved to outputs/captions/figures/<filename>.png.

Also writes outputs/captions/captions.json ({filename: caption}) so the
captions are available without re-running the model or re-parsing figures.

Usage:
    PYTHONPATH=src python src/caption_and_visualize.py --config configs/config.yaml
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from dataset import DrivingSceneDataset, Frame
from models.blip_baseline import BlipBaseline


def render_frame(frame: Frame, caption: str, out_path: Path):
    """Image on the left, caption (word-wrapped) on the right."""
    img = frame.load()

    fig, (ax_img, ax_text) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(frame.filename, fontsize=10)

    ax_text.axis("off")
    meta_bits = [b for b in (frame.weather, frame.timeofday, frame.scene) if b]
    meta_line = " · ".join(meta_bits) if meta_bits else None

    wrapped = textwrap.fill(caption, width=42)
    y = 0.65
    if meta_line:
        ax_text.text(
            0.0, 0.85, meta_line, fontsize=10, color="#666666",
            transform=ax_text.transAxes, va="top",
        )
        y = 0.72
    ax_text.text(
        0.0, y, wrapped, fontsize=13, color="#111111",
        transform=ax_text.transAxes, va="top", wrap=True,
    )
    ax_text.set_title("BLIP caption", fontsize=10, loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(cfg: dict, max_new_tokens: int = None):
    ds = DrivingSceneDataset(
        images_dir=cfg["data"]["images_dir"],
        annotations_dir=cfg["data"]["annotations_dir"],
    )
    print(f"Loaded {len(ds)} frames")

    model = BlipBaseline(cfg)
    max_new_tokens = max_new_tokens or cfg.get("models", {}).get("caption", {}).get("max_new_tokens", 40)

    figures_dir = Path(cfg["captions"]["figures_dir"])
    json_path = Path(cfg["captions"]["json_path"])

    captions = {}
    for frame in tqdm(ds, desc="Captioning"):
        img = frame.load()
        caption = model.caption(img, max_new_tokens=max_new_tokens)
        captions[frame.filename] = caption

        stem = Path(frame.filename).stem
        render_frame(frame, caption, figures_dir / f"{stem}.png")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(captions, f, indent=2)

    print(f"\nSaved {len(captions)} figures to {figures_dir}/")
    print(f"Saved captions to {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override caption length cap")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
