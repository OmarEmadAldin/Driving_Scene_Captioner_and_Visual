"""
dataset.py
----------
Thin wrapper around data/images + data/annotations so every other script
(visualization, detection, eval) shares one source of truth.

Expects:
    data/images/<filename>       frame images (jpg/png/...)
    data/annotations/<one>.json  a single JSON file covering all images

The annotations file can be either the official BDD100K label shape:
    [{"name": "abc.jpg", "attributes": {"weather": ..., "timeofday": ..., "scene": ...}}, ...]
or a flat shape:
    [{"filename": "abc.jpg", "weather": ..., "timeofday": ..., "scene": ...}, ...]

Images with no matching annotation entry still load fine -- weather /
timeofday / scene just come back as None (shown as "unknown" in the
dataset-overview figure).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Frame:
    filename: str
    path: Path
    weather: Optional[str] = None
    timeofday: Optional[str] = None
    scene: Optional[str] = None

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")


def _load_annotations(annotations_dir: Path) -> dict:
    """Reads the single annotations JSON file in annotations_dir and returns
    {filename: {"weather": ..., "timeofday": ..., "scene": ...}}.

    Returns {} (no annotations) if the folder is missing/empty, so the
    dataset still works before you've added labels.
    """
    if not annotations_dir.exists():
        return {}

    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        return {}
    if len(json_files) > 1:
        raise FileNotFoundError(
            f"Expected exactly one annotations JSON file in {annotations_dir}, "
            f"found {len(json_files)}: {[p.name for p in json_files]}. "
            "Combine them into a single file."
        )

    with open(json_files[0]) as f:
        raw = json.load(f)

    by_filename = {}
    for entry in raw:
        filename = entry.get("name") or entry.get("filename")
        if not filename:
            continue
        attrs = entry.get("attributes", entry)
        by_filename[filename] = {
            "weather": attrs.get("weather"),
            "timeofday": attrs.get("timeofday"),
            "scene": attrs.get("scene"),
        }
    return by_filename


class DrivingSceneDataset:
    def __init__(self, images_dir: str = "data/images", annotations_dir: str = "data/annotations"):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"No images directory at {self.images_dir}. Put your frames in data/images/."
            )

        annotations = _load_annotations(self.annotations_dir)

        image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}.")

        self.frames = [
            Frame(filename=p.name, path=p, **annotations.get(p.name, {}))
            for p in image_paths
        ]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def by_filename(self, filename: str) -> Frame:
        for f in self.frames:
            if f.filename == filename:
                return f
        raise KeyError(filename)

    def summary(self) -> dict:
        """Counts of weather / time-of-day / scene, for the dataset-overview viz."""
        counts = {"weather": {}, "timeofday": {}, "scene": {}}
        for f in self.frames:
            for field, val in [("weather", f.weather), ("timeofday", f.timeofday), ("scene", f.scene)]:
                key = val or "unknown"
                counts[field][key] = counts[field].get(key, 0) + 1
        return counts
