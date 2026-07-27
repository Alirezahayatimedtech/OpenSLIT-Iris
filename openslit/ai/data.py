"""Dataset utilities for OpenSLIT multi-class semantic segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_split_table(path: Path, split: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"image_id", "image_file", "mask_file", "split"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    selected = data[data["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No images are assigned to split {split!r}")
    return selected.reset_index(drop=True)


class SegmentationDataset:
    """A minimal PyTorch-compatible dataset with deterministic resizing.

    Strong augmentation is deliberately not enabled by default because the pilot
    must first quantify acquisition and annotation variability. Additional
    augmentation can be introduced through a reviewed experiment configuration.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        image_dir: Path,
        mask_dir: Path,
        input_size: int,
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = int(input_size)
        if self.input_size < 64:
            raise ValueError("input_size must be at least 64 pixels")

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install AI dependencies with: python -m pip install -e '.[ai]'"
            ) from exc
        row = self.table.iloc[index]
        image_path = self.image_dir / str(row["image_file"])
        mask_path = self.mask_dir / str(row["mask_file"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            original_size = (rgb.height, rgb.width)
            image = rgb.resize(
                (self.input_size, self.input_size),
                resample=Image.Resampling.BILINEAR,
            )
            image_array = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(mask_path) as source_mask:
            mask = source_mask.resize(
                (self.input_size, self.input_size),
                resample=Image.Resampling.NEAREST,
            )
            mask_array = np.asarray(mask, dtype=np.int64)
        normalized = (image_array - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_array).long()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": str(row["image_id"]),
            "image_file": str(row["image_file"]),
            "original_size": original_size,
        }
