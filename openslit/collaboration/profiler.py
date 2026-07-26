"""Create the minimal image profile required by the reusable pilot builder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def laplacian_variance(gray: np.ndarray) -> float:
    values = gray.astype(np.float32)
    if values.shape[0] < 3 or values.shape[1] < 3:
        return 0.0
    center = values[1:-1, 1:-1]
    laplacian = (
        values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
        - 4 * center
    )
    return float(laplacian.var())


def profile_one_image(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        gray = np.clip(
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2],
            0,
            255,
        ).astype(np.uint8)
        clipped = np.any((rgb <= 2) | (rgb >= 253), axis=2)
        return {
            "readable": True,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "sha256": image_sha256(path),
            "brightness_mean": float(gray.mean()),
            "overexposed_fraction": float((gray >= 250).mean()),
            "channel_clip_fraction": float(clipped.mean()),
            "laplacian_variance": laplacian_variance(gray),
            "error": "",
        }
    except Exception as exc:
        return {
            "readable": False,
            "width": np.nan,
            "height": np.nan,
            "sha256": "",
            "brightness_mean": np.nan,
            "overexposed_fraction": np.nan,
            "channel_clip_fraction": np.nan,
            "laplacian_variance": np.nan,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_image_profile(
    manifest_path: Path,
    output_path: Path,
    participant_column: str = "participant_id",
    image_column: str = "image_path",
) -> pd.DataFrame:
    manifest_path = manifest_path.resolve()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {participant_column, image_column}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    if manifest[participant_column].str.strip().eq("").any():
        raise ValueError("Every image requires a trusted participant identifier")
    if manifest[image_column].str.strip().eq("").any():
        raise ValueError("Every manifest row requires an image path")

    rows = []
    for _, source in manifest.iterrows():
        path = Path(source[image_column]).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        record = {
            "participant_id": source[participant_column],
            "image_path": str(path),
        }
        record.update(profile_one_image(path))
        rows.append(record)
    profile = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path, index=False)
    return profile
