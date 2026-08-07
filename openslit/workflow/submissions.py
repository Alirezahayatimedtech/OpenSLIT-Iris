"""Resolve versioned segmentation submissions without losing unchanged images."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .state import WorkflowState


@dataclass(frozen=True)
class ResolvedSegmentationMask:
    """The newest frozen mask available for one image."""

    image_id: str
    mask_file: str
    masks_dir: Path
    manifest_path: Path
    submission_version: int

    @property
    def path(self) -> Path:
        return self.masks_dir / self.mask_file


def resolve_segmentation_masks(
    state: WorkflowState,
    grader_id: str,
    required_image_ids: Iterable[str] | None = None,
) -> dict[str, ResolvedSegmentationMask]:
    """Overlay submission manifests in version order and return per-image masks.

    Revision submissions may contain only the images that changed. Earlier masks
    remain authoritative for every image absent from a later submission.
    """

    submissions = state.grader_state(grader_id).get("segmentation_submissions", [])
    if not submissions:
        raise RuntimeError(f"No frozen segmentation submission exists for {grader_id}")

    versions: set[int] = set()
    ordered: list[tuple[int, int, dict[str, Any]]] = []
    for position, submission in enumerate(submissions):
        version = int(submission.get("version", position + 1))
        if version in versions:
            raise ValueError(
                f"Duplicate segmentation submission version {version} for {grader_id}"
            )
        versions.add(version)
        ordered.append((version, position, submission))

    resolved: dict[str, ResolvedSegmentationMask] = {}
    for version, _, submission in sorted(ordered):
        manifest_path = Path(str(submission["manifest_path"]))
        masks_dir = Path(str(submission["masks_dir"]))
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if not masks_dir.is_dir():
            raise FileNotFoundError(masks_dir)
        manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        required = {"image_id", "mask_file"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(
                f"Segmentation submission v{version} for {grader_id} is missing "
                f"columns: {sorted(missing)}"
            )
        if manifest["image_id"].duplicated().any():
            raise ValueError(
                f"Segmentation submission v{version} for {grader_id} contains "
                "duplicate image IDs"
            )
        for row in manifest.itertuples(index=False):
            image_id = str(row.image_id)
            mask_file = str(row.mask_file)
            item = ResolvedSegmentationMask(
                image_id=image_id,
                mask_file=mask_file,
                masks_dir=masks_dir,
                manifest_path=manifest_path,
                submission_version=version,
            )
            if not item.path.is_file():
                raise FileNotFoundError(item.path)
            resolved[image_id] = item

    if required_image_ids is not None:
        required_ids = {str(value) for value in required_image_ids}
        missing_ids = sorted(required_ids - set(resolved))
        if missing_ids:
            raise ValueError(
                f"Frozen submissions for {grader_id} are missing images: "
                f"{missing_ids[:20]}"
            )
    return resolved
