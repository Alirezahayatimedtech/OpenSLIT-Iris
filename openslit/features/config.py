"""Configuration for versioned quantitative iris feature extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceRequirements:
    allowed_review_status: tuple[str, ...]
    require_gradable: bool
    require_mask_sha256: bool


@dataclass(frozen=True)
class NormalizationConfig:
    angular_samples: int
    radial_samples: int
    minimum_valid_angle_fraction: float


@dataclass(frozen=True)
class ColorConfig:
    normalization: str
    radial_zones: int
    angular_sectors: int


@dataclass(frozen=True)
class TextureConfig:
    gray_levels: int
    glcm_distances: tuple[int, ...]
    glcm_angles_degrees: tuple[int, ...]
    lbp_points: int
    lbp_radius: int
    haar_levels: int


@dataclass(frozen=True)
class QualityConfig:
    minimum_visible_iris_pixels: int
    minimum_valid_polar_fraction: float
    maximum_uncertain_fraction: float
    maximum_artifact_fraction: float
    minimum_laplacian_variance: float


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool
    max_images: int


@dataclass(frozen=True)
class DriveFeatureConfig:
    upload_enabled: bool
    folder_name: str
    grader_role: str
    adjudicator_role: str


@dataclass(frozen=True)
class FeatureExtractionConfig:
    config_path: Path
    workflow_config_path: Path
    feature_version: str
    image_dir: Path
    manifest_path: Path
    masks_dir: Path
    schema_path: Path
    output_dir: Path
    source_requirements: SourceRequirements
    normalization: NormalizationConfig
    color: ColorConfig
    texture: TextureConfig
    quality: QualityConfig
    preview: PreviewConfig
    drive: DriveFeatureConfig

    def validate(self, require_runtime_files: bool = True) -> dict[str, Any]:
        if not self.feature_version.strip():
            raise ValueError("feature_version cannot be empty")
        if self.normalization.angular_samples < 32:
            raise ValueError("angular_samples must be at least 32")
        if self.normalization.radial_samples < 16:
            raise ValueError("radial_samples must be at least 16")
        if not 0 < self.normalization.minimum_valid_angle_fraction <= 1:
            raise ValueError("minimum_valid_angle_fraction must be within (0, 1]")
        if self.color.normalization not in {"none", "gray_world"}:
            raise ValueError("color.normalization must be 'none' or 'gray_world'")
        if self.color.radial_zones < 1:
            raise ValueError("radial_zones must be positive")
        if self.color.angular_sectors < 2 or self.color.angular_sectors % 2:
            raise ValueError("angular_sectors must be an even integer >= 2")
        if not 2 <= self.texture.gray_levels <= 256:
            raise ValueError("texture.gray_levels must be between 2 and 256")
        if any(value < 1 for value in self.texture.glcm_distances):
            raise ValueError("GLCM distances must be positive")
        if self.texture.lbp_points != 8 or self.texture.lbp_radius != 1:
            raise ValueError("Feature protocol v1 currently supports LBP(8,1) only")
        if self.texture.haar_levels < 1:
            raise ValueError("haar_levels must be positive")
        if self.preview.max_images < 0:
            raise ValueError("preview.max_images cannot be negative")
        for role in [self.drive.grader_role, self.drive.adjudicator_role]:
            if role not in {"reader", "writer"}:
                raise ValueError("Drive roles must be reader or writer")

        missing: list[str] = []
        if require_runtime_files:
            for path in [
                self.workflow_config_path,
                self.manifest_path,
                self.schema_path,
            ]:
                if not path.is_file():
                    missing.append(str(path))
            for path in [self.image_dir, self.masks_dir]:
                if not path.is_dir():
                    missing.append(str(path))
            if missing:
                raise FileNotFoundError(f"Missing feature-extraction inputs: {missing}")

        return {
            "feature_version": self.feature_version,
            "image_dir": str(self.image_dir),
            "manifest_path": str(self.manifest_path),
            "masks_dir": str(self.masks_dir),
            "output_dir": str(self.output_dir),
            "normalization": {
                "angular_samples": self.normalization.angular_samples,
                "radial_samples": self.normalization.radial_samples,
            },
            "color_normalization": self.color.normalization,
            "drive_upload_enabled": self.drive.upload_enabled,
            "runtime_files_checked": require_runtime_files,
        }


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_feature_config(path: str | Path) -> FeatureExtractionConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    source = raw.get("source_requirements", {})
    normalization = raw.get("normalization", {})
    color = raw.get("color", {})
    texture = raw.get("texture", {})
    quality = raw.get("quality", {})
    preview = raw.get("preview", {})
    drive = raw.get("drive", {})
    return FeatureExtractionConfig(
        config_path=config_path,
        workflow_config_path=_resolve(base, str(raw["workflow_config_path"])),
        feature_version=str(raw["feature_version"]).strip(),
        image_dir=_resolve(base, str(raw["image_dir"])),
        manifest_path=_resolve(base, str(raw["manifest_path"])),
        masks_dir=_resolve(base, str(raw["masks_dir"])),
        schema_path=_resolve(base, str(raw["schema_path"])),
        output_dir=_resolve(base, str(raw["output_dir"])),
        source_requirements=SourceRequirements(
            allowed_review_status=tuple(
                str(value).strip() for value in source.get("allowed_review_status", [])
            ),
            require_gradable=bool(source.get("require_gradable", True)),
            require_mask_sha256=bool(source.get("require_mask_sha256", True)),
        ),
        normalization=NormalizationConfig(
            angular_samples=int(normalization.get("angular_samples", 360)),
            radial_samples=int(normalization.get("radial_samples", 64)),
            minimum_valid_angle_fraction=float(
                normalization.get("minimum_valid_angle_fraction", 0.35)
            ),
        ),
        color=ColorConfig(
            normalization=str(color.get("normalization", "gray_world")),
            radial_zones=int(color.get("radial_zones", 3)),
            angular_sectors=int(color.get("angular_sectors", 8)),
        ),
        texture=TextureConfig(
            gray_levels=int(texture.get("gray_levels", 16)),
            glcm_distances=tuple(
                int(value) for value in texture.get("glcm_distances", [1, 2])
            ),
            glcm_angles_degrees=tuple(
                int(value)
                for value in texture.get("glcm_angles_degrees", [0, 45, 90, 135])
            ),
            lbp_points=int(texture.get("lbp_points", 8)),
            lbp_radius=int(texture.get("lbp_radius", 1)),
            haar_levels=int(texture.get("haar_levels", 2)),
        ),
        quality=QualityConfig(
            minimum_visible_iris_pixels=int(
                quality.get("minimum_visible_iris_pixels", 500)
            ),
            minimum_valid_polar_fraction=float(
                quality.get("minimum_valid_polar_fraction", 0.30)
            ),
            maximum_uncertain_fraction=float(
                quality.get("maximum_uncertain_fraction", 0.20)
            ),
            maximum_artifact_fraction=float(
                quality.get("maximum_artifact_fraction", 0.45)
            ),
            minimum_laplacian_variance=float(
                quality.get("minimum_laplacian_variance", 15.0)
            ),
        ),
        preview=PreviewConfig(
            enabled=bool(preview.get("enabled", True)),
            max_images=int(preview.get("max_images", 50)),
        ),
        drive=DriveFeatureConfig(
            upload_enabled=bool(drive.get("upload_enabled", True)),
            folder_name=str(drive.get("folder_name", "06_Feature_Extraction")),
            grader_role=str(drive.get("grader_role", "reader")),
            adjudicator_role=str(drive.get("adjudicator_role", "writer")),
        ),
    )
