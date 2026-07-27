"""Generate a compact machine-readable feature dictionary."""

from __future__ import annotations

from typing import Any

import pandas as pd


PREFIX_DESCRIPTIONS = {
    "pupil_": "Pupil geometry derived from the pupil segmentation mask.",
    "iris_visible_": "Geometry of the visible iris segmentation mask.",
    "iris_raw_": "Color statistic measured from unnormalized RGB image pixels.",
    "iris_normalized_": "Color statistic after configured illumination normalization.",
    "polar_": "Statistic measured in the normalized polar iris representation.",
    "texture_": "Texture statistic measured on the valid normalized iris strip.",
    "lbp_": "Local binary pattern texture statistic.",
    "glcm_": "Gray-level co-occurrence matrix texture statistic.",
    "haar_": "Haar wavelet detail energy.",
    "image_": "Image-quality or acquisition statistic.",
    "feature_": "Feature-extraction quality gate or provenance field.",
    "source_": "Source mask provenance field.",
}

UNIT_SUFFIXES = {
    "_px": "pixels",
    "_area_px": "pixels^2",
    "_fraction": "proportion",
    "_ratio": "ratio",
    "_deg": "degrees",
    "_seconds": "seconds",
}


def infer_unit(name: str) -> str:
    for suffix, unit in UNIT_SUFFIXES.items():
        if name.endswith(suffix):
            return unit
    if any(
        token in name
        for token in [
            "dice",
            "iou",
            "entropy",
            "energy",
            "circularity",
            "eccentricity",
            "correlation",
            "homogeneity",
            "contrast",
            "std",
            "mean",
            "median",
            "p10",
            "p90",
        ]
    ):
        return "unitless"
    return ""


def describe_feature(name: str) -> str:
    for prefix, description in PREFIX_DESCRIPTIONS.items():
        if name.startswith(prefix):
            return description
    descriptions: dict[str, str] = {
        "image_id": "Aliased image identifier.",
        "image_file": "Aliased image filename.",
        "mask_file": "Versioned segmentation-mask filename.",
        "blinded_patient_id": "Aliased participant identifier used for grouped splitting and repeatability.",
        "feature_version": "Version of the feature definition and extraction configuration.",
        "feature_source": "Provenance category of the segmentation mask.",
    }
    return descriptions.get(
        name,
        "Derived OpenSLIT-Iris feature or provenance field.",
    )


def build_feature_dictionary(
    columns: list[str],
    feature_version: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        rows.append(
            {
                "feature_name": column,
                "feature_version": feature_version,
                "unit": infer_unit(column),
                "description": describe_feature(column),
            }
        )
    return pd.DataFrame(rows)
