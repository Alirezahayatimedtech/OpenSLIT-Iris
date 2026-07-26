#!/usr/bin/env python3
"""Audit whether the local SLIT dataset can support OpenSLIT-Iris version 1.

The audit is read-only with respect to the source dataset. It profiles the
corrected 448 px release manifest, checks image integrity and coarse technical
quality, and writes compact evidence tables. Historical center/nasal/temporal
and left/right-derived labels are explicitly excluded from cohort selection and
feasibility claims. Heuristic image statistics are not clinical quality grades.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_RELEASE_MANIFEST = Path(
    "slit-project/paper2_runs/angle_closure_corrected_p2_views_80_20/"
    "corrected_release_view_image_manifest_80_20.csv"
)
DEFAULT_OUTPUT_DIR = Path("OpenSLIT-Iris/feasibility_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-image-scan",
        action="store_true",
        help="Skip JPEG decoding/hashing and use manifest-only checks.",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def dhash_hex(gray: np.ndarray) -> str:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def scan_image(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        exact_hash = hashlib.sha256(payload).hexdigest()
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        low_clip = np.any(rgb <= 2, axis=2)
        high_clip = np.any(rgb >= 253, axis=2)
        return {
            "exists": True,
            "readable": True,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "file_bytes": int(len(payload)),
            "brightness_mean": float(gray.mean()),
            "brightness_std": float(gray.std()),
            "underexposed_fraction": float((gray <= 5).mean()),
            "overexposed_fraction": float((gray >= 250).mean()),
            "channel_clip_fraction": float((low_clip | high_clip).mean()),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "sha256": exact_hash,
            "dhash": dhash_hex(gray),
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - exercised only by corrupt input
        return {
            "exists": path.exists(),
            "readable": False,
            "width": np.nan,
            "height": np.nan,
            "file_bytes": path.stat().st_size if path.exists() else 0,
            "brightness_mean": np.nan,
            "brightness_std": np.nan,
            "underexposed_fraction": np.nan,
            "overexposed_fraction": np.nan,
            "channel_clip_fraction": np.nan,
            "laplacian_variance": np.nan,
            "sha256": "",
            "dhash": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_metadata_matrix(release: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("image_id", "Derivable from release filename/index", 1.0, "ready"),
        ("filename", "image_path basename", 1.0, "ready"),
        ("patient_id", "participant_id", release["participant_id"].notna().mean(), "ready"),
        (
            "eye_id",
            "Historical combo_key exists but is excluded by user instruction",
            np.nan,
            "untrusted; rebuild from source",
        ),
        (
            "laterality",
            "Historical eye_code exists but is excluded by user instruction",
            np.nan,
            "untrusted; rebuild from source",
        ),
        (
            "visit_id",
            "No slit-photo visit identifier",
            0.0,
            "missing; cannot separate sessions",
        ),
        (
            "acquisition_date_or_period",
            "No verified slit-photo acquisition timestamp in release manifest",
            0.0,
            "missing",
        ),
        (
            "device_id",
            "CASIA2 metadata describes paired AS-OCT, not the slit-lamp camera",
            0.0,
            "missing for slit photographs",
        ),
        ("site_id", "No slit-photo site identifier", 0.0, "missing"),
        ("age", "Available only through a separate local eye table/crosswalk", np.nan, "partial"),
        ("sex", "Available only through a separate local eye table/crosswalk", np.nan, "partial"),
        ("diagnosis", "No iris diagnosis field", 0.0, "missing"),
        ("dilation_status", "No field", 0.0, "missing"),
        (
            "illumination_type",
            "Transferred view labels indicate beam position, not calibrated illumination mode",
            0.0,
            "missing",
        ),
        ("magnification", "No field", 0.0, "missing"),
        ("camera_model", "No field", 0.0, "missing"),
        ("slit_lamp_model", "No field", 0.0, "missing"),
        ("image_resolution", "Derivable from JPEG files", 1.0, "ready"),
        ("image_quality_grade", "No expert quality grade", 0.0, "missing"),
        ("annotator_id", "No segmentation annotator field", 0.0, "missing"),
        ("annotation_status", "No compatible multiclass segmentation masks", 0.0, "missing"),
        (
            "repeat_image_group",
            "Participant grouping exists; trusted eye/session grouping does not",
            np.nan,
            "not ready",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["spec_field", "available_source", "row_coverage", "assessment"]
    )


def build_requirement_matrix() -> pd.DataFrame:
    rows = [
        (
            "Frontal/nearly frontal images",
            "unknown until relabeling",
            "Historical center/nasal/temporal labels are excluded as unreliable",
        ),
        (
            "Diffuse/broad-beam input",
            "unknown until relabeling",
            "Illumination eligibility requires new blinded manual review",
        ),
        (
            "Patient grouping",
            "ready",
            "participant_id is complete and supports patient-disjoint splitting",
        ),
        (
            "Eye/laterality grouping",
            "not ready",
            "Historical eye and center/nasal/temporal assignments are excluded as unreliable",
        ),
        (
            "Patient-level split",
            "ready with revision",
            "Existing 80/20 split is patient-disjoint, but a new immutable OpenSLIT split is required",
        ),
        (
            "Multiclass pupil/iris/artefact masks",
            "not ready",
            "No compatible masks for the required classes",
        ),
        (
            "Expert annotation protocol",
            "not ready",
            "Current labels describe acquisition view, not tissue boundaries",
        ),
        (
            "Image QC labels",
            "not ready",
            "No expert A/B/C/D grades or region-visibility labels",
        ),
        (
            "Geometry features",
            "feasible after segmentation",
            "Pixel and ratio measurements are supportable; millimeters are not",
        ),
        (
            "Color/pigmentation features",
            "limited",
            "Color calibration, white balance, device, and illumination metadata are absent",
        ),
        (
            "Classical texture",
            "pilot only",
            "448 px JPEG derivatives can support coarse texture, not validated fine microstructure",
        ),
        (
            "Crypt/furrow/atrophy features",
            "experimental/high risk",
            "Resolution and absent expert structure labels prevent current validation",
        ),
        (
            "Same-eye repeatability",
            "not ready",
            "Repeated images exist, but trusted eye/session grouping does not",
        ),
        (
            "Longitudinal validation",
            "not ready",
            "No verified slit-photo visit or acquisition date in the release manifest",
        ),
        (
            "Device/site generalization",
            "not ready",
            "Slit-camera and site identifiers are absent",
        ),
        (
            "Pigmentation subgroup analysis",
            "not ready",
            "No expert iris-color labels",
        ),
    ]
    return pd.DataFrame(rows, columns=["requirement", "status", "evidence"])


def duplicate_summary(image_profile: pd.DataFrame) -> dict[str, Any]:
    readable = image_profile[image_profile["readable"]].copy()
    exact_counts = readable["sha256"].value_counts()
    dhash_counts = readable["dhash"].value_counts()
    exact_groups = set(exact_counts[exact_counts > 1].index)
    dhash_groups = set(dhash_counts[dhash_counts > 1].index)

    exact_cross_patient = 0
    exact_affected_participants = 0
    if exact_groups:
        exact_rows = readable[readable["sha256"].isin(exact_groups)]
        exact_cross_patient = int(
            (exact_rows.groupby("sha256")["participant_id"].nunique() > 1).sum()
        )
        exact_affected_participants = int(exact_rows["participant_id"].nunique())
    dhash_cross_patient = 0
    if dhash_groups:
        dhash_cross_patient = int(
            (
                readable[readable["dhash"].isin(dhash_groups)]
                .groupby("dhash")["participant_id"]
                .nunique()
                > 1
            ).sum()
        )
    return {
        "exact_duplicate_hash_groups": int(len(exact_groups)),
        "images_in_exact_duplicate_groups": int(readable["sha256"].isin(exact_groups).sum()),
        "exact_duplicate_groups_crossing_participants": exact_cross_patient,
        "participants_affected_by_exact_duplicates": exact_affected_participants,
        "identical_dhash_groups": int(len(dhash_groups)),
        "images_in_identical_dhash_groups": int(readable["dhash"].isin(dhash_groups).sum()),
        "identical_dhash_groups_crossing_participants": dhash_cross_patient,
        "dhash_caution": (
            "Identical 64-bit difference hashes are a screening signal, not proof of duplicate images."
        ),
    }


def quality_profile(image_profile: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "brightness_mean",
        "brightness_std",
        "underexposed_fraction",
        "overexposed_fraction",
        "channel_clip_fraction",
        "laplacian_variance",
        "file_bytes",
    ]
    readable = image_profile[image_profile["readable"]]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = pd.to_numeric(readable[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "nonmissing": int(len(values)),
                "p05": float(values.quantile(0.05)),
                "median": float(values.quantile(0.5)),
                "p95": float(values.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def select_pilot(image_profile: pd.DataFrame) -> pd.DataFrame:
    candidates = image_profile[image_profile["readable"]].copy()
    candidates["brightness_bin"] = pd.qcut(
        candidates["brightness_mean"], 5, labels=False, duplicates="drop"
    )
    candidates["sharpness_bin"] = pd.qcut(
        candidates["laplacian_variance"], 5, labels=False, duplicates="drop"
    )
    candidates["clipping_bin"] = pd.qcut(
        candidates["channel_clip_fraction"], 3, labels=False, duplicates="drop"
    )
    candidates["technical_stratum"] = candidates.apply(
        lambda row: (
            f"brightness_q{int(row.brightness_bin) + 1}-"
            f"sharpness_q{int(row.sharpness_bin) + 1}-"
            f"clipping_q{int(row.clipping_bin) + 1}"
        ),
        axis=1,
    )
    candidates = candidates.sort_values(
        ["technical_stratum", "participant_id", "image_index", "image_path"]
    )
    first_per_stratum = candidates.groupby("technical_stratum", as_index=False).head(1)
    pilot = first_per_stratum.drop_duplicates("participant_id").head(50)
    if len(pilot) < 50:
        remaining = candidates[
            ~candidates["image_path"].isin(pilot["image_path"])
            & ~candidates["participant_id"].isin(pilot["participant_id"])
        ]
        pilot = pd.concat(
            [pilot, remaining.drop_duplicates("participant_id").head(50 - len(pilot))]
        )
    if len(pilot) < 50:
        remaining = candidates[~candidates["image_path"].isin(pilot["image_path"])]
        pilot = pd.concat([pilot, remaining.head(50 - len(pilot))])
    pilot = pilot.head(50).copy()
    pilot["pilot_role"] = "blinded acquisition-eligibility and segmentation pilot"
    keep = [
        "participant_id",
        "image_index",
        "image_path",
        "pilot_role",
        "technical_stratum",
        "brightness_mean",
        "channel_clip_fraction",
        "laplacian_variance",
    ]
    return pilot[keep]


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    release_manifest_path = resolve(repo_root, args.release_manifest)
    output_dir = resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    release = pd.read_csv(release_manifest_path)
    required = {
        "participant_id",
        "image_index",
        "image_path",
        "split",
    }
    missing = required.difference(release.columns)
    if missing:
        raise ValueError(f"Release manifest missing columns: {sorted(missing)}")
    if release["image_path"].duplicated().any():
        raise ValueError("Release manifest contains duplicate image paths")

    release["image_path"] = release["image_path"].map(str)
    release["resolved_image_path"] = release["image_path"].map(
        lambda value: str(resolve(repo_root, Path(value)))
    )
    release["path_exists"] = release["resolved_image_path"].map(lambda value: Path(value).is_file())

    profile_rows: list[dict[str, Any]] = []
    if args.skip_image_scan:
        for row in release.itertuples(index=False):
            profile_rows.append(
                {
                    "participant_id": row.participant_id,
                    "image_index": row.image_index,
                    "image_path": row.image_path,
                    "exists": bool(row.path_exists),
                    "readable": np.nan,
                }
            )
    else:
        for position, row in enumerate(release.itertuples(index=False), start=1):
            metrics = scan_image(Path(row.resolved_image_path))
            metrics.update(
                {
                    "participant_id": row.participant_id,
                    "image_index": row.image_index,
                    "image_path": row.image_path,
                }
            )
            profile_rows.append(metrics)
            if position % 2000 == 0:
                print(f"Scanned {position:,}/{len(release):,} images")

    image_profile = pd.DataFrame(profile_rows)
    metadata = build_metadata_matrix(release)
    requirements = build_requirement_matrix()
    images_per_participant = release.groupby("participant_id").size()
    summary: dict[str, Any] = {
        "source_manifest": str(args.release_manifest),
        "images": int(len(release)),
        "participants": int(release["participant_id"].nunique()),
        "historical_eye_groups_excluded_from_analysis": True,
        "duplicate_image_paths": int(release["image_path"].duplicated().sum()),
        "participant_overlap_between_existing_splits": int(
            len(
                set(release.loc[release["split"].eq("train"), "participant_id"]).intersection(
                    set(release.loc[release["split"].eq("val"), "participant_id"])
                )
            )
        ),
        "images_per_participant": {
            "median": float(images_per_participant.median()),
            "p05": float(images_per_participant.quantile(0.05)),
            "p95": float(images_per_participant.quantile(0.95)),
            "max": int(images_per_participant.max()),
        },
        "historical_view_and_laterality_labels_used": False,
        "compatible_multiclass_segmentation_masks": 0,
        "compatible_expert_quality_grades": 0,
        "feasibility_decision": (
            "GO for a bounded Phase 0/1 pilot using a new blinded eligibility review; "
            "NO-GO for direct feature extraction or validation until expert masks and QC labels exist."
        ),
    }

    if not args.skip_image_scan:
        summary["image_integrity"] = {
            "files_present": int(image_profile["exists"].sum()),
            "files_readable": int(image_profile["readable"].sum()),
            "corrupt_or_unreadable": int((~image_profile["readable"]).sum()),
            "resolution_counts": {
                f"{int(width)}x{int(height)}": int(count)
                for (width, height), count in image_profile[
                    image_profile["readable"]
                ]
                .groupby(["width", "height"])
                .size()
                .items()
            },
        }
        summary["duplicate_screen"] = duplicate_summary(image_profile)
        quality = quality_profile(image_profile)
        pilot = select_pilot(image_profile)
        quality.to_csv(output_dir / "image_quality_profile.csv", index=False)
        pilot.to_csv(output_dir / "pilot_selection_50_label_free.csv", index=False)
        image_profile.to_csv(output_dir / "image_profile.csv", index=False)

    metadata.to_csv(output_dir / "metadata_coverage.csv", index=False)
    requirements.to_csv(output_dir / "requirement_matrix.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
