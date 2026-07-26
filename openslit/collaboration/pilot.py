"""Build a blinded and reproducible collaborative pilot package."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import FORBIDDEN_SHARED_COLUMNS
from .workbook import write_grader_workbook


REQUIRED_PROFILE_COLUMNS = {
    "participant_id",
    "image_path",
    "sha256",
    "readable",
    "brightness_mean",
    "overexposed_fraction",
    "channel_clip_fraction",
    "laplacian_variance",
}

CHALLENGE_CRITERIA = [
    ("dark", "brightness_mean", True),
    ("bright", "brightness_mean", False),
    ("blur", "laplacian_variance", True),
    ("clipping", "channel_clip_fraction", False),
    ("overexposure", "overexposed_fraction", False),
]


@dataclass(frozen=True)
class PilotConfig:
    image_profile: Path
    source_manifest: Path
    output_dir: Path
    seed: int = 20260726
    distribution_participants: int = 40
    challenge_participants: int = 10
    double_mask_images: int = 20
    grader_ids: tuple[str, ...] = ("grader_01", "grader_02")
    copy_images: bool = True

    @property
    def total_images(self) -> int:
        return self.distribution_participants + self.challenge_participants


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_token(seed: int, *values: object) -> str:
    payload = "|".join([str(seed), *[str(value) for value in values]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path) -> PilotConfig:
    raw: dict[str, Any] = json.loads(path.read_text())
    base = path.resolve().parent

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    return PilotConfig(
        image_profile=resolve(raw["image_profile"]),
        source_manifest=resolve(raw["source_manifest"]),
        output_dir=resolve(raw["output_dir"]),
        seed=int(raw.get("seed", 20260726)),
        distribution_participants=int(raw.get("distribution_participants", 40)),
        challenge_participants=int(raw.get("challenge_participants", 10)),
        double_mask_images=int(raw.get("double_mask_images", 20)),
        grader_ids=tuple(raw.get("grader_ids", ["grader_01", "grader_02"])),
        copy_images=bool(raw.get("copy_images", True)),
    )


def validate_profile(profile: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PROFILE_COLUMNS - set(profile.columns)
    if missing:
        raise ValueError(f"Image profile is missing columns: {sorted(missing)}")
    data = profile.copy()
    data = data[data["readable"].astype(str).str.lower().isin({"true", "1"})]
    data["participant_id"] = data["participant_id"].astype(str)
    data["image_path"] = data["image_path"].map(str)
    data = data.sort_values(
        ["participant_id", "sha256", "image_path"], kind="mergesort"
    ).reset_index(drop=True)
    if data.empty:
        raise ValueError("No readable images remain after filtering")
    if data["sha256"].eq("").any() or data["sha256"].isna().any():
        raise ValueError("Every candidate image must have a SHA-256 value")
    return data


def remove_exact_duplicate_images(profile: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_sha = profile.groupby("sha256").size()
    duplicate_sha = set(duplicate_sha[duplicate_sha > 1].index)
    clean = profile[~profile["sha256"].isin(duplicate_sha)].copy()
    return clean, int(profile["sha256"].isin(duplicate_sha).sum())


def choose_random_panel(
    candidates: pd.DataFrame, count: int, seed: int
) -> pd.DataFrame:
    participants = sorted(candidates["participant_id"].unique())
    if count > len(participants):
        raise ValueError(
            f"Requested {count} distribution participants but only "
            f"{len(participants)} are available"
        )
    rng = np.random.default_rng(seed)
    selected_participants = rng.choice(participants, size=count, replace=False)
    rows = []
    for participant_id in selected_participants:
        group = candidates[candidates["participant_id"] == participant_id]
        row_index = int(rng.integers(0, len(group)))
        row = group.iloc[row_index].copy()
        row["selection_arm"] = "seeded_distribution"
        row["selection_reason"] = "seeded participant and image sample"
        rows.append(row)
    return pd.DataFrame(rows)


def choose_challenge_panel(
    candidates: pd.DataFrame,
    excluded_participants: set[str],
    count: int,
    seed: int,
) -> pd.DataFrame:
    remaining = candidates[
        ~candidates["participant_id"].isin(excluded_participants)
    ].copy()
    if remaining["participant_id"].nunique() < count:
        raise ValueError("Not enough independent participants for challenge panel")

    base, remainder = divmod(count, len(CHALLENGE_CRITERIA))
    quotas = {
        name: base + (index < remainder)
        for index, (name, _, _) in enumerate(CHALLENGE_CRITERIA)
    }
    chosen_rows: list[pd.Series] = []
    chosen_participants = set(excluded_participants)

    for criterion, column, ascending in CHALLENGE_CRITERIA:
        ranked = remaining.copy()
        ranked["_tie"] = ranked.apply(
            lambda row: stable_token(
                seed, criterion, row["participant_id"], row["sha256"]
            ),
            axis=1,
        )
        ranked = ranked.sort_values(
            [column, "_tie"],
            ascending=[ascending, True],
            kind="mergesort",
        )
        added = 0
        for _, row in ranked.iterrows():
            participant_id = row["participant_id"]
            if participant_id in chosen_participants:
                continue
            selected = row.drop(labels=["_tie"]).copy()
            selected["selection_arm"] = "technical_challenge"
            selected["selection_reason"] = criterion
            chosen_rows.append(selected)
            chosen_participants.add(participant_id)
            added += 1
            if added == quotas[criterion]:
                break
        if added != quotas[criterion]:
            raise ValueError(f"Could not fill challenge criterion: {criterion}")

    return pd.DataFrame(chosen_rows)


def assign_blinded_ids(selection: pd.DataFrame, seed: int) -> pd.DataFrame:
    blinded = selection.copy()
    blinded["_order"] = blinded.apply(
        lambda row: stable_token(
            seed, "blind", row["participant_id"], row["sha256"]
        ),
        axis=1,
    )
    blinded = blinded.sort_values("_order", kind="mergesort").reset_index(drop=True)
    blinded["blinded_patient_id"] = [
        f"PILOT-P{index:03d}" for index in range(1, len(blinded) + 1)
    ]
    blinded["blinded_image_id"] = [
        f"PILOT-I{index:03d}" for index in range(1, len(blinded) + 1)
    ]
    blinded["image_file"] = blinded.apply(
        lambda row: f"{row['blinded_image_id']}{Path(row['image_path']).suffix.lower()}",
        axis=1,
    )
    return blinded.drop(columns=["_order"])


def assign_double_mask_subset(
    selection: pd.DataFrame, count: int, seed: int
) -> pd.Series:
    if count < 0 or count > len(selection):
        raise ValueError("double_mask_images must be between zero and pilot size")
    order = selection["blinded_image_id"].map(
        lambda value: stable_token(seed, "double-mask", value)
    )
    selected = set(selection.loc[order.sort_values().index[:count], "blinded_image_id"])
    return selection["blinded_image_id"].isin(selected)


def build_shared_table(selection: pd.DataFrame) -> pd.DataFrame:
    shared = selection[
        [
            "blinded_image_id",
            "blinded_patient_id",
            "image_file",
        ]
    ].copy()
    shared["drive_url"] = ""
    if FORBIDDEN_SHARED_COLUMNS.intersection(shared.columns):
        raise AssertionError("A forbidden source field entered the shared table")
    return shared


def build_pilot(config: PilotConfig) -> dict[str, Any]:
    output = config.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty pilot directory: {output}. "
            "Use a new versioned output_dir so submitted grades cannot be lost."
        )

    profile = validate_profile(pd.read_csv(config.image_profile))
    profile, excluded_duplicate_images = remove_exact_duplicate_images(profile)
    if profile["participant_id"].nunique() < config.total_images:
        raise ValueError("Pilot size exceeds available independent participants")

    distribution = choose_random_panel(
        profile, config.distribution_participants, config.seed
    )
    challenge = choose_challenge_panel(
        profile,
        set(distribution["participant_id"]),
        config.challenge_participants,
        config.seed,
    )
    selection = pd.concat([distribution, challenge], ignore_index=True)
    selection = assign_blinded_ids(selection, config.seed)
    selection["double_mask_annotation"] = assign_double_mask_subset(
        selection, config.double_mask_images, config.seed
    )

    if selection["participant_id"].nunique() != len(selection):
        raise AssertionError("Pilot must contain exactly one image per participant")
    if selection["sha256"].nunique() != len(selection):
        raise AssertionError("Pilot contains an exact duplicate image")

    private_dir = output / "private"
    shared_dir = output / "shared"
    upload_dir = output / "drive_upload" / "images"
    for directory in [private_dir, shared_dir, upload_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    private_columns = [
        "blinded_image_id",
        "blinded_patient_id",
        "participant_id",
        "image_path",
        "image_file",
        "sha256",
        "selection_arm",
        "selection_reason",
        "brightness_mean",
        "laplacian_variance",
        "channel_clip_fraction",
        "overexposed_fraction",
        "double_mask_annotation",
    ]
    private_key = selection[private_columns].copy()
    private_key.to_csv(private_dir / "pilot_private_key.csv", index=False)

    participants = private_key[
        ["blinded_patient_id", "participant_id", "selection_arm"]
    ].sort_values("blinded_patient_id")
    participants.to_csv(private_dir / "selected_participants.csv", index=False)

    shared = build_shared_table(selection)
    shared.to_csv(shared_dir / "pilot_image_index.csv", index=False)
    shared[["blinded_image_id", "image_file", "drive_url"]].to_csv(
        shared_dir / "drive_links.csv", index=False
    )

    mask_tasks = shared.copy()
    mask_tasks["mask_file"] = mask_tasks["blinded_image_id"] + "_mask.png"
    double_map = selection.set_index("blinded_image_id")["double_mask_annotation"]
    mask_tasks["independent_double_annotation"] = mask_tasks[
        "blinded_image_id"
    ].map(double_map)
    mask_tasks["annotation_status"] = "not_started"
    mask_tasks["review_status"] = "not_reviewed"
    mask_tasks["annotator_id"] = ""
    mask_tasks["reviewer_id"] = ""
    mask_tasks["comments"] = ""
    mask_tasks.to_csv(shared_dir / "mask_task_manifest.csv", index=False)

    for grader_id in config.grader_ids:
        write_grader_workbook(
            shared,
            shared_dir / f"{grader_id}_quality_grading.xlsx",
            grader_id=grader_id,
        )

    if config.copy_images:
        for _, row in selection.iterrows():
            source = Path(row["image_path"])
            destination = upload_dir / row["image_file"]
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination)
            if sha256_file(destination) != row["sha256"]:
                raise RuntimeError(f"Copied image checksum mismatch: {destination}")

    manifest_hash = sha256_file(config.source_manifest)
    profile_hash = sha256_file(config.image_profile)
    private_key_hash = sha256_file(private_dir / "pilot_private_key.csv")
    run_manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "selection_design": {
            "seeded_distribution_participants": config.distribution_participants,
            "technical_challenge_participants": config.challenge_participants,
            "one_image_per_participant": True,
            "exact_duplicate_groups_excluded": True,
            "double_mask_images": config.double_mask_images,
            "quality_graders": list(config.grader_ids),
        },
        "source_manifest": str(config.source_manifest),
        "source_manifest_sha256": manifest_hash,
        "image_profile": str(config.image_profile),
        "image_profile_sha256": profile_hash,
        "pilot_private_key_sha256": private_key_hash,
        "selected_images": len(selection),
        "selected_participants": int(selection["participant_id"].nunique()),
        "excluded_images_from_exact_duplicate_groups": excluded_duplicate_images,
        "historical_view_labels_used": False,
        "historical_laterality_used": False,
        "forbidden_fields": sorted(FORBIDDEN_SHARED_COLUMNS),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    return run_manifest
