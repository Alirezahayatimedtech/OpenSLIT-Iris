"""Command-line interface for collaborative pilot work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .pilot import build_pilot, load_config
from .profiler import build_image_profile
from .validation import merge_submissions, validate_submission
from .workbook import apply_drive_links, apply_drive_links_to_csv


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="openslit-collab")
    commands = root.add_subparsers(dest="command", required=True)

    profile = commands.add_parser(
        "profile", help="Create a minimal pilot image profile from a manifest"
    )
    profile.add_argument("--manifest", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--participant-column", default="participant_id")
    profile.add_argument("--image-column", default="image_path")

    build = commands.add_parser("build", help="Build a blinded pilot package")
    build.add_argument("--config", type=Path, required=True)

    links = commands.add_parser(
        "apply-links", help="Insert Drive URLs into grader workbooks"
    )
    links.add_argument("--links", type=Path, required=True)
    links.add_argument("--workbook", type=Path, action="append", required=True)
    links.add_argument(
        "--index",
        type=Path,
        action="append",
        help="Optionally update one or more shared CSV manifests with the same URLs",
    )

    validate = commands.add_parser(
        "validate", help="Validate one grader submission"
    )
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--allow-incomplete", action="store_true")

    merge = commands.add_parser(
        "merge", help="Merge two independent grader submissions"
    )
    merge.add_argument("--first", type=Path, required=True)
    merge.add_argument("--second", type=Path, required=True)
    merge.add_argument("--index", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "profile":
        profile = build_image_profile(
            args.manifest,
            args.output,
            participant_column=args.participant_column,
            image_column=args.image_column,
        )
        print(
            json.dumps(
                {
                    "rows": len(profile),
                    "participants": int(profile["participant_id"].nunique()),
                    "readable": int(profile["readable"].sum()),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return
    if args.command == "build":
        manifest = build_pilot(load_config(args.config))
        print(json.dumps(manifest, indent=2))
        return
    if args.command == "apply-links":
        links = pd.read_csv(args.links, dtype=str, keep_default_na=False)
        for workbook in args.workbook:
            apply_drive_links(workbook, links)
            print(f"updated {workbook}")
        for index in args.index or []:
            apply_drive_links_to_csv(index, links)
            print(f"updated {index}")
        return
    if args.command == "validate":
        _, errors = validate_submission(
            args.submission, args.index, not args.allow_incomplete
        )
        if errors:
            raise SystemExit("\n".join(errors))
        print("valid")
        return
    if args.command == "merge":
        metrics = merge_submissions(
            args.first, args.second, args.index, args.output
        )
        print(json.dumps(metrics, indent=2))
