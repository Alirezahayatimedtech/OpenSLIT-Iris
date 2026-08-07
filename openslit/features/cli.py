"""Command-line interface for OpenSLIT quantitative iris phenotyping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openslit.workflow.config import load_workflow_config
from openslit.workflow.state import WorkflowState

from .config import load_feature_config
from .drive import upload_feature_run
from .extract import extract_feature_table
from .repeatability import analyze_repeatability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openslit-features",
        description=(
            "Extract versioned iris geometry, color, texture, normalization, and "
            "quality features from approved segmentation masks."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/feature_extraction_v1.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Validate feature configuration")
    check.add_argument("--configuration-only", action="store_true")
    commands.add_parser("status", help="Show feature-extraction workflow state")

    extract = commands.add_parser("extract", help="Run feature extraction")
    extract.add_argument("--run-id", default=None)

    repeatability = commands.add_parser(
        "repeatability",
        help=(
            "Calculate ICC, within-group CV, repeatability coefficient, and "
            "Bland-Altman summaries"
        ),
    )
    repeatability.add_argument("--features", type=Path, required=True)
    repeatability.add_argument("--group-column", required=True)
    repeatability.add_argument("--output-dir", type=Path, required=True)
    repeatability.add_argument("--feature", action="append", default=None)

    upload = commands.add_parser(
        "upload-drive",
        help="Upload feature tables, reports, manifests, and previews to Drive",
    )
    upload.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_feature_config(args.config)
    if args.command == "check":
        result = config.validate(require_runtime_files=not args.configuration_only)
    elif args.command == "status":
        workflow_config = load_workflow_config(config.workflow_config_path)
        state = WorkflowState.load_or_create(workflow_config)
        result = state.data.get(
            "features",
            {"status": "LOCKED_UNTIL_FINAL_MASKS", "runs": []},
        )
    elif args.command == "extract":
        result = extract_feature_table(
            config,
            run_id=args.run_id,
        )
    elif args.command == "repeatability":
        result = analyze_repeatability(
            args.features,
            args.group_column,
            args.output_dir,
            feature_columns=args.feature,
        )
    elif args.command == "upload-drive":
        result = upload_feature_run(config, run_id=args.run_id)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
