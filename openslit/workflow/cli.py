"""Command-line interface for the end-to-end grader workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adjudication import (
    build_mask_disagreement_package,
    finalize_adjudication,
    record_revision_request,
)
from .config import load_workflow_config
from .cvat_bridge import (
    create_revision_task,
    export_and_freeze_segmentation,
    setup_independent_cvat_projects,
)
from .google_drive import (
    bootstrap_drive,
    freeze_grading_submission,
    upload_adjudication_package,
)
from .state import WorkflowState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openslit-workflow",
        description=(
            "Run the blinded OpenSLIT-Iris workflow: Drive grading, isolated CVAT "
            "segmentation, disagreement analysis, revision, and adjudication."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/workflow_pilot_v1.json"),
        help="Workflow JSON configuration",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Validate configuration and pilot files")
    check.add_argument(
        "--configuration-only",
        action="store_true",
        help="Do not require generated local pilot files",
    )

    commands.add_parser("status", help="Print current workflow state")
    commands.add_parser(
        "bootstrap-drive",
        help="Create the blinded Google Drive folders, images, and private grader Sheets",
    )

    freeze_grading = commands.add_parser(
        "freeze-grading",
        help="Export, validate, hash, and lock one grader's Google Sheet",
    )
    freeze_grading.add_argument("--grader", required=True)

    setup_cvat = commands.add_parser(
        "setup-cvat",
        help="Create one isolated CVAT project/task per grader after grading is frozen",
    )
    setup_cvat.add_argument("--allow-existing", action="store_true")

    freeze_segmentation = commands.add_parser(
        "freeze-segmentation",
        help="Export, normalize, validate, hash, and freeze one CVAT submission",
    )
    freeze_segmentation.add_argument("--grader", required=True)

    build_adjudication = commands.add_parser(
        "build-adjudication",
        help="Compare both frozen submissions and generate senior-review reports",
    )
    build_adjudication.add_argument("--output", type=Path, default=None)

    upload_adjudication = commands.add_parser(
        "upload-adjudication",
        help="Upload the generated senior package and create the adjudication Sheet",
    )
    upload_adjudication.add_argument("--package-dir", type=Path, default=None)
    upload_adjudication.add_argument("--queue", type=Path, default=None)

    revision = commands.add_parser(
        "request-revision",
        help="Record a versioned senior request without overwriting original masks",
    )
    revision.add_argument("--image-id", required=True)
    revision.add_argument("--from-grader", required=True)
    revision.add_argument("--reason", required=True)
    revision.add_argument("--protocol-reference", default="")

    revision_task = commands.add_parser(
        "create-revision-task",
        help="Create a pre-populated CVAT task for a grader's open revision requests",
    )
    revision_task.add_argument("--grader", required=True)

    finalize = commands.add_parser(
        "finalize-adjudication",
        help="Validate completed senior decisions and freeze the final record",
    )
    finalize.add_argument("--queue", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_workflow_config(args.config)

    if args.command == "check":
        print(
            json.dumps(
                config.validate(
                    require_runtime_files=not args.configuration_only,
                ),
                indent=2,
            )
        )
        return

    state = WorkflowState.load_or_create(config)
    if args.command == "status":
        print(json.dumps(state.summary(), indent=2))
        return
    if args.command == "bootstrap-drive":
        result = bootstrap_drive(config, state)
    elif args.command == "freeze-grading":
        result = freeze_grading_submission(config, state, args.grader)
    elif args.command == "setup-cvat":
        result = setup_independent_cvat_projects(
            config,
            state,
            allow_existing=args.allow_existing,
        )
    elif args.command == "freeze-segmentation":
        result = export_and_freeze_segmentation(config, state, args.grader)
    elif args.command == "build-adjudication":
        result = build_mask_disagreement_package(config, state, args.output)
    elif args.command == "upload-adjudication":
        package_dir = args.package_dir or Path(
            state.data["adjudication"].get("package_dir") or ""
        )
        queue = args.queue or Path(
            state.data["adjudication"].get("mask_queue_path") or ""
        )
        if not package_dir.is_dir() or not queue.is_file():
            raise SystemExit(
                "Build the adjudication package first or supply --package-dir and --queue."
            )
        result = upload_adjudication_package(
            config,
            state,
            package_dir=package_dir,
            queue_csv=queue,
        )
    elif args.command == "request-revision":
        result = record_revision_request(
            config,
            state,
            image_id=args.image_id,
            requested_from=args.from_grader,
            reason=args.reason,
            protocol_reference=args.protocol_reference,
        )
    elif args.command == "create-revision-task":
        result = create_revision_task(config, state, args.grader)
    elif args.command == "finalize-adjudication":
        result = finalize_adjudication(config, state, args.queue)
    else:  # pragma: no cover
        raise AssertionError(args.command)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
