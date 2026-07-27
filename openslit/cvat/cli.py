"""Command-line interface for local CVAT setup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .api import setup_cvat_workspace
from .config import load_cvat_setup_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openslit-cvat",
        description="Validate or create the OpenSLIT-Iris CVAT annotation workspace.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    check = subcommands.add_parser(
        "check", help="Validate local files and print the planned CVAT setup"
    )
    check.add_argument("--config", type=Path, required=True)

    setup = subcommands.add_parser(
        "setup", help="Create the CVAT project and independent annotation tasks"
    )
    setup.add_argument("--config", type=Path, required=True)
    setup.add_argument(
        "--host",
        default=None,
        help="CVAT base URL; defaults to CVAT_BASE_URL or http://localhost:8080",
    )
    setup.add_argument(
        "--allow-existing",
        action="store_true",
        help="Reuse an existing matching project and skip existing matching tasks",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_cvat_setup_config(args.config)

    if args.command == "check":
        print(json.dumps(config.validate(), indent=2))
        return

    host = args.host or os.getenv("CVAT_BASE_URL", "http://localhost:8080")
    result = setup_cvat_workspace(
        config=config,
        host=host.rstrip("/"),
        allow_existing=args.allow_existing,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
