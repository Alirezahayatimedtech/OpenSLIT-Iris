"""Command-line interface for OpenSLIT AI benchmarking and assistance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openslit.workflow.config import load_workflow_config
from openslit.workflow.state import WorkflowState

from .active_learning import select_active_learning_batch
from .benchmark import build_comparison_matrix, compare_source_to_consensus
from .config import load_ai_workflow_config
from .consensus import materialize_consensus_dataset
from .cvat_assist import approve_model_for_assistance, create_ai_assisted_task
from .infer import run_inference
from .productivity import create_crossover_plan, summarize_crossover_results
from .splits import create_grouped_splits, verify_split_manifest
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openslit-ai",
        description=(
            "Run the OpenSLIT AI stages: senior consensus, patient-level splits, "
            "baseline training, held-out benchmarking, CVAT pre-annotation, and active learning."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ai_workflow_v1.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Validate AI configuration")
    check.add_argument("--configuration-only", action="store_true")
    commands.add_parser("status", help="Show the AI section of the workflow state")

    consensus = commands.add_parser(
        "materialize-consensus",
        help="Create training-ready masks from final senior adjudication",
    )
    consensus.add_argument("--queue", type=Path, required=True)

    splits = commands.add_parser("prepare-splits", help="Create patient-level splits")
    splits.add_argument("--output", type=Path, default=None)

    verify = commands.add_parser("verify-splits", help="Check split leakage")
    verify.add_argument("--manifest", type=Path, required=True)

    train = commands.add_parser("train", help="Train a configured baseline model")
    train.add_argument("--model", required=True)
    train.add_argument("--split-manifest", type=Path, required=True)
    train.add_argument("--consensus-masks", type=Path, required=True)
    train.add_argument("--output", type=Path, default=None)

    infer = commands.add_parser("infer", help="Run frozen-model inference")
    infer.add_argument("--model", required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--split-manifest", type=Path, required=True)
    infer.add_argument("--split", default="test")
    infer.add_argument("--output", type=Path, default=None)

    benchmark = commands.add_parser(
        "benchmark", help="Compare one AI or human source with senior consensus"
    )
    benchmark.add_argument("--source", required=True)
    benchmark.add_argument("--source-manifest", type=Path, required=True)
    benchmark.add_argument("--source-masks", type=Path, required=True)
    benchmark.add_argument("--consensus-masks", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, default=None)

    matrix = commands.add_parser(
        "comparison-matrix", help="Combine multiple benchmark summaries"
    )
    matrix.add_argument("--summary", type=Path, action="append", required=True)
    matrix.add_argument("--output", type=Path, required=True)

    approve = commands.add_parser(
        "approve-model", help="Record senior approval for AI-assisted CVAT"
    )
    approve.add_argument("--model", required=True)
    approve.add_argument("--benchmark-summary", type=Path, required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--notes", default="")

    assisted = commands.add_parser(
        "create-assisted-task", help="Create a CVAT task pre-populated with AI masks"
    )
    assisted.add_argument("--grader", required=True)
    assisted.add_argument("--model", required=True)
    assisted.add_argument("--batch-id", required=True)
    assisted.add_argument("--batch-manifest", type=Path, required=True)
    assisted.add_argument("--prediction-manifest", type=Path, required=True)
    assisted.add_argument("--prediction-masks", type=Path, required=True)
    assisted.add_argument("--allow-existing", action="store_true")

    active = commands.add_parser(
        "select-active-batch", help="Select a balanced active-learning batch"
    )
    active.add_argument("--candidates", type=Path, required=True)
    active.add_argument("--embeddings", type=Path, default=None)
    active.add_argument("--output", type=Path, default=None)

    crossover = commands.add_parser(
        "create-crossover-plan",
        help="Randomize blank versus AI-assisted annotation across both graders",
    )
    crossover.add_argument("--batch-manifest", type=Path, required=True)
    crossover.add_argument("--output", type=Path, default=None)

    summarize = commands.add_parser(
        "summarize-crossover",
        help="Summarize annotation time, corrections, and senior referrals",
    )
    summarize.add_argument("--completed-plan", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_state(config):
    workflow_config = load_workflow_config(config.workflow_config_path)
    return workflow_config, WorkflowState.load_or_create(workflow_config)


def main() -> None:
    args = build_parser().parse_args()
    config = load_ai_workflow_config(args.config)
    if args.command == "check":
        result = config.validate(require_runtime_files=not args.configuration_only)
    elif args.command == "status":
        _, state = _load_state(config)
        result = state.data.get("ai", {"status": "LOCKED_UNTIL_CONSENSUS"})
    elif args.command == "materialize-consensus":
        workflow_config, state = _load_state(config)
        result = materialize_consensus_dataset(
            workflow_config,
            config,
            state,
            args.queue,
        )
    elif args.command == "prepare-splits":
        result = create_grouped_splits(config, args.output)
    elif args.command == "verify-splits":
        result = verify_split_manifest(args.manifest, config.split.group_column)
    elif args.command == "train":
        result = train_model(
            config,
            args.model,
            args.split_manifest,
            args.consensus_masks,
            args.output,
        )
    elif args.command == "infer":
        result = run_inference(
            config,
            args.model,
            args.checkpoint,
            args.split_manifest,
            split=args.split,
            output_dir=args.output,
        )
    elif args.command == "benchmark":
        result = compare_source_to_consensus(
            config,
            source_name=args.source,
            source_manifest_path=args.source_manifest,
            source_masks_dir=args.source_masks,
            consensus_masks_dir=args.consensus_masks,
            output_dir=args.output,
        )
    elif args.command == "comparison-matrix":
        result = build_comparison_matrix(args.summary, args.output)
    elif args.command == "approve-model":
        _, state = _load_state(config)
        result = approve_model_for_assistance(
            state,
            args.model,
            args.benchmark_summary,
            args.approved_by,
            args.notes,
        )
    elif args.command == "create-assisted-task":
        workflow_config, state = _load_state(config)
        result = create_ai_assisted_task(
            workflow_config,
            config,
            state,
            grader_id=args.grader,
            model_id=args.model,
            batch_manifest_path=args.batch_manifest,
            prediction_manifest_path=args.prediction_manifest,
            prediction_masks_dir=args.prediction_masks,
            batch_id=args.batch_id,
            allow_existing=args.allow_existing,
        )
    elif args.command == "select-active-batch":
        result = select_active_learning_batch(
            config,
            args.candidates,
            output_path=args.output,
            embeddings_path=args.embeddings,
        )
    elif args.command == "create-crossover-plan":
        workflow_config = load_workflow_config(config.workflow_config_path)
        result = create_crossover_plan(
            config,
            args.batch_manifest,
            tuple(grader.grader_id for grader in workflow_config.graders),
            args.output,
        )
    elif args.command == "summarize-crossover":
        result = summarize_crossover_results(
            args.completed_plan,
            args.output_dir,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
