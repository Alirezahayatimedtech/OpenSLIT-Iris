"""Lazy model registry for reproducible OpenSLIT segmentation baselines."""

from __future__ import annotations

from typing import Any

from .config import ModelSpec


SUPPORTED_FAMILIES = {
    "segmentation_models_pytorch",
    "huggingface_segformer",
    "nnunet_external",
    "interactive_helper",
}


def describe_model(spec: ModelSpec) -> dict[str, Any]:
    if spec.family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported model family: {spec.family}")
    trainable_here = spec.family in {
        "segmentation_models_pytorch",
        "huggingface_segformer",
    }
    return {
        "model_id": spec.model_id,
        "family": spec.family,
        "role": spec.role,
        "enabled": spec.enabled,
        "trainable_in_openslit": trainable_here,
        "parameters": spec.parameters,
    }


def build_model(spec: ModelSpec, num_classes: int) -> Any:
    """Build a supported PyTorch semantic-segmentation model lazily.

    nnU-Net is intentionally treated as an external reference pipeline. Interactive
    helpers such as SAM 2 are not trained as protocol-class classifiers here.
    """

    if spec.family == "segmentation_models_pytorch":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install AI dependencies with: python -m pip install -e '.[ai]'"
            ) from exc
        architecture = str(spec.parameters.get("architecture", "Unet"))
        constructor = getattr(smp, architecture, None)
        if constructor is None:
            raise ValueError(f"Unknown segmentation_models_pytorch architecture: {architecture}")
        return constructor(
            encoder_name=str(spec.parameters.get("encoder_name", "resnet34")),
            encoder_weights=spec.parameters.get("encoder_weights", "imagenet"),
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    if spec.family == "huggingface_segformer":
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install AI dependencies with: python -m pip install -e '.[ai]'"
            ) from exc
        checkpoint = str(spec.parameters.get("checkpoint", "nvidia/mit-b0"))
        return SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    if spec.family == "nnunet_external":
        raise RuntimeError(
            "nnU-Net is an external reference pipeline. Export the split manifest "
            "with openslit-ai prepare-splits and follow docs/AI_ASSISTED_SEGMENTATION.md."
        )

    if spec.family == "interactive_helper":
        raise RuntimeError(
            "Interactive helpers are optional contour-refinement tools, not trained "
            "as the primary OpenSLIT multi-class benchmark."
        )

    raise ValueError(f"Unsupported model family: {spec.family}")


def extract_logits(model_output: Any) -> Any:
    """Normalize outputs from SMP and Hugging Face segmentation models."""

    return getattr(model_output, "logits", model_output)
