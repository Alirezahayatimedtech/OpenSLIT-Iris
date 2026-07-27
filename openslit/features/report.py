"""Human-readable previews and an HTML summary for feature extraction runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from openslit.annotation.schema import AnnotationSchema

from .normalization import PolarIris, radial_profile


def _resize_keep(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - copy.width) // 2
    y = (height - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    schema: AnnotationSchema,
) -> Image.Image:
    base = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").convert("RGBA")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for item in schema.classes:
        if item.name == "background":
            continue
        selected = mask == item.id
        rgba[selected, :3] = item.color_rgb
        rgba[selected, 3] = 105
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert(
        "RGB"
    )


def _profile_panel(polar: PolarIris, width: int, height: int) -> Image.Image:
    gray = (
        0.2126 * polar.image[..., 0]
        + 0.7152 * polar.image[..., 1]
        + 0.0722 * polar.image[..., 2]
    )
    profile = radial_profile(gray, polar.valid)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline="black")
    finite = np.isfinite(profile)
    if finite.sum() < 2:
        draw.text((10, 10), "Insufficient polar coverage", fill="black")
        return canvas
    minimum = float(np.nanmin(profile))
    maximum = float(np.nanmax(profile))
    scale = maximum - minimum or 1.0
    points: list[tuple[int, int]] = []
    for index, value in enumerate(profile):
        if not np.isfinite(value):
            continue
        x = int(round(index / max(1, len(profile) - 1) * (width - 20))) + 10
        y = height - 10 - int(round((value - minimum) / scale * (height - 20)))
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill="black", width=2)
    draw.text((8, 4), "Radial intensity profile", fill="black")
    return canvas


def create_feature_preview(
    image_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    schema: AnnotationSchema,
    polar: PolarIris | None,
    output_path: Path,
    flags: tuple[str, ...] = (),
) -> Path:
    panel_width = 360
    panel_height = 260
    title_height = 54
    panels: list[tuple[str, Image.Image]] = [
        ("Original", _resize_keep(Image.fromarray(image), panel_width, panel_height)),
        (
            "Segmentation overlay",
            _resize_keep(_overlay(image, mask, schema), panel_width, panel_height),
        ),
    ]
    if polar is not None:
        polar_image = Image.fromarray(polar.image, mode="RGB").resize(
            (panel_width, panel_height),
            Image.Resampling.NEAREST,
        )
        panels.append(("Normalized iris strip", polar_image))
        panels.append(("Radial profile", _profile_panel(polar, panel_width, panel_height)))
    else:
        blank = Image.new("RGB", (panel_width, panel_height), "white")
        ImageDraw.Draw(blank).text((10, 10), "Normalization unavailable", fill="black")
        panels.extend(
            [("Normalized iris strip", blank), ("Radial profile", blank.copy())]
        )

    canvas = Image.new(
        "RGB",
        (panel_width * 2, title_height + panel_height * 2 + 44),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"OpenSLIT feature preview: {image_id}", fill="black")
    flag_text = "PASS" if not flags else " | ".join(flags)
    draw.text((12, 30), f"Feature gate: {flag_text}", fill="black")
    for index, (title, panel) in enumerate(panels):
        column = index % 2
        row = index // 2
        x = column * panel_width
        y = title_height + row * (panel_height + 22)
        canvas.paste(panel, (x, y + 20))
        draw.text((x + 8, y + 2), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return output_path


def write_html_report(
    summary: dict[str, Any],
    features: pd.DataFrame,
    quality: pd.DataFrame,
    previews: list[Path],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    passed = int(quality["feature_gate_passed"].sum()) if len(quality) else 0
    rows = []
    for item in summary.get("top_quality_flags", []):
        rows.append(
            f"<tr><td>{html.escape(str(item['flag']))}</td><td>{int(item['count'])}</td></tr>"
        )
    preview_html = "\n".join(
        f'<figure><img src="{html.escape(path.as_posix())}" alt="Feature preview"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
        for path in previews
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OpenSLIT-Iris feature extraction report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; color: #111827; }}
.cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; min-width: 160px; }}
table {{ border-collapse: collapse; margin-top: 12px; }} th, td {{ border: 1px solid #d1d5db; padding: 6px 10px; }}
.preview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
figure {{ margin: 0; }} img {{ width: 100%; border: 1px solid #d1d5db; }}
code {{ background: #f3f4f6; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>OpenSLIT-Iris feature extraction</h1>
<p>Feature version <code>{html.escape(str(summary.get('feature_version', '')))}</code>. Features are derived only from versioned segmentation masks and remain subject to the quality gate.</p>
<div class="cards">
<div class="card"><strong>Images</strong><br>{len(features)}</div>
<div class="card"><strong>Passed gate</strong><br>{passed}</div>
<div class="card"><strong>Flagged</strong><br>{max(0, len(quality) - passed)}</div>
<div class="card"><strong>Feature columns</strong><br>{len(features.columns)}</div>
</div>
<h2>Most frequent quality flags</h2>
<table><thead><tr><th>Flag</th><th>Count</th></tr></thead><tbody>{''.join(rows) or '<tr><td>None</td><td>0</td></tr>'}</tbody></table>
<h2>Preview gallery</h2>
<div class="preview-grid">{preview_html or '<p>No previews were generated.</p>'}</div>
<h2>Run manifest</h2>
<pre>{html.escape(json.dumps(summary, indent=2))}</pre>
</body></html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
