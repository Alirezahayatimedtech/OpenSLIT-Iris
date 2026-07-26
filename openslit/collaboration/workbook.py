"""Create and read locked-reference collaborative grading workbooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from .schema import (
    ALLOWED_VALUES,
    FIELD_VALUE_DEFINITIONS,
    MASK_CLASS_DEFINITIONS,
    QUALITY_GRADE_DEFINITIONS,
    REFERENCE_COLUMNS,
    RESPONSE_COLUMNS,
)


def write_grader_workbook(
    shared: pd.DataFrame, destination: Path, grader_id: str
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "quality_grading"
    columns = REFERENCE_COLUMNS + RESPONSE_COLUMNS
    sheet.append(columns)

    for _, source in shared.iterrows():
        row = {column: "" for column in columns}
        for column in REFERENCE_COLUMNS:
            row[column] = source.get(column, "")
        row["grader_id"] = grader_id
        sheet.append([row[column] for column in columns])

    header_fill = PatternFill("solid", fgColor="1F4E78")
    reference_fill = PatternFill("solid", fgColor="D9EAF7")
    response_fill = PatternFill("solid", fgColor="FFF2CC")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        cell.protection = Protection(locked=True)

    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            column = columns[cell.column - 1]
            if column in REFERENCE_COLUMNS or column == "grader_id":
                cell.fill = reference_fill
                cell.protection = Protection(locked=True)
            else:
                cell.fill = response_fill
                cell.protection = Protection(locked=False)
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    for column, allowed in ALLOWED_VALUES.items():
        column_index = columns.index(column) + 1
        values = ",".join(allowed)
        validation = DataValidation(
            type="list",
            formula1=f'"{values}"',
            allow_blank=True,
            error="Select one value from the controlled list.",
            errorTitle="Invalid value",
        )
        sheet.add_data_validation(validation)
        validation.add(
            f"{get_column_letter(column_index)}2:"
            f"{get_column_letter(column_index)}{len(shared) + 1}"
        )

    drive_column = columns.index("drive_url") + 1
    for row_number in range(2, len(shared) + 2):
        cell = sheet.cell(row=row_number, column=drive_column)
        if cell.value:
            cell.hyperlink = str(cell.value)
            cell.style = "Hyperlink"

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(shared) + 1}"
    widths = {
        "blinded_image_id": 18,
        "blinded_patient_id": 20,
        "image_file": 24,
        "drive_url": 38,
        "comments": 45,
        "primary_exclusion_reason": 25,
    }
    for index, column in enumerate(columns, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = widths.get(
            column, max(15, min(24, len(column) + 2))
        )
    sheet.protection.sheet = True
    sheet.protection.password = "openslit-reference"
    sheet.protection.autoFilter = False
    sheet.protection.sort = False

    instructions = workbook.create_sheet("instructions")
    instructions_rows = [
        ("Purpose", "Independent blinded quality and eligibility grading."),
        (
            "Independence",
            "Do not view another grader's responses before submitting this workbook.",
        ),
        (
            "Anatomy",
            "Grade only visible structures. Never infer hidden tissue.",
        ),
        (
            "Prohibited labels",
            "Do not assign or infer left/right, laterality, center, nasal, or temporal.",
        ),
        (
            "Required",
            "Complete every yellow response cell except comments, which are optional.",
        ),
        (
            "Quality A",
            QUALITY_GRADE_DEFINITIONS["A"],
        ),
        ("Quality B", QUALITY_GRADE_DEFINITIONS["B"]),
        ("Quality C", QUALITY_GRADE_DEFINITIONS["C"]),
        ("Quality D", QUALITY_GRADE_DEFINITIONS["D"]),
        (
            "Mask work",
            "Draw masks in CVAT or equivalent software, not inside this spreadsheet.",
        ),
    ]
    for row in instructions_rows:
        instructions.append(row)
    instructions.column_dimensions["A"].width = 24
    instructions.column_dimensions["B"].width = 105
    for cell in instructions[1]:
        cell.font = Font(bold=True)
    for row in instructions.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    codebook = workbook.create_sheet("codebook")
    codebook.append(["field", "allowed_value", "definition"])
    for grade, definition in QUALITY_GRADE_DEFINITIONS.items():
        codebook.append(["quality_grade", grade, definition])
    for field, values in ALLOWED_VALUES.items():
        if field == "quality_grade":
            continue
        for value in values:
            codebook.append(
                [field, value, FIELD_VALUE_DEFINITIONS.get((field, value), "")]
            )
    for class_id, definition in MASK_CLASS_DEFINITIONS.items():
        codebook.append(["mask_class", str(class_id), definition])
    codebook.column_dimensions["A"].width = 32
    codebook.column_dimensions["B"].width = 28
    codebook.column_dimensions["C"].width = 85
    for cell in codebook[1]:
        cell.font = Font(bold=True)
    for row in codebook.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    destination.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(destination)


def read_grading_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    workbook = load_workbook(path, data_only=True, read_only=True)
    if "quality_grading" not in workbook.sheetnames:
        raise ValueError(f"{path} lacks a quality_grading sheet")
    sheet = workbook["quality_grading"]
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"{path} contains no grading rows")
    headers = [str(value) if value is not None else "" for value in rows[0]]
    data = [
        ["" if value is None else str(value).strip() for value in row]
        for row in rows[1:]
    ]
    return pd.DataFrame(data, columns=headers)


def apply_drive_links(workbook_path: Path, links: pd.DataFrame) -> None:
    required = {"blinded_image_id", "drive_url"}
    if not required.issubset(links.columns):
        raise ValueError(f"Drive-link table must contain {sorted(required)}")
    link_map = dict(zip(links["blinded_image_id"], links["drive_url"]))
    workbook = load_workbook(workbook_path)
    sheet = workbook["quality_grading"]
    headers = [cell.value for cell in sheet[1]]
    id_column = headers.index("blinded_image_id") + 1
    url_column = headers.index("drive_url") + 1
    for row_number in range(2, sheet.max_row + 1):
        image_id = sheet.cell(row=row_number, column=id_column).value
        url = str(link_map.get(image_id, "") or "").strip()
        cell = sheet.cell(row=row_number, column=url_column)
        cell.value = url
        cell.hyperlink = url or None
        if url:
            cell.style = "Hyperlink"
    workbook.save(workbook_path)


def apply_drive_links_to_csv(table_path: Path, links: pd.DataFrame) -> None:
    required = {"blinded_image_id", "drive_url"}
    if not required.issubset(links.columns):
        raise ValueError(f"Drive-link table must contain {sorted(required)}")
    table = pd.read_csv(table_path, dtype=str, keep_default_na=False)
    if "blinded_image_id" not in table.columns or "drive_url" not in table.columns:
        raise ValueError(f"{table_path} lacks blinded_image_id or drive_url")
    link_map = dict(zip(links["blinded_image_id"], links["drive_url"]))
    missing = set(table["blinded_image_id"]) - set(link_map)
    if missing:
        raise ValueError(f"Drive links missing image IDs: {sorted(missing)}")
    table["drive_url"] = table["blinded_image_id"].map(link_map).fillna("")
    table.to_csv(table_path, index=False)
