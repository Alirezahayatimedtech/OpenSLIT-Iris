"""Controlled vocabulary for collaborative pilot grading."""

from __future__ import annotations

REFERENCE_COLUMNS = [
    "blinded_image_id",
    "blinded_patient_id",
    "image_file",
    "drive_url",
]

RESPONSE_COLUMNS = [
    "grader_id",
    "review_date_yyyy_mm_dd",
    "acquisition_eligible",
    "quality_grade",
    "focus_problem",
    "exposure_problem",
    "reflection_burden",
    "slit_beam_burden",
    "eyelid_eyelash_burden",
    "off_axis_problem",
    "pupil_visibility",
    "outer_iris_visibility",
    "segmentation_feasibility",
    "include_for_mask_annotation",
    "primary_exclusion_reason",
    "confidence",
    "comments",
]

ALLOWED_VALUES = {
    "acquisition_eligible": ["yes", "no", "uncertain"],
    "quality_grade": ["A", "B", "C", "D"],
    "focus_problem": ["none", "mild", "moderate", "severe", "uncertain"],
    "exposure_problem": ["none", "mild", "moderate", "severe", "uncertain"],
    "reflection_burden": ["none", "mild", "moderate", "severe", "uncertain"],
    "slit_beam_burden": ["none", "mild", "moderate", "severe", "uncertain"],
    "eyelid_eyelash_burden": ["none", "mild", "moderate", "severe", "uncertain"],
    "off_axis_problem": ["none", "mild", "moderate", "severe", "uncertain"],
    "pupil_visibility": ["complete", "partial", "not_visible", "uncertain"],
    "outer_iris_visibility": ["complete", "partial", "not_visible", "uncertain"],
    "segmentation_feasibility": ["suitable", "difficult", "ungradable"],
    "include_for_mask_annotation": ["yes", "no", "adjudicate"],
    "primary_exclusion_reason": [
        "none",
        "not_frontal_enough",
        "iris_not_visible",
        "pupil_not_visible",
        "blur",
        "underexposure",
        "overexposure",
        "reflection",
        "slit_beam",
        "eyelid_eyelash",
        "corneal_opacity",
        "other",
    ],
    "confidence": ["high", "medium", "low"],
}

REQUIRED_RESPONSE_COLUMNS = [
    "grader_id",
    "review_date_yyyy_mm_dd",
    "acquisition_eligible",
    "quality_grade",
    "focus_problem",
    "exposure_problem",
    "reflection_burden",
    "slit_beam_burden",
    "eyelid_eyelash_burden",
    "off_axis_problem",
    "pupil_visibility",
    "outer_iris_visibility",
    "segmentation_feasibility",
    "include_for_mask_annotation",
    "primary_exclusion_reason",
    "confidence",
]

QUALITY_GRADE_DEFINITIONS = {
    "A": "Suitable for geometry and exploratory color/texture measurements.",
    "B": "Suitable for geometry and only coarse exploratory color/texture.",
    "C": "Suitable only for limited geometry.",
    "D": "Reject: insufficient visible or reliable iris information.",
}

FIELD_VALUE_DEFINITIONS = {
    ("acquisition_eligible", "yes"): (
        "Intended slit-lamp anterior-eye photograph with enough visible pupil "
        "and iris to attempt version-1 grading."
    ),
    ("acquisition_eligible", "no"): (
        "Outside the intended acquisition scope or insufficient visible pupil/"
        "iris for version-1 grading."
    ),
    ("acquisition_eligible", "uncertain"): (
        "Eligibility cannot be decided confidently from the visible image."
    ),
    ("focus_problem", "none"): "No visible focus problem affecting iris detail.",
    ("focus_problem", "mild"): (
        "Slight softness; pupil and visible iris boundaries remain reliable."
    ),
    ("focus_problem", "moderate"): (
        "Blur makes part of a relevant boundary or coarse iris detail uncertain."
    ),
    ("focus_problem", "severe"): (
        "Blur prevents reliable delineation of pupil or visible iris."
    ),
    ("exposure_problem", "none"): (
        "Exposure preserves relevant visible pupil and iris information."
    ),
    ("exposure_problem", "mild"): (
        "Small dark or saturated regions without material boundary loss."
    ),
    ("exposure_problem", "moderate"): (
        "Exposure removes part of a relevant boundary or iris region."
    ),
    ("exposure_problem", "severe"): (
        "Exposure prevents reliable delineation or measurement."
    ),
    ("off_axis_problem", "none"): (
        "View is sufficiently frontal for visible pupil/iris delineation."
    ),
    ("off_axis_problem", "mild"): (
        "Small perspective distortion without material boundary uncertainty."
    ),
    ("off_axis_problem", "moderate"): (
        "Perspective distortion makes part of the visible boundary uncertain."
    ),
    ("off_axis_problem", "severe"): (
        "View is too oblique for reliable version-1 measurements."
    ),
    ("pupil_visibility", "complete"): "Entire visible pupil boundary is identifiable.",
    ("pupil_visibility", "partial"): (
        "Pupil is identifiable but part of its boundary is obscured or uncertain."
    ),
    ("pupil_visibility", "not_visible"): (
        "Pupil or its boundary cannot be identified reliably."
    ),
    ("outer_iris_visibility", "complete"): (
        "The full visible outer iris/limbal boundary is identifiable."
    ),
    ("outer_iris_visibility", "partial"): (
        "Outer iris is identifiable but part is outside the frame, obscured, or uncertain."
    ),
    ("outer_iris_visibility", "not_visible"): (
        "Outer visible iris boundary cannot be identified reliably."
    ),
    ("segmentation_feasibility", "suitable"): (
        "Required visible classes can be delineated with routine correction."
    ),
    ("segmentation_feasibility", "difficult"): (
        "Delineation is possible but needs substantial expert correction or uncertainty labels."
    ),
    ("segmentation_feasibility", "ungradable"): (
        "Required pupil/iris structures cannot be delineated reliably."
    ),
    ("include_for_mask_annotation", "yes"): (
        "Include in the next mask-annotation task."
    ),
    ("include_for_mask_annotation", "no"): (
        "Exclude from routine mask annotation; retain the quality grade."
    ),
    ("include_for_mask_annotation", "adjudicate"): (
        "Do not decide until independent grades are adjudicated."
    ),
    ("confidence", "high"): "Little realistic chance the core decision would change.",
    ("confidence", "medium"): "Some ambiguity, but one decision is more defensible.",
    ("confidence", "low"): "Substantial ambiguity; adjudication is required.",
}

for burden_field in [
    "reflection_burden",
    "slit_beam_burden",
    "eyelid_eyelash_burden",
]:
    FIELD_VALUE_DEFINITIONS[(burden_field, "none")] = (
        "No visible burden over the pupil or iris."
    )
    FIELD_VALUE_DEFINITIONS[(burden_field, "mild")] = (
        "Approximately less than 10% of visible iris affected; main boundaries remain reliable."
    )
    FIELD_VALUE_DEFINITIONS[(burden_field, "moderate")] = (
        "Approximately 10–30% affected or one relevant boundary segment is uncertain."
    )
    FIELD_VALUE_DEFINITIONS[(burden_field, "severe")] = (
        "Approximately more than 30% affected or reliable delineation is prevented."
    )

for field in [
    "focus_problem",
    "exposure_problem",
    "reflection_burden",
    "slit_beam_burden",
    "eyelid_eyelash_burden",
    "off_axis_problem",
    "pupil_visibility",
    "outer_iris_visibility",
]:
    FIELD_VALUE_DEFINITIONS[(field, "uncertain")] = (
        "The grader cannot assign another category confidently."
    )

MASK_CLASS_DEFINITIONS = {
    0: "background/non-eye",
    1: "pupil",
    2: "visible iris tissue",
    3: "sclera/conjunctiva",
    4: "eyelid",
    5: "eyelash or dense lash occlusion",
    6: "corneal/specular reflection",
    7: "slit-beam or overexposed illumination artefact",
    8: "uncertain/ungradable region",
}

FORBIDDEN_SHARED_COLUMNS = {
    "participant_id",
    "patient_id",
    "eye_code",
    "eye_id",
    "laterality",
    "view_label",
    "view_source",
    "combo_key",
    "center",
    "nasal",
    "temporal",
    "angle_grade",
    "closure_label",
    "image_path",
    "source_path",
    "original_filename",
}
