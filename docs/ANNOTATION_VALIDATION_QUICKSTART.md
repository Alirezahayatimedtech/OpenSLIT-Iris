# Annotation validation quickstart

Install the package from the repository root:

```bash
python -m pip install -e .
```

Prepare this structure:

```text
annotations/
├── images/
├── masks/
└── annotation_manifest.csv
```

Start the manifest from:

```text
templates/annotation_manifest_template.csv
```

Validate the submission:

```bash
openslit-validate-masks \
  --schema configs/annotation_schema_v1.json \
  --manifest annotations/annotation_manifest.csv \
  --images annotations/images \
  --masks annotations/masks \
  --report annotations/validation_report.csv
```

The command exits with status 1 when errors are detected. Warnings do not cause failure. Review every warning before freezing a submission.

The validator currently checks:

- required and forbidden manifest columns;
- duplicate image identifiers;
- image and mask existence;
- readable image and mask files;
- exact image-mask dimensions;
- single-channel indexed masks;
- permitted class IDs;
- required pupil and iris classes for gradable images;
- protocol-version consistency;
- annotator identity;
- basic area plausibility warnings.

This validator does not replace ophthalmologist review or adjudication.
