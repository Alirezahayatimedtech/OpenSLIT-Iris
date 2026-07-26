# OpenSLIT-Iris

OpenSLIT-Iris is a reproducible workspace for quality-controlled iris
phenotyping from slit-lamp photographs. The current implementation covers the
first collaborative gate: blinded pilot selection, independent image-quality
grading, mask-task preparation, submission validation, agreement analysis, and
adjudication.

It does not perform diagnosis. It does not use historical center, nasal,
temporal, eye-side, or laterality labels.

## Current pilot: start here

Each grader uses a separate Google Sheet:

- [Grader 01 workbook](https://docs.google.com/spreadsheets/d/1Z3Hheb3DMOPX-XePyoeF3MQP0iO3nipPWOvQVDRlKo0/edit)
- [Grader 02 workbook](https://docs.google.com/spreadsheets/d/105JYMkOkLbhONBMvmSlSLvGvOt_JYGz08_dvptPZfC4/edit)
- [Aliased image folder](https://drive.google.com/drive/folders/1BwlZhfXiEw10ga31zuBNqOQmJrO211_3)

For a grader:

1. Open the assigned sheet and read `START HERE`.
2. Open `Review Images`.
3. Click `OPEN IMAGE`.
4. Complete the visible yellow dropdown cells.
5. Repeat until `Remaining` is 0.

Do not infer or record left/right, laterality, center, nasal, or temporal.
Graders must not view each other's sheets before both submissions are frozen.

## Scientific design of the first pilot

The included SLIT configuration selects 50 images from 50 unique participants:

- 40 participants and one image per participant are sampled using a recorded
  pseudorandom seed.
- 10 additional participants cover objective technical challenges: darkness,
  brightness, blur, clipping, and overexposure.
- Images from exact SHA-256 duplicate groups are excluded.
- Source patient and filename information remains in a private key.
- Shared images receive blinded patient and image aliases.
- Two graders independently grade all 50 images.
- Twenty predetermined images are marked for independent double mask
  annotation.

The technical challenge panel is for protocol stress-testing. It must not be
used to estimate the prevalence of poor-quality images in the full dataset.

## Build the local SLIT pilot

From this directory:

```bash
python3 -m openslit.collaboration build \
  --config configs/pilot_slit_dataset.json
```

Generated material is written under
`collaboration_runs/slit_pilot_v1/`:

```text
private/
  pilot_private_key.csv
  selected_participants.csv
shared/
  pilot_image_index.csv
  drive_links.csv
  grader_01_quality_grading.xlsx
  grader_02_quality_grading.xlsx
  mask_task_manifest.csv
drive_upload/
  images/PILOT-I001.jpg ...
run_manifest.json
```

`private/` must never be placed in the collaborators' Google Drive folder.
The builder refuses to overwrite a non-empty pilot directory. Change
`output_dir` to a new version such as `slit_pilot_v2` when the frozen design
changes. This prevents accidental deletion of submitted grades.

## Google Drive workflow

1. Build the pilot.
2. Upload only `drive_upload/images/` to a restricted Google Drive folder.
   The current generated pilot also includes `drive_upload_images.zip` for
   transfer; extract it before grading so each aliased image remains directly
   accessible.
3. Upload one grader workbook to each grader's separate private folder.
4. Do not allow graders to see each other's completed workbook.
5. Optionally paste image URLs into `shared/drive_links.csv`.
6. Apply the URLs to both workbooks:

```bash
python3 -m openslit.collaboration apply-links \
  --links collaboration_runs/slit_pilot_v1/shared/drive_links.csv \
  --workbook collaboration_runs/slit_pilot_v1/shared/grader_01_quality_grading.xlsx \
  --workbook collaboration_runs/slit_pilot_v1/shared/grader_02_quality_grading.xlsx \
  --index collaboration_runs/slit_pilot_v1/shared/pilot_image_index.csv
```

7. Import each XLSX file into Google Sheets or edit it directly. The generated
   workbook opens on `START HERE`; the review tab exposes only the core fields.
8. Download completed files as XLSX.
9. Validate each submission before analysis.

Google Sheets is appropriate for quality grading and task tracking. Pixel masks
must be drawn in CVAT or another image annotation system and exported as
indexed PNG or COCO data.

## Validate submissions

```bash
python3 -m openslit.collaboration validate \
  --submission completed_grader_01.xlsx \
  --index collaboration_runs/slit_pilot_v1/shared/pilot_image_index.csv
```

The validator checks:

- exact pilot image coverage;
- unchanged blinded reference fields;
- controlled vocabulary;
- required responses;
- ISO review dates;
- one consistent grader identity;
- absence of patient, laterality, view, and clinical outcome fields.

## Merge independent grades

```bash
python3 -m openslit.collaboration merge \
  --first completed_grader_01.xlsx \
  --second completed_grader_02.xlsx \
  --index collaboration_runs/slit_pilot_v1/shared/pilot_image_index.csv \
  --output collaboration_runs/slit_pilot_v1/agreement
```

Outputs:

- `grades_long.csv`;
- `adjudication_queue.csv`;
- `agreement_summary.json`;
- percent agreement for core decisions;
- quadratic-weighted Cohen's kappa for A/B/C/D quality grades.

Disagreements remain unresolved until an adjudicator fills the adjudication
fields.

## Use with another dataset

Create an image profile with these columns:

```text
participant_id
image_path
sha256
readable
brightness_mean
overexposed_fraction
channel_clip_fraction
laplacian_variance
```

Each row represents one image. `participant_id` must be a trusted subject-level
identifier. Configure paths and sample sizes using
[`configs/pilot_template.json`](configs/pilot_template.json).

Start from [`templates/source_manifest_template.csv`](templates/source_manifest_template.csv)
and generate the profile:

```bash
python3 -m openslit.collaboration profile \
  --manifest /path/to/source_manifest.csv \
  --output /path/to/image_profile.csv
```

Datasets using different column names can specify `--participant-column` and
`--image-column`.

The pilot builder intentionally ignores every other column. This prevents
historical view, laterality, disease label, or outcome fields from affecting
selection.

## Project documentation

- [Contribution and dataset onboarding guide](CONTRIBUTING.md)
- [Collaborative pilot protocol](docs/COLLABORATIVE_PILOT_PROTOCOL.md)
- [Google Drive and grader procedure](docs/GOOGLE_DRIVE_WORKFLOW.md)
- [Master implementation specification](OpenSLIT_Iris_Master_Implementation_Spec.md)

Dataset-derived feasibility tables, reports, source paths, participant mappings,
images, and generated pilot runs remain local and are excluded from Git.

## Current scope

Implemented:

- reproducible patient and image selection;
- blind aliases;
- private/shared separation;
- independent grader workbooks;
- controlled grading vocabulary;
- Google Drive links;
- mask task manifest;
- validation;
- intergrader agreement;
- adjudication queue.

Not yet implemented:

- CVAT deployment and project import;
- final annotation manual with accepted visual examples;
- segmentation baselines;
- supervised training;
- feature extraction;
- independent test validation;
- release packaging.

The pilot must be completed and adjudicated before those stages become valid.
