# OpenSLIT-Iris

<p align="center">
  <img src="docs/assets/annotation_segmentation_example.svg" alt="OpenSLIT-Iris segmentation example showing pupil, iris, reflection, slit-beam, eyelid, and eyelash labels" width="100%">
</p>

OpenSLIT-Iris is a reproducible workspace for quality-controlled iris
phenotyping from slit-lamp photographs. The current implementation covers the
first collaborative gate: blinded pilot selection, independent image-quality
grading, mask-task preparation, submission validation, agreement analysis, and
adjudication.

It does not perform diagnosis. It does not use historical center, nasal,
temporal, eye-side, or laterality labels.

## Interactive annotation with local CVAT

The repository now includes a free, self-hosted CVAT Community deployment and
Python SDK integration. CVAT runs on the computer or institutional server that
stores the aliased pilot images; raw clinical data do not need to be uploaded to
a commercial annotation service.

Start the local server:

```bash
cp deployment/cvat/.env.example deployment/cvat/.env
chmod +x deployment/cvat/cvat.sh
deployment/cvat/cvat.sh up
deployment/cvat/cvat.sh create-superuser
```

Install the matching CVAT SDK integration:

```bash
python -m pip install -e '.[cvat]'
```

Validate the local pilot plan:

```bash
openslit-cvat check --config configs/cvat_pilot_v1.json
```

After creating separate CVAT accounts and replacing the placeholder usernames
in `configs/cvat_pilot_v1.json`, create the project and two independent
annotation tasks:

```bash
set -a
source deployment/cvat/.env
set +a
openslit-cvat setup --config configs/cvat_pilot_v1.json
```

See [the local CVAT deployment guide](deployment/cvat/README.md). The setup uses
the machine-readable [Annotation Protocol v1.0](docs/ANNOTATION_PROTOCOL_V1.md)
and uploads only images marked for independent double annotation.

## Current pilot: start here

Each grader uses a separate Google Sheet for the image-quality grading stage:

- [Grader 01 workbook](https://docs.google.com/spreadsheets/d/1Z3Hheb3DMOPX-XePyoeF3MQP0iO3nipPWOvQVDRlKo0/edit)
- [Grader 02 workbook](https://docs.google.com/spreadsheets/d/105JYMkOkLbhONBMvmSlSLvGvOt_JYGz08_dvptPZfC4/edit)
- [Aliased image folder](https://drive.google.com/drive/folders/1BwlZhfXiEw10ga31zuBNqOQmJrO211_3)
- [Living manuscript draft](https://docs.google.com/document/d/1yfS8j6wyrARqofp_8xMleIZ_PPzjXGJLIT7EFIzHRT4/edit)

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

`private/` must never be placed in the collaborators' Google Drive folder or
uploaded to CVAT. The builder refuses to overwrite a non-empty pilot directory.
Change `output_dir` to a new version such as `slit_pilot_v2` when the frozen
design changes. This prevents accidental deletion of submitted grades.

## Google Drive quality-grading workflow

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

Google Sheets is used only for image-quality grading and task tracking.
Pixel-level masks are created in the self-hosted CVAT workspace and exported as
indexed PNG or COCO data.

## Validate quality-grading submissions

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
- [Local CVAT deployment and API setup](deployment/cvat/README.md)
- [Annotation Protocol v1.0](docs/ANNOTATION_PROTOCOL_V1.md)
- [Annotation examples](docs/ANNOTATION_EXAMPLES.md)
- [Master implementation specification](OpenSLIT_Iris_Master_Implementation_Spec.md)

Dataset-derived feasibility tables, reports, source paths, participant mappings,
images, CVAT credentials, exports, and generated pilot runs remain local and are
excluded from Git.

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
- adjudication queue;
- annotation protocol v1.0;
- machine-readable annotation schema;
- annotation mask validation;
- local CVAT Community deployment wrapper;
- CVAT SDK project and independent-task creation.

Not yet implemented:

- CVAT annotation export conversion and disagreement maps;
- accepted real-image annotation examples after adjudication;
- segmentation baselines;
- supervised training;
- feature extraction;
- independent test validation;
- release packaging.

The pilot must be completed and adjudicated before those stages become valid.
