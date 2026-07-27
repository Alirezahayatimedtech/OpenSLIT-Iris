# OpenSLIT-Iris

<p align="center">
  <img src="docs/assets/annotation_segmentation_example.svg" alt="OpenSLIT-Iris segmentation example showing pupil, iris, reflection, slit-beam, eyelid, and eyelash labels" width="100%">
</p>

OpenSLIT-Iris is a reproducible workspace for quality-controlled iris phenotyping from slit-lamp photographs. The current release provides an end-to-end collaboration pathway from blinded image selection to independent quality grading, isolated CVAT segmentation, automated disagreement analysis, senior adjudication, and versioned revision.

It does not perform diagnosis. It does not use historical center, nasal, temporal, eye-side, or laterality labels.

## Start here: integrated two-grader workflow

The recommended workflow uses:

- **Google Drive and Google Sheets** for aliased-image access and structured image-quality grading;
- **self-hosted CVAT Community** for pixel segmentation on your own computer or institutional server;
- **OpenSLIT code** for blinding, access provisioning, state gates, validation, hashes, disagreement maps, senior review, and revisions;
- **GitHub** for code and protocol versions only. Patient images, credentials, submissions, and exports remain outside Git.

```text
Data custodian
      │
      ├── selects 50 aliased images and locks the pilot
      │
      ├───────────────┬────────────────┐
      ▼               ▼                │
  Grader 01       Grader 02            │
  private Sheet   private Sheet        │
      │               │                │
  isolated CVAT   isolated CVAT        │
  project         project              │
      └───────────────┬────────────────┘
                      ▼
          automated comparison package
                      ▼
             senior ophthalmologist
                      │
       accept A / accept B / consensus /
          versioned revision request
                      ▼
              final consensus record
```

### Access design

The bootstrap command creates this Google Drive structure:

```text
OpenSLIT-Iris Pilot v1/
├── 01_Aliased_Images/
├── 02_grader_01/
├── 03_grader_02/
├── 04_Adjudication/
└── 05_Final_Consensus/
```

- Both graders and the senior receive read-only access to the aliased images.
- Each grader can edit only their own quality-grading Sheet.
- Graders cannot see each other's Sheet or CVAT project.
- The senior receives the adjudication material only after the independent submissions are frozen.
- Frozen submissions are never overwritten. Revisions create a new version.

### Install

Build the local blinded pilot first:

```bash
python -m openslit.collaboration build \
  --config configs/pilot_slit_dataset.json
```

Install the Google Drive and CVAT integrations:

```bash
python -m pip install -e '.[cvat,google]'
```

Start free self-hosted CVAT Community:

```bash
cp deployment/cvat/.env.example deployment/cvat/.env
chmod +x deployment/cvat/cvat.sh
deployment/cvat/cvat.sh up
deployment/cvat/cvat.sh create-superuser
```

Create two ordinary CVAT accounts for the graders. Do not make them administrators or staff.

Create a Google Drive parent folder, share it as Editor with the Google service-account email, then configure:

```text
configs/workflow_pilot_v1.json
```

Replace the placeholder grader emails, CVAT usernames, senior email, and Drive parent-folder ID. Set Google credentials locally:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

Never commit the credential file.

### Run the workflow

Validate everything:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json check
```

Create the Drive folders, upload aliased images, insert image links, and create the two private Google Sheets:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json bootstrap-drive
```

After each ophthalmologist finishes quality grading, freeze their Sheet:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-grading --grader grader_01

openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-grading --grader grader_02
```

Create two isolated CVAT projects containing the same predetermined double-annotation images:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json setup-cvat
```

This command is blocked until both quality submissions are frozen.

After each grader completes their CVAT task, export and freeze it:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-segmentation --grader grader_01

openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-segmentation --grader grader_02
```

Build and upload the senior disagreement package:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json build-adjudication
openslit-workflow --config configs/workflow_pilot_v1.json upload-adjudication
```

The senior package contains quality-grade disagreements, both masks, colored disagreement overlays, class-level Dice and IoU, pupil-center difference, visible-iris area difference, and an editable adjudication Sheet.

Request a targeted revision without changing either original submission:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  request-revision \
  --image-id PILOT-I017 \
  --from-grader grader_01 \
  --reason "Superior eyelash was labelled as iris" \
  --protocol-reference "Annotation Protocol v1.0, Eyelash class"

openslit-workflow --config configs/workflow_pilot_v1.json \
  create-revision-task --grader grader_01
```

The correction task contains only the disputed images and begins from the grader's latest frozen masks. The original CVAT task and v1 export remain unchanged.

Check progress at any time:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json status
```

Read the complete [end-to-end grader workflow](docs/END_TO_END_WORKFLOW.md).

## Scientific design of the first pilot

The included SLIT configuration selects 50 images from 50 unique participants:

- 40 participants and one image per participant are sampled using a recorded pseudorandom seed;
- 10 additional participants cover objective technical challenges: darkness, brightness, blur, clipping, and overexposure;
- images from exact SHA-256 duplicate groups are excluded;
- source patient and filename information remains in a private key;
- shared images receive blinded patient and image aliases;
- two graders independently grade all 50 images;
- 20 predetermined images are assigned to both graders for independent mask annotation.

The technical challenge panel is for protocol stress-testing. It must not be used to estimate the prevalence of poor-quality images in the full dataset.

## Local pilot outputs

```bash
python -m openslit.collaboration build \
  --config configs/pilot_slit_dataset.json
```

The builder creates:

```text
collaboration_runs/slit_pilot_v1/
├── private/
│   ├── pilot_private_key.csv
│   └── selected_participants.csv
├── shared/
│   ├── pilot_image_index.csv
│   ├── drive_links.csv
│   ├── grader_01_quality_grading.xlsx
│   ├── grader_02_quality_grading.xlsx
│   └── mask_task_manifest.csv
├── drive_upload/
│   └── images/PILOT-I001.jpg ...
└── run_manifest.json
```

`private/` must never be placed in the collaborators' Google Drive folder or uploaded to CVAT. The builder refuses to overwrite a non-empty pilot directory. Change `output_dir` to a new version when the frozen design changes.

## Quality-grading variables

Each ophthalmologist independently records:

- acquisition eligibility;
- A/B/C/D image-quality grade;
- focus and exposure problems;
- reflection and slit-beam burden;
- eyelid/eyelash burden;
- off-axis problem;
- pupil and outer-iris visibility;
- segmentation feasibility;
- recommended inclusion for mask annotation;
- exclusion reason;
- confidence and optional comments.

A grader's recommendation does not change the predetermined double-annotation subset. Both graders segment the same locked images so their masks remain comparable.

## Validate and compare downloaded grading workbooks

Validate one submission:

```bash
python -m openslit.collaboration validate \
  --submission completed_grader_01.xlsx \
  --index collaboration_runs/slit_pilot_v1/shared/pilot_image_index.csv
```

Merge two independent submissions:

```bash
python -m openslit.collaboration merge \
  --first completed_grader_01.xlsx \
  --second completed_grader_02.xlsx \
  --index collaboration_runs/slit_pilot_v1/shared/pilot_image_index.csv \
  --output collaboration_runs/slit_pilot_v1/agreement
```

Outputs include:

- `grades_long.csv`;
- `adjudication_queue.csv`;
- `agreement_summary.json`;
- percent agreement for core decisions;
- quadratic-weighted Cohen's kappa for A/B/C/D quality grades.

The original grades remain preserved after adjudication.

## Annotation classes

The machine-readable source of truth is:

```text
configs/annotation_schema_v1.json
```

Protocol v1 uses exactly these indexed mask values:

```text
0 = background / other ocular tissue
1 = pupil
2 = visible iris
3 = reflection
4 = slit-beam artefact
5 = eyelid occlusion
6 = eyelash occlusion
7 = uncertain / ungradable region
```

Background is implicit in CVAT, so graders draw the seven non-background labels. Hidden anatomy is never inferred beneath lids, lashes, reflections, or illumination artefacts.

Validate normalized masks:

```bash
openslit-validate-masks \
  --schema configs/annotation_schema_v1.json \
  --manifest annotations/annotation_manifest.csv \
  --images annotations/images \
  --masks annotations/masks \
  --report annotations/validation_report.csv
```

## Use with another dataset

Create a source manifest with at least:

```text
participant_id
image_path
```

Generate the objective image profile:

```bash
python -m openslit.collaboration profile \
  --manifest /path/to/source_manifest.csv \
  --output /path/to/image_profile.csv
```

The profile contains:

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

The pilot builder intentionally ignores disease, outcome, laterality, historical view, and other clinical fields during selection.

## Project documentation

- [End-to-end grader workflow](docs/END_TO_END_WORKFLOW.md)
- [Contribution and dataset onboarding guide](CONTRIBUTING.md)
- [Collaborative pilot protocol](docs/COLLABORATIVE_PILOT_PROTOCOL.md)
- [Google Drive and grader procedure](docs/GOOGLE_DRIVE_WORKFLOW.md)
- [Local CVAT deployment and API setup](deployment/cvat/README.md)
- [Annotation Protocol v1.0](docs/ANNOTATION_PROTOCOL_V1.md)
- [Annotation examples](docs/ANNOTATION_EXAMPLES.md)
- [Master implementation specification](OpenSLIT_Iris_Master_Implementation_Spec.md)

Dataset-derived reports, source paths, participant mappings, images, Google credentials, CVAT credentials, exports, workflow state, and generated pilot runs remain local and are excluded from Git.

## Current scope

Implemented:

- reproducible subject-level image selection and technical-challenge sampling;
- blind aliases and private/shared separation;
- independent grader workbooks and controlled vocabulary;
- Google Drive API provisioning and private Google Sheets;
- grader-specific permission downgrade when a submission is frozen;
- immutable versioned grading snapshots with SHA-256 hashes;
- local CVAT Community deployment;
- two isolated grader CVAT projects created through the Python SDK;
- gated CVAT access after both quality submissions freeze;
- CVAT Segmentation Mask export normalization to OpenSLIT class IDs;
- independent mask validation and immutable versioned snapshots;
- class-level Dice and IoU, disagreement overlays, and geometric differences;
- senior adjudication Sheet and structured outcomes;
- versioned revision requests without overwriting original annotations;
- pre-populated CVAT correction tasks containing only disputed images;
- final adjudication validation.

Next stages:

- accepted real-image annotation examples after adjudication;
- segmentation baselines and AI-assisted annotation;
- supervised training;
- quantitative feature extraction;
- independent test validation;
- release packaging.

The pilot must be completed and adjudicated before large-scale annotation or model training becomes scientifically valid.
