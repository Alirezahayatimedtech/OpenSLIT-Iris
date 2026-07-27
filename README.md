# OpenSLIT-Iris

<p align="center">
  <img src="docs/assets/annotation_segmentation_example.svg" alt="OpenSLIT-Iris segmentation example showing pupil, iris, reflection, slit-beam, eyelid, and eyelash labels" width="100%">
</p>

OpenSLIT-Iris is an open-source workflow for building reliable iris-segmentation datasets from slit-lamp photographs.

It connects:

- **Google Drive and Google Sheets** for image access and quality grading;
- **self-hosted CVAT** for pixel-level annotation;
- **OpenSLIT tools** for blinding, validation, disagreement analysis and senior adjudication.

It does not perform diagnosis.

## How the collaboration works

```text
                         Data custodian
                              │
                  selects and aliases images
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
         Grader 01                         Grader 02
     private Google Sheet              private Google Sheet
             │                                 │
     private CVAT project               private CVAT project
             └────────────────┬────────────────┘
                              ▼
                    automated comparison
                              ▼
                    senior ophthalmologist
                              ▼
                  final consensus dataset
```

The two graders work independently from beginning to end. They cannot see each other's Sheet or CVAT project.

The senior ophthalmologist reviews only the disagreements after both submissions are frozen. Original annotations are never overwritten.

## What each person does

### Grader 01 and Grader 02

1. Open the assigned private Google Sheet.
2. Review every aliased slit-lamp image.
3. Record image quality, visibility, artefacts and segmentation feasibility.
4. Submit the Sheet for freezing.
5. Open the assigned private CVAT project.
6. Segment the locked double-annotation images.
7. Submit the CVAT task.

### Senior ophthalmologist

The senior receives a disagreement package containing:

- both quality assessments;
- both segmentation masks;
- disagreement overlays;
- class-level Dice and IoU;
- pupil-centre and visible-iris area differences;
- structured options to accept one annotation, create consensus or request revision.

## Quick start for the project administrator

### 1. Build the blinded pilot

```bash
python -m openslit.collaboration build \
  --config configs/pilot_slit_dataset.json
```

The current pilot contains:

- 50 images from 50 participants for independent quality grading;
- 20 predetermined images for independent double segmentation.

### 2. Install the integrations

```bash
python -m pip install -e '.[cvat,google]'
```

### 3. Start local CVAT

```bash
cp deployment/cvat/.env.example deployment/cvat/.env
chmod +x deployment/cvat/cvat.sh
deployment/cvat/cvat.sh up
deployment/cvat/cvat.sh create-superuser
```

CVAT will be available at:

```text
http://localhost:8080
```

Create one ordinary CVAT account for each grader.

### 4. Configure the workflow

Edit:

```text
configs/workflow_pilot_v1.json
```

Replace the placeholder values for:

- grader Google accounts;
- grader CVAT usernames;
- senior ophthalmologist Google account;
- Google Drive parent-folder ID.

Set the Google service-account credential locally:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

Never commit credentials or clinical images to GitHub.

### 5. Create the Drive workspace

```bash
openslit-workflow --config configs/workflow_pilot_v1.json check
openslit-workflow --config configs/workflow_pilot_v1.json bootstrap-drive
```

This creates:

```text
OpenSLIT-Iris Pilot v1/
├── 01_Aliased_Images/
├── 02_grader_01/
├── 03_grader_02/
├── 04_Adjudication/
└── 05_Final_Consensus/
```

### 6. Freeze grading and open CVAT

After both graders complete their Sheets:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-grading --grader grader_01

openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-grading --grader grader_02

openslit-workflow --config configs/workflow_pilot_v1.json setup-cvat
```

The CVAT projects are created only after both grading submissions are frozen.

### 7. Freeze segmentation and build senior review

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-segmentation --grader grader_01

openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-segmentation --grader grader_02

openslit-workflow --config configs/workflow_pilot_v1.json build-adjudication
openslit-workflow --config configs/workflow_pilot_v1.json upload-adjudication
```

Check progress at any time:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json status
```

## Annotation classes

Protocol v1 uses:

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

Background is implicit in CVAT. Annotators draw the seven non-background classes and label only visible structures.

The machine-readable source of truth is:

```text
configs/annotation_schema_v1.json
```

## Data protection

Keep these outside GitHub:

- patient identifiers and private source keys;
- slit-lamp images;
- Google credentials;
- CVAT credentials;
- grader submissions;
- exported masks;
- workflow state and adjudication files.

Aliased eye photographs remain sensitive research data and still require approved storage and access control.

## Documentation

- [End-to-end grader workflow](docs/END_TO_END_WORKFLOW.md)
- [Collaborative pilot protocol](docs/COLLABORATIVE_PILOT_PROTOCOL.md)
- [Google Drive and grader procedure](docs/GOOGLE_DRIVE_WORKFLOW.md)
- [Local CVAT deployment](deployment/cvat/README.md)
- [Annotation Protocol v1.0](docs/ANNOTATION_PROTOCOL_V1.md)
- [Annotation examples](docs/ANNOTATION_EXAMPLES.md)
- [Dataset onboarding guide](CONTRIBUTING.md)

## Current stage

Implemented:

- reproducible image selection and blinding;
- private Google Sheets for two independent graders;
- self-hosted CVAT integration;
- frozen and versioned submissions;
- mask validation and disagreement analysis;
- senior adjudication and targeted revision tasks.

Next:

- complete and adjudicate the pilot;
- add accepted real-image examples;
- develop segmentation baselines;
- introduce AI-assisted annotation;
- validate quantitative iris features.
