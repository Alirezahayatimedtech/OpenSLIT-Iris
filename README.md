# OpenSLIT-Iris

<p align="center">
  <img src="docs/assets/annotation_segmentation_example.svg" alt="OpenSLIT-Iris segmentation example" width="100%">
</p>

OpenSLIT-Iris is an open-source workflow for turning slit-lamp photographs into reliable iris segmentations and quantitative iris features.

It connects:

- **Google Drive and Sheets** for controlled image access and quality grading;
- **self-hosted CVAT** for pixel-level annotation and AI-assisted correction;
- **OpenSLIT tools** for blinding, comparison, senior consensus, AI evaluation and feature extraction.

It does not perform diagnosis.

## Workflow

```text
Aliased images
      ↓
Two independent ophthalmologists
      ↓
Senior consensus masks
      ↓
AI benchmark and assisted correction
      ↓
Quality-controlled iris features
      ↓
Repeatability and clinical research
```

The two graders cannot see each other's Sheet or CVAT project. AI masks are never shown during the independent manual pilot. Original submissions are frozen and never overwritten.

## Feature extraction

<p align="center">
  <img src="docs/assets/iris_feature_extraction_overview.svg" alt="OpenSLIT-Iris feature extraction overview" width="100%">
</p>

The feature module extracts:

- **geometry:** pupil and iris area, diameter, circularity, eccentricity, centers and occlusion fractions;
- **color:** raw and normalized RGB, HSV and CIELAB statistics, radial gradients and sector asymmetry;
- **texture:** LBP, GLCM and Haar-wavelet measurements from a normalized iris strip;
- **quality:** blur, exposure, illumination, visible coverage, artefact burden and source-mask provenance.

Color and texture features are produced only when the final mask passes the feature-quality gate. Geometry and quality fields remain available for audit.

## Quick start

Install the main integrations:

```bash
python -m pip install -e '.[cvat,google]'
```

Start local CVAT:

```bash
cp deployment/cvat/.env.example deployment/cvat/.env
chmod +x deployment/cvat/cvat.sh
deployment/cvat/cvat.sh up
deployment/cvat/cvat.sh create-superuser
```

Edit `configs/workflow_pilot_v1.json` with the Google accounts, CVAT usernames and Drive parent-folder ID.

Create the Drive workspace:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json check
openslit-workflow --config configs/workflow_pilot_v1.json bootstrap-drive
```

After quality grading and segmentation are complete:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json freeze-grading --grader grader_01
openslit-workflow --config configs/workflow_pilot_v1.json freeze-grading --grader grader_02
openslit-workflow --config configs/workflow_pilot_v1.json setup-cvat

openslit-workflow --config configs/workflow_pilot_v1.json freeze-segmentation --grader grader_01
openslit-workflow --config configs/workflow_pilot_v1.json freeze-segmentation --grader grader_02
openslit-workflow --config configs/workflow_pilot_v1.json build-adjudication
openslit-workflow --config configs/workflow_pilot_v1.json upload-adjudication
```

## AI stage

The AI stage starts only after senior consensus.

```bash
python -m pip install -e '.[ai]'
openslit-ai --config configs/ai_workflow_v1.json check --configuration-only
```

It supports U-Net, SegFormer, an external nnU-Net reference, uncertainty maps, human–AI comparison, senior-approved CVAT pre-annotations, crossover evaluation and balanced active learning.

## Extract iris features

Validate the configuration:

```bash
openslit-features --config configs/feature_extraction_v1.json \
  check --configuration-only
```

After final consensus masks exist:

```bash
openslit-features --config configs/feature_extraction_v1.json check
openslit-features --config configs/feature_extraction_v1.json \
  extract --run-id pilot_features_v1
```

Each run produces:

```text
iris_features.csv
iris_features.xlsx
feature_quality.csv
feature_dictionary.csv
feature_report.html
previews/
```

Upload the derived results to the existing controlled Drive workspace:

```bash
openslit-features --config configs/feature_extraction_v1.json \
  upload-drive --run-id pilot_features_v1
```

This creates `06_Feature_Extraction/<run_id>/`. Source images and masks are not duplicated.

## Repeatability

Use repeated images of the same eye before testing clinical associations:

```bash
openslit-features repeatability \
  --features /path/to/iris_features.csv \
  --group-column repeat_group_id \
  --output-dir /path/to/repeatability_results
```

The module reports ICC(2,1), within-group coefficient of variation, repeatability coefficient and Bland–Altman agreement.

## Annotation classes

```text
0  background / other ocular tissue
1  pupil
2  visible iris
3  reflection
4  slit-beam artefact
5  eyelid occlusion
6  eyelash occlusion
7  uncertain / ungradable region
```

The source of truth is `configs/annotation_schema_v1.json`.

## Data protection

Never commit patient identifiers, images, credentials, submissions, masks, checkpoints, probability maps, feature outputs or workflow state. Aliased eye photographs remain sensitive research data.

## Documentation

- [End-to-end grader workflow](docs/END_TO_END_WORKFLOW.md)
- [AI-assisted segmentation and active learning](docs/AI_ASSISTED_SEGMENTATION.md)
- [Quantitative iris feature extraction](docs/FEATURE_EXTRACTION.md)
- [Google Drive procedure](docs/GOOGLE_DRIVE_WORKFLOW.md)
- [Local CVAT deployment](deployment/cvat/README.md)
- [Local AI runtime](deployment/ai/README.md)
- [Annotation Protocol v1.0](docs/ANNOTATION_PROTOCOL_V1.md)

## Current stage

The annotation, adjudication, AI and feature-extraction infrastructure is implemented. Real feature analysis remains locked until the two-grader pilot is completed and final masks are approved.
