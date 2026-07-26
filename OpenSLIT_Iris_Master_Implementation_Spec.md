# OpenSLIT-Iris
## Master Implementation Specification for a Coding Agent

### Purpose of this document

Build a complete, reproducible, open-source research pipeline that converts routine slit-lamp photographs of the anterior eye into validated quantitative iris measurements.

The first version must focus on frontal, high-quality slit-lamp photographs. It must not make claims about systemic disease prediction. Its purpose is to create reliable infrastructure for segmentation, quality control, artefact removal, iris normalization, feature extraction, and scientific validation.

The final system should be usable by ophthalmology researchers who have a folder of slit-lamp images and limited programming experience.

---

# 1. Core objective

Implement an end-to-end pipeline:

```text
Raw slit-lamp photograph
        ↓
Dataset audit and metadata validation
        ↓
Image quality control
        ↓
Pupil, iris, and artefact segmentation
        ↓
Post-processing and usable-iris mask
        ↓
Iris normalization and regional mapping
        ↓
Interpretable feature extraction
        ↓
Optional deep feature embeddings
        ↓
Validation and robustness analysis
        ↓
CSV/JSON outputs, overlays, reports, and reusable model weights
```

The system must produce measurements only when the relevant image region is sufficiently visible and the measurement is technically valid. It must not silently return unreliable values.

Every image-level output must include:

- segmentation masks;
- visual overlays;
- image-quality scores;
- measurement-validity flags;
- feature values;
- model confidence or uncertainty;
- failure reasons when processing is rejected.

---

# 2. Project positioning

This is not a biometric identity product and not traditional iridology.

The project is a clinical and research toolkit for quantitative iris phenotyping from slit-lamp photography.

Primary scientific claim for version 1:

> Routine slit-lamp photographs can be converted into reproducible, interpretable, and quality-controlled quantitative iris measurements using an open-source computational pipeline.

Potential later use cases include:

- iris lesion monitoring;
- uveitis phenotyping;
- iris atrophy quantification;
- pigment-dispersion research;
- pseudoexfoliation research;
- congenital and genetic iris abnormalities;
- longitudinal anterior-segment studies;
- biomarker discovery;
- future mechanism-based iridomics research.

Do not implement systemic disease prediction in version 1.

---

# 3. Version 1 scope

## 3.1 Supported image type

Primary input:

- frontal or nearly frontal slit-lamp photograph;
- diffuse illumination or broad-beam illumination;
- pupil and most of the iris visible;
- conventional visible-light color image;
- one eye per image.

Images may contain:

- vertical or oblique slit-beam artefacts;
- corneal reflections;
- specular highlights;
- eyelids and eyelashes;
- variable pupil size;
- variable iris pigmentation;
- partial limbal occlusion;
- mild rotation or off-axis gaze;
- clinical abnormalities.

## 3.2 Explicit non-goals for version 1

Do not claim reliable support for:

- gonioscopy;
- retroillumination;
- narrow optical sections;
- AS-OCT;
- UBM;
- smartphone images;
- external-eye photographs from unknown cameras;
- video pupillometry;
- vascular quantification from standard color photographs;
- lesion malignancy prediction;
- systemic disease diagnosis;
- physical measurements in millimeters without a valid calibration reference.

The architecture should be modular enough to support these later.

---

# 4. Required outputs

## 4.1 Segmentation classes

Implement multi-class segmentation with these classes:

```text
0 = background/non-eye
1 = pupil
2 = iris tissue
3 = sclera/conjunctiva
4 = eyelid
5 = eyelash or dense lash occlusion
6 = corneal/specular reflection
7 = slit-beam or overexposed illumination artefact
8 = uncertain/ungradable region
```

Optional classes, enabled only when annotations exist:

```text
9 = iris lesion
10 = iris atrophy
11 = pigment spot or nevus
12 = posterior synechiae/pupillary-margin abnormality
```

The final usable iris mask must be calculated as:

```text
usable_iris =
    iris
    minus pupil
    minus reflection
    minus slit-beam artefact
    minus eyelid
    minus eyelashes
    minus uncertain/ungradable regions
```

Do not calculate texture or pigmentation features from excluded pixels.

## 4.2 Quality-control outputs

For every image, calculate and store:

- focus/sharpness score;
- exposure score;
- underexposure fraction;
- overexposure fraction;
- visible iris fraction;
- pupil visibility;
- outer limbus visibility;
- reflection burden;
- slit-beam burden;
- eyelid/eyelash occlusion burden;
- off-axis estimate;
- image resolution;
- color-channel clipping;
- overall quality category;
- accepted/rejected status;
- rejection reason.

Suggested quality categories:

```text
A = suitable for all version-1 measurements
B = suitable for geometry and coarse color/texture
C = suitable only for limited geometry
D = reject
```

Quality thresholds must be configurable and later calibrated empirically.

## 4.3 Feature outputs

### A. Geometry

Extract:

- image width and height;
- pupil centroid;
- pupil area;
- equivalent pupil diameter in pixels;
- pupil perimeter;
- pupil circularity;
- pupil eccentricity;
- pupil major and minor axes;
- pupil boundary irregularity;
- iris centroid;
- visible iris area;
- equivalent iris diameter in pixels;
- pupil-to-iris diameter ratio;
- pupil-to-iris area ratio;
- pupil-to-iris center displacement;
- superior, inferior, nasal, and temporal visible-iris fractions;
- radial sector visibility;
- left-right and superior-inferior asymmetry;
- pupillary-margin irregularity.

Do not report millimeters unless a trusted image calibration value is supplied.

### B. Color and pigmentation

Use at least:

- RGB;
- HSV;
- CIELAB.

Extract from the usable iris region:

- channel means and standard deviations;
- robust medians and interquartile ranges;
- color histograms;
- pigmentation heterogeneity;
- central versus peripheral color differences;
- sectoral color differences;
- radial pigmentation gradients;
- left-right and superior-inferior asymmetry;
- proportions of dark, intermediate, and light pixels using configurable clustering;
- color-uniformity metrics.

Color features must be marked invalid or low-confidence when:

- white balance is unknown and uncontrolled;
- the slit beam affects a large region;
- channel clipping is present;
- color calibration is absent.

Implement optional color normalization, but preserve raw values and record every transformation.

### C. Classical texture

Extract from the normalized and masked iris:

- local binary pattern histograms;
- GLCM contrast;
- GLCM correlation;
- GLCM energy;
- GLCM homogeneity;
- entropy;
- local variance;
- edge density;
- Gabor-filter responses at multiple scales and orientations;
- wavelet summary features;
- radial texture profiles;
- angular texture profiles;
- optional fractal-dimension estimates.

Calculate texture globally and by predefined radial/angular sectors.

### D. Anatomical surface features

Version 1 should implement an extensible framework for:

- collarette localization;
- collarette radius and irregularity;
- crypt candidate detection;
- crypt count, area, and spatial distribution;
- contraction-furrow candidate detection;
- furrow count, length, continuity, and circumferential burden;
- pigment-spot candidate detection;
- sectoral stromal irregularity;
- possible atrophy candidate regions.

These features must remain marked as experimental until expert validation is completed.

Do not label a structure clinically without a documented annotation definition.

### E. Deep representations

Optionally extract:

- encoder embeddings from the trained segmentation encoder;
- embeddings from a pretrained ophthalmic or iris model;
- normalized-image embeddings;
- regional embeddings.

Store embeddings separately from interpretable features.

Do not describe a deep embedding as biologically interpretable.

---

# 5. Dataset ingestion

## 5.1 Accepted dataset structure

Support a folder structure such as:

```text
dataset/
├── images/
│   ├── image_000001.jpg
│   ├── image_000002.png
│   └── ...
├── metadata.csv
└── annotations/
    ├── image_000001.png
    └── ...
```

## 5.2 Required metadata columns

At minimum:

```text
image_id
filename
patient_id
eye_id
laterality
visit_id
acquisition_date_or_period
device_id
site_id
```

Recommended:

```text
age
sex
diagnosis
dilation_status
illumination_type
magnification
camera_model
slit_lamp_model
image_resolution
image_quality_grade
annotator_id
annotation_status
repeat_image_group
```

The system must validate metadata before training.

It must detect:

- duplicate filenames;
- duplicate images using perceptual hashing;
- missing patient identifiers;
- conflicting laterality;
- multiple images from the same patient;
- repeated visits;
- corrupted files;
- unsupported image types;
- severe resolution differences;
- likely screenshots containing software interfaces;
- accidental train-test leakage.

## 5.3 Privacy

Never copy patient names or medical-record numbers into training outputs.

Create a de-identification check for:

- embedded text;
- burned-in patient identifiers;
- EXIF metadata;
- filenames containing personal identifiers.

Do not modify source data. Create a de-identified working copy.

---

# 6. Dataset audit before model development

The coding agent must run a complete audit before training.

Produce:

```text
reports/dataset_audit.html
reports/dataset_audit.csv
```

The audit must include:

- total images;
- unique patients;
- unique eyes;
- images per patient;
- images per eye;
- images per visit;
- device distribution;
- site distribution;
- image-size distribution;
- laterality distribution;
- diagnosis distribution if available;
- iris-color distribution if labelled;
- duplicate-image report;
- corrupted-image report;
- example montage;
- quality distribution;
- artefact distribution;
- missing-metadata report;
- proposed patient-level data split.

The agent must not train the final model until the audit is complete.

---

# 7. Annotation system

## 7.1 Annotation tool

Provide an annotation workflow compatible with one of:

- CVAT;
- Label Studio;
- MONAI Label;
- a custom lightweight Gradio or Streamlit application.

Prefer existing annotation software unless a custom interface provides a clear advantage.

The annotation interface should support:

- polygon masks;
- brush correction;
- separate semantic classes;
- zoom;
- opacity control;
- pretrained-model suggestions;
- correction of proposed masks;
- annotator identity;
- review status;
- disagreement resolution;
- export to indexed PNG masks or COCO format.

## 7.2 Annotation protocol

Create a written annotation manual with visual examples.

Define:

- pupil boundary;
- outer visible iris boundary;
- iris obscured by corneal reflection;
- slit-beam artefact;
- eyelash occlusion;
- eyelid occlusion;
- uncertain tissue;
- whether partially visible limbal regions are included;
- how to handle posterior synechiae;
- how to handle iris lesions;
- how to handle severe corneal opacity.

The manual must specify that annotators label only visible structures, not inferred hidden anatomy.

## 7.3 Annotation strategy

Start with a stratified pilot set covering:

- light and dark irises;
- small and large pupils;
- high and low illumination;
- clear and blurred images;
- weak and strong reflections;
- weak and strong slit beams;
- normal and abnormal iris anatomy;
- all devices and sites;
- repeated images.

Suggested initial process:

1. Select 50 representative images.
2. Generate masks using pretrained models.
3. Have an ophthalmologist correct them.
4. Review failure patterns.
5. Finalize the annotation protocol.
6. Annotate the main development set.
7. Independently double-annotate at least 10–20% of the test set.

---

# 8. Use of pretrained models

Do not train every component from random initialization.

Benchmark available pretrained systems where licenses permit:

- IrisParseNet for iris/pupil localization;
- Open-Iris components for iris segmentation, normalization, and classical encoding;
- SAM 2 for prompt-based mask generation and annotation assistance;
- MedSAM for medical-image prompt-based segmentation;
- a standard ImageNet-pretrained or ophthalmic-pretrained encoder for supervised segmentation.

The main purpose of these models is:

- baseline comparison;
- annotation acceleration;
- transfer learning;
- encoder initialization;
- pseudo-label generation with expert correction.

Do not assume that a model trained on biometric near-infrared iris images will generalize to clinical slit-lamp photographs.

Every pretrained model must be evaluated on the local pilot dataset before adoption.

Before redistributing code or weights, verify:

- repository license;
- checkpoint license;
- commercial-use restrictions;
- attribution requirements;
- derivative-work requirements.

---

# 9. Segmentation model development

## 9.1 Baselines

Implement at least three segmentation baselines:

1. U-Net with a pretrained encoder.
2. DeepLabV3+ or SegFormer.
3. A promptable SAM/MedSAM-based approach or an adapted iris-specific pretrained model.

Use a common evaluation pipeline.

## 9.2 Recommended model structure

Primary supervised model:

```text
Input image
   ↓
Pretrained encoder
   ↓
Multi-scale decoder
   ↓
Multi-class segmentation map
   ↓
Boundary-refinement/post-processing
   ↓
Per-class confidence map
```

Recommended options:

- SegFormer-B0/B1 for an efficient transformer baseline;
- U-Net or U-Net++ with ResNet/EfficientNet encoder;
- DeepLabV3+;
- SAM/MedSAM mask decoder adaptation;
- parameter-efficient fine-tuning where appropriate.

Do not choose the final architecture based only on average Dice. Include boundary accuracy, reliability, speed, and failure behavior.

## 9.3 Loss functions

Implement configurable combinations of:

- weighted cross-entropy;
- Dice loss;
- focal loss;
- Tversky loss;
- boundary loss;
- optional Hausdorff-inspired loss.

Class weights must address small artefact classes and imbalance.

## 9.4 Augmentation

Use realistic augmentation:

- mild rotation;
- translation;
- scale;
- crop;
- horizontal flip only when laterality handling remains correct;
- brightness and contrast variation;
- gamma variation;
- mild blur;
- JPEG compression;
- sensor noise;
- mild color-temperature shift;
- partial synthetic reflection;
- partial synthetic slit beam;
- eyelash-like occlusion;
- mild perspective transformation.

Do not apply augmentations that destroy clinical anatomy.

Store augmentation settings in configuration files.

## 9.5 Training behavior

Requirements:

- mixed-precision training;
- deterministic random seeds where possible;
- early stopping;
- learning-rate scheduling;
- checkpointing;
- experiment logging;
- gradient clipping if required;
- resume-from-checkpoint;
- configurable image size;
- batch-size auto-adjustment for available VRAM;
- training curves;
- per-class metrics;
- sample predictions after each epoch.

Never use the held-out test set for architecture selection, threshold selection, or early stopping.

---

# 10. Patient-level splitting

All splits must be performed by patient.

Default split:

```text
70% training
15% validation
15% internal test
```

Use stratification where possible by:

- device;
- site;
- iris pigmentation;
- diagnosis;
- image quality;
- laterality.

All images from one patient must remain in one split.

Also create:

- temporal test set when acquisition dates permit;
- device-held-out test set when multiple devices exist;
- site-held-out external test set when multiple sites exist.

Save split manifests as immutable CSV files.

---

# 11. Segmentation post-processing

Implement configurable post-processing:

- remove impossible isolated components;
- preserve the largest anatomically plausible pupil;
- enforce pupil-inside-iris constraints;
- fill small holes;
- smooth jagged mask edges without erasing real abnormalities;
- identify incomplete outer iris boundaries;
- estimate visibility by angular sector;
- reject anatomically impossible masks;
- flag pupil and iris-center inconsistencies;
- flag excessive overlap between artefact and usable iris.

Do not force all irises into perfect circles.

Use ellipse or spline representations only as optional geometric summaries.

Retain the original pixel-level segmentation mask.

---

# 12. Iris normalization

Implement Daugman-style polar/rubber-sheet normalization as an optional representation:

```text
Cartesian iris annulus
        ↓
Radial-angular normalized strip
```

Requirements:

- support non-circular pupil and iris boundaries;
- carry the occlusion and artefact mask into normalized coordinates;
- preserve a mapping back to the original image;
- provide fixed-size normalized output;
- store the rotation reference;
- support sector-based measurements.

Never fill excluded regions with plausible-looking synthetic iris texture.

Use a mask channel for excluded pixels.

---

# 13. Feature extraction design

## 13.1 Modular architecture

Every feature extractor must implement a common interface:

```python
class IrisFeatureExtractor:
    name: str
    version: str

    def validate_input(self, image, masks, metadata) -> ValidationResult:
        ...

    def extract(self, image, masks, metadata) -> dict:
        ...

    def feature_schema(self) -> dict:
        ...
```

Every returned feature must include:

- feature name;
- numeric value;
- unit;
- extraction version;
- validity status;
- confidence;
- reason for invalidity if applicable.

## 13.2 Regional analysis

Divide the iris into configurable regions:

- central/peripupillary;
- middle;
- peripheral;
- superior;
- inferior;
- nasal;
- temporal;
- 8, 12, or 16 angular sectors.

Laterality must be normalized so anatomical nasal and temporal sectors are correctly represented for both eyes.

## 13.3 Reproducibility

Feature extraction must be deterministic for the same image, model checkpoint, and configuration.

Record:

- model version;
- checkpoint hash;
- preprocessing version;
- feature-extractor version;
- configuration hash;
- software environment.

---

# 14. Validation

Validation is required at four levels.

## 14.1 Segmentation validation

Calculate per class:

- Dice coefficient;
- Intersection over Union;
- precision;
- recall;
- specificity where meaningful;
- pixel accuracy;
- average symmetric surface distance;
- 95th-percentile Hausdorff distance;
- boundary F1 score;
- pupil-center error;
- pupil-diameter error;
- iris-boundary error;
- usable-iris inclusion error;
- artefact-exclusion error.

Report:

- mean;
- standard deviation;
- median;
- interquartile range;
- bootstrap 95% confidence interval;
- worst-case examples;
- subgroup performance.

## 14.2 Human agreement

For double-annotated images, calculate:

- intergrader Dice;
- boundary agreement;
- intraclass correlation for derived measurements;
- adjudicated consensus.

Compare model performance with human-human agreement.

## 14.3 Feature validity

For manually measurable features, compare automated results against expert measurements using:

- intraclass correlation coefficient;
- Pearson or Spearman correlation where appropriate;
- mean absolute error;
- root mean squared error;
- Bland-Altman analysis;
- repeatability coefficient;
- coefficient of variation.

Examples:

- pupil diameter;
- pupil area;
- lesion area;
- crypt count;
- collarette radius;
- furrow burden.

## 14.4 Repeatability and robustness

When repeated images are available, test:

- same-eye same-session repeatability;
- same-eye different-session repeatability;
- illumination sensitivity;
- slit-beam-position sensitivity;
- pupil-size sensitivity;
- image-compression sensitivity;
- mild blur sensitivity;
- device sensitivity;
- site sensitivity.

A feature should be flagged as unstable if acquisition variation produces large changes.

---

# 15. Subgroup and bias analysis

Evaluate performance by:

- iris pigmentation;
- age group;
- sex if available;
- device;
- site;
- diagnosis;
- normal versus abnormal anatomy;
- pupil size;
- image quality;
- reflection burden;
- slit-beam burden;
- degree of visible iris;
- laterality.

Do not report only an overall average.

Investigate whether poorer segmentation in dark or light irises causes biased feature estimates.

---

# 16. Failure detection and uncertainty

The system must detect likely failures.

Implement:

- softmax or probabilistic confidence;
- test-time augmentation uncertainty;
- optional model ensemble;
- out-of-distribution score;
- anatomical plausibility checks;
- quality-based rejection;
- mask-consistency checks.

Possible statuses:

```text
SUCCESS
SUCCESS_WITH_LIMITATIONS
LOW_CONFIDENCE
UNGRADABLE
SEGMENTATION_FAILURE
UNSUPPORTED_ACQUISITION
```

Every failure must have a machine-readable reason.

Do not quietly replace a failed result with a heuristic circle.

Heuristic fallback may be shown as a separate low-confidence result but must never be presented as equivalent to model segmentation.

---

# 17. Command-line interface

Provide commands such as:

```bash
openslit audit \
  --images data/images \
  --metadata data/metadata.csv \
  --output outputs/audit

openslit annotate-assist \
  --images data/images \
  --model sam2 \
  --output outputs/pseudo_masks

openslit train \
  --config configs/segmentation_segformer.yaml

openslit evaluate \
  --checkpoint checkpoints/best.ckpt \
  --split manifests/test.csv \
  --output outputs/evaluation

openslit extract \
  --images data/images \
  --metadata data/metadata.csv \
  --checkpoint checkpoints/best.ckpt \
  --output outputs/features

openslit report \
  --results outputs/evaluation \
  --output reports/final_report.html
```

---

# 18. Graphical interface

Provide a simple Gradio or Streamlit application with:

- drag-and-drop image upload;
- image preview;
- segmentation overlay;
- class-mask toggles;
- quality-control result;
- accepted/rejected status;
- feature table;
- normalized iris strip;
- downloadable CSV/JSON;
- downloadable overlay and masks;
- model/checkpoint information;
- warning that the software is for research use.

The interface must not present diagnostic conclusions.

---

# 19. Output format

For each image, save:

```text
outputs/
├── masks/
│   ├── image_id_multiclass.png
│   ├── image_id_pupil.png
│   ├── image_id_iris.png
│   └── image_id_usable_iris.png
├── overlays/
│   └── image_id_overlay.png
├── normalized/
│   ├── image_id_normalized.png
│   └── image_id_normalized_mask.png
├── per_image_json/
│   └── image_id.json
├── features.csv
├── quality.csv
├── failures.csv
└── run_manifest.json
```

`features.csv` must contain one row per image.

`run_manifest.json` must include:

- run date;
- git commit;
- model name;
- checkpoint hash;
- configuration;
- software versions;
- GPU information;
- dataset manifest hash.

---

# 20. Repository structure

Use a structure similar to:

```text
openslit-iris/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── environment.yml
├── Dockerfile
├── Makefile
├── configs/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── features/
├── openslit/
│   ├── cli/
│   ├── data/
│   ├── audit/
│   ├── annotation/
│   ├── preprocessing/
│   ├── quality/
│   ├── segmentation/
│   ├── postprocessing/
│   ├── normalization/
│   ├── features/
│   ├── validation/
│   ├── reporting/
│   ├── visualization/
│   └── utils/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_pretrained_baselines.ipynb
│   ├── 03_training.ipynb
│   ├── 04_validation.ipynb
│   └── 05_feature_analysis.ipynb
├── docs/
│   ├── annotation_protocol.md
│   ├── feature_dictionary.md
│   ├── model_card.md
│   └── data_card_template.md
└── examples/
```

Core functionality must live in the Python package, not only in notebooks.

---

# 21. Suggested technical stack

Use:

- Python 3.10 or 3.11;
- PyTorch;
- OpenCV;
- NumPy;
- pandas;
- scikit-image;
- SciPy;
- Albumentations;
- MONAI or segmentation-model libraries where useful;
- Hugging Face Transformers where useful for SegFormer;
- scikit-learn;
- matplotlib;
- Plotly for interactive HTML reports if needed;
- Pydantic or Hydra for configuration;
- pytest;
- Gradio or Streamlit;
- TensorBoard, MLflow, or Weights & Biases for experiment tracking;
- Docker;
- Git and Git LFS.

Detect the installed CUDA and PyTorch configuration before installation.

Do not break a working CUDA environment by blindly upgrading PyTorch.

Use pinned dependencies after the environment is confirmed.

---

# 22. GPU and computational behavior

The agent should inspect the available hardware:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Then:

- select image resolution based on VRAM;
- use mixed precision;
- start with a small model;
- benchmark memory use;
- avoid unnecessary full-resolution loading;
- use gradient accumulation when required;
- record GPU model and training time.

A modern NVIDIA GPU with 8–12 GB VRAM should support pilot training with efficient models and reasonable image sizes. More VRAM permits larger input resolution and batch size.

Final inference should support both GPU and CPU, although CPU will be slower.

---

# 23. Testing requirements

Implement:

## Unit tests

- metadata validation;
- mask class validation;
- geometry calculations;
- color-space conversion;
- regional mapping;
- laterality normalization;
- normalized-strip generation;
- feature schema;
- configuration loading.

## Integration tests

- one-image end-to-end processing;
- folder-level processing;
- checkpoint loading;
- CPU inference;
- GPU inference;
- output-file generation;
- failure handling.

## Regression tests

Maintain a small, de-identified test set with expected outputs.

Fail tests if:

- mask dimensions change unexpectedly;
- feature names disappear;
- values change beyond a justified tolerance;
- output files are missing;
- model loading becomes incompatible.

---

# 24. Acceptance criteria for version 1

Version 1 is complete only when all of the following are satisfied:

1. Dataset audit runs successfully.
2. Patient-level split manifests are generated and checked.
3. Annotation protocol is documented.
4. At least three segmentation approaches are benchmarked.
5. The selected model produces pupil, iris, and artefact masks.
6. Quality-control and failure detection are implemented.
7. Segmentation is evaluated on an untouched patient-level test set.
8. Metrics are reported with confidence intervals.
9. Subgroup analysis is completed.
10. Geometry, color, and classical texture extraction are implemented.
11. Every feature has a definition, unit, and validity rule.
12. Outputs include masks, overlays, CSV, JSON, and a run manifest.
13. A command-line interface works.
14. A minimal graphical interface works.
15. CPU and GPU inference are tested.
16. Unit and integration tests pass.
17. A model card and data-card template are written.
18. The repository is installable from a clean environment.
19. The code contains no patient identifiers or embedded credentials.
20. No medical diagnostic claim is shown in the interface.

---

# 25. Development phases

## Phase 0 — Environment inspection

Deliver:

- hardware report;
- CUDA/PyTorch report;
- dataset path confirmation;
- repository initialization;
- dependency plan.

## Phase 1 — Dataset audit

Deliver:

- audit report;
- duplicate detection;
- metadata validation;
- image montage;
- initial quality analysis;
- proposed patient-level splits.

Do not start final training before approval of this phase.

## Phase 2 — Pretrained baseline evaluation

Run:

- IrisParseNet if technically compatible;
- Open-Iris components;
- SAM 2;
- MedSAM;
- one conventional pretrained U-Net/SegFormer baseline.

Deliver:

- masks for representative images;
- failure analysis;
- runtime and memory report;
- recommendation for annotation assistance and final architecture.

## Phase 3 — Annotation workflow

Deliver:

- annotation tool;
- annotation schema;
- annotation manual;
- pilot corrected masks;
- intergrader pilot analysis.

## Phase 4 — Supervised segmentation

Deliver:

- training configuration;
- checkpoints;
- training curves;
- validation metrics;
- sample predictions;
- ablation comparison;
- selected final model.

## Phase 5 — Independent segmentation validation

Deliver:

- untouched test-set metrics;
- confidence intervals;
- subgroup analysis;
- failure gallery;
- uncertainty and rejection analysis.

## Phase 6 — Feature extraction

Deliver:

- feature dictionary;
- geometry features;
- color features;
- texture features;
- regional features;
- normalized iris outputs;
- per-image feature files.

## Phase 7 — Feature validation

Deliver:

- expert comparison;
- repeatability analysis;
- Bland-Altman plots;
- ICC estimates;
- robustness analysis;
- list of validated versus experimental features.

## Phase 8 — Packaging

Deliver:

- installable package;
- CLI;
- graphical interface;
- Dockerfile;
- tests;
- documentation;
- model card;
- sample data workflow;
- release checklist.

---

# 26. Required reporting behavior for the coding agent

After every phase, produce:

```text
1. What was implemented
2. What data were used
3. What commands were run
4. What files were created
5. What metrics were obtained
6. What failed
7. Why it failed
8. What remains uncertain
9. What should happen next
```

Never invent metrics.

Never state that a model is validated without an untouched test set and expert ground truth.

Never hide failed images.

Store a failure gallery containing the worst predictions.

---

# 27. Scientific safeguards

The coding agent must enforce these rules:

- no image-level random split when patients have multiple images;
- no test-set use during development;
- no feature extraction from artefact-contaminated pixels;
- no circular enforcement that erases pathological pupil shapes;
- no synthetic completion of hidden iris tissue;
- no claims that texture predicts systemic disease;
- no claim of millimeter accuracy without calibration;
- no diagnostic output;
- no unreported exclusion of difficult images;
- no removal of poor-performing subgroups from final reporting;
- no redistribution of clinical images without explicit permission;
- no redistribution of pretrained weights without license review.

---

# 28. Final expected product

The final open-source product should allow a researcher to run:

```bash
openslit extract \
  --images /path/to/slit_lamp_images \
  --metadata /path/to/metadata.csv \
  --output /path/to/results
```

and receive:

- quality-controlled segmentation;
- pupil and usable-iris masks;
- artefact masks;
- visual overlays;
- normalized iris strips;
- reproducible geometry;
- reproducible color and texture features;
- validity and uncertainty indicators;
- failure reasons;
- CSV and JSON outputs;
- an HTML summary report.

The result should be a research platform, not merely a demonstration notebook.

---

# 29. First action for the coding agent

Perform only the following before building the full model:

1. Inspect the computer, GPU, CUDA, Python, and storage.
2. Inspect the dataset structure without altering source files.
3. Generate the dataset audit.
4. Identify missing metadata and patient-level leakage risks.
5. Select 30–50 representative images.
6. Test pretrained segmentation approaches on those images.
7. Produce a comparison report and visual failure gallery.
8. Recommend the minimum annotation set and final training plan.
9. Wait for approval before large-scale annotation or training.

Do not begin by writing a large monolithic training script.

Build the project as a modular, tested, reproducible package.
