# AI-assisted segmentation and active-learning protocol

Protocol status: infrastructure ready; model use remains locked until the independent human pilot and senior consensus are complete.

## Purpose

This layer adds AI without contaminating the independent human benchmark. It separates three scientific questions:

1. **Human reproducibility:** how closely do two ophthalmologists agree when neither sees AI output?
2. **Independent AI performance:** how closely does a frozen model match the senior consensus on an untouched patient-level test set?
3. **AI-assisted productivity:** does correcting an AI mask reduce annotation time without reducing final quality or creating automation bias?

These questions must not be merged into one experiment.

## Ordered workflow

```text
Manual grading and segmentation by Grader 01 and Grader 02
                         ↓
                Senior consensus masks
                         ↓
            Patient-level train/validation/test split
                         ↓
       Train U-Net and SegFormer on training participants
                         ↓
     Select models on validation participants only
                         ↓
      One-time evaluation on untouched test participants
                         ↓
  Senior approval or rejection for AI-assisted annotation
                         ↓
        Separate CVAT correction tasks with AI masks
                         ↓
      Time, corrections, uncertainty and quality recorded
                         ↓
 Balanced active-learning batches from the unlabelled pool
```

The test set remains locked. It is never used for active learning, model selection, threshold tuning, or retraining.

## Recommended model set

### Primary conventional baseline: U-Net with ResNet-34 encoder

Configured as `unet_resnet34`. It provides an interpretable, widely used convolutional baseline and is implemented through `segmentation_models_pytorch`.

### Primary transformer baseline: SegFormer-B0

Configured as `segformer_b0`. It provides a lightweight transformer comparison with a multiscale encoder and semantic-segmentation decoder.

### Self-configuring reference: nnU-Net 2D

Configured as `nnunet_2d` but disabled by default. It is run as an external reference pipeline because nnU-Net controls its own preprocessing, training and inference conventions. OpenSLIT generates the patient-level split manifest and evaluates the resulting masks with the same benchmark code.

### Optional interactive helper: SAM 2

Configured as `sam2_interactive` but disabled by default. It may be evaluated later for click- or box-guided boundary refinement. It is not treated as the primary multiclass model because the OpenSLIT protocol requires seven explicit foreground classes and one class per pixel.

Foundation or interactive models must not replace the conventional baselines.

## Data requirements

The AI stage begins only when a frozen consensus manifest exists with at least:

```text
image_id
image_file
mask_file
blinded_patient_id
```

The patient alias is required to prevent images from the same participant appearing in more than one split.

The default split is:

- 70% of participants for training;
- 15% for validation;
- 15% for the untouched test set.

For a very small pilot, these splits are infrastructure tests rather than definitive model-performance estimates. Large-scale training starts only after the annotation protocol is frozen and the dataset is expanded.

## Training

Install optional AI dependencies:

```bash
python -m pip install -e '.[ai]'
```

Validate the configuration:

```bash
openslit-ai --config configs/ai_workflow_v1.json \
  check --configuration-only
```

After consensus files exist:

```bash
openslit-ai --config configs/ai_workflow_v1.json prepare-splits
```

Train the two primary baselines:

```bash
openslit-ai --config configs/ai_workflow_v1.json train \
  --model unet_resnet34 \
  --split-manifest collaboration_runs/slit_pilot_v1/ai/split_manifest.csv \
  --consensus-masks /path/to/consensus/masks

openslit-ai --config configs/ai_workflow_v1.json train \
  --model segformer_b0 \
  --split-manifest collaboration_runs/slit_pilot_v1/ai/split_manifest.csv \
  --consensus-masks /path/to/consensus/masks
```

The training command loads only the training and validation splits. It saves the checkpoint with the lowest validation loss and records that the test set was not used.

The initial loss is cross-entropy plus soft Dice. Later experiments may compare focal, Tversky, boundary or class-weighted losses, but each change must be versioned as a new experiment rather than silently replacing the baseline.

## Independent inference

```bash
openslit-ai --config configs/ai_workflow_v1.json infer \
  --model unet_resnet34 \
  --checkpoint /path/to/best_checkpoint.pt \
  --split-manifest collaboration_runs/slit_pilot_v1/ai/split_manifest.csv \
  --split test
```

Each prediction run exports:

- indexed PNG segmentation mask;
- per-class probability array;
- normalized predictive-entropy map;
- mean confidence;
- mean entropy;
- high-uncertainty pixel fraction;
- run summary and prediction manifest.

## Benchmark design

The senior consensus is the primary reference. The same comparison command is run for:

- AI model vs senior consensus;
- Grader 01 vs senior consensus;
- Grader 02 vs senior consensus;
- optionally Grader 01 vs Grader 02 through the existing adjudication reports.

For every protocol class, report:

- Dice;
- IoU;
- precision;
- recall;
- false-positive and false-negative pixels;
- centroid difference;
- area relative error.

For pupil and iris, geometric errors must be interpreted alongside overlap metrics. A high whole-mask Dice score can still hide a clinically relevant local boundary error.

Results must also be stratified by:

- quality grade;
- focus and exposure problems;
- reflection burden;
- slit-beam burden;
- eyelid/eyelash burden;
- difficult versus suitable segmentation rating;
- acquisition subgroup when multiple cameras or sites are later added.

The final paper should report uncertainty intervals using participant-level bootstrap resampling. The code configuration reserves `bootstrap_iterations`, while formal confidence-interval reporting should be activated only when the sample is large enough to support it.

## Senior model approval

A model cannot create AI-assisted tasks until the senior ophthalmologist records approval after reviewing:

- the independent test summary;
- class-specific failures;
- uncertainty maps;
- a failure gallery;
- cases where both humans disagree with the AI;
- cases where the AI agrees with only one grader;
- image-quality and artefact subgroups;
- unacceptable boundary errors despite apparently good Dice.

Record approval:

```bash
openslit-ai --config configs/ai_workflow_v1.json approve-model \
  --model unet_resnet34 \
  --benchmark-summary /path/to/test/summary.json \
  --approved-by senior_ophthalmologist \
  --notes "Approved for correction tasks; not autonomous use"
```

Approval means only that the model may pre-populate a correction task. It does not make the AI output ground truth and does not permit autonomous release.

## AI-assisted CVAT mode

AI-assisted work happens in new CVAT projects. Manual pilot tasks and original masks remain unchanged.

The grader sees:

- the original image;
- the AI mask as an editable starting point;
- protocol classes and colors;
- optional uncertainty information outside the mask layer;
- a correction category;
- a route to senior review.

Recommended correction categories:

```text
ACCEPTED_WITHOUT_CHANGE
MINOR_CORRECTION
MAJOR_CORRECTION
REJECTED_AND_REDRAWN
UNGRADABLE
SEND_TO_SENIOR
```

The workflow should record active annotation time and correction category. CVAT idle time must be excluded when possible.

Create a pre-populated task after approval:

```bash
openslit-ai --config configs/ai_workflow_v1.json create-assisted-task \
  --grader grader_01 \
  --model unet_resnet34 \
  --batch-id batch_001 \
  --batch-manifest /path/to/batch_001.csv \
  --prediction-manifest /path/to/prediction_manifest.csv \
  --prediction-masks /path/to/prediction/masks
```

The command creates an isolated project/task and initiates import of a CVAT Segmentation Mask archive. Check the CVAT Requests page until import finishes before annotation begins.

## Human-factors comparison

Use a randomized crossover study:

- one arm starts from a blank mask;
- one arm starts from AI pre-annotation;
- each ophthalmologist performs both modes on different but comparable images;
- assignment order is randomized;
- the senior consensus remains the reference.

Primary outcomes:

- active annotation time per image;
- final Dice and boundary/geometric error against consensus;
- major-error rate;
- senior-referral rate.

Secondary outcomes:

- number of corrected classes;
- correction category;
- acceptance-without-change frequency;
- uncertainty calibration;
- annotator workload and trust calibration;
- automation-bias events where an incorrect AI boundary is retained.

AI assistance is useful only when it reduces work without reducing final quality.

## Active learning

Active learning begins after the first model is independently evaluated. It never samples from the test set.

The default batch combines:

- 40% highest uncertainty;
- 25% greatest disagreement between U-Net and SegFormer;
- 20% diversity selection using image embeddings when available;
- 15% random controls.

A random component is mandatory. Selecting only difficult or uncertain images would distort the training distribution and make later performance estimates harder to interpret.

Candidate table columns:

```text
image_id
split
labelled
uncertainty_score
model_disagreement_score
```

Optional columns such as quality grade, artefact burden, camera and site should be retained in the selected batch for auditing and stratification.

```bash
openslit-ai --config configs/ai_workflow_v1.json select-active-batch \
  --candidates /path/to/candidate_scores.csv \
  --embeddings /path/to/image_embeddings.csv
```

The senior or data custodian reviews the selected batch before tasks are created. Active learning recommends images; it does not bypass clinical oversight.

## Retraining cycle

Each cycle is versioned:

```text
model_v1 → assisted batch 001 → corrected consensus → model_v2
model_v2 → assisted batch 002 → corrected consensus → model_v3
```

Store for every version:

- training and validation participants;
- untouched test participants;
- code commit;
- configuration file;
- checkpoint hash;
- prediction manifest;
- benchmark summary;
- senior approval decision;
- active-learning selection manifest;
- CVAT task IDs;
- correction-time and correction-category results.

Do not continuously retrain a production model without preserving these versions.

## Gate before iris-feature extraction

Iris features may be calculated only from masks that pass the final segmentation-quality rules. Feature extraction must exclude or explicitly flag:

- ungradable images;
- masks with failed anatomical plausibility checks;
- excessive uncertain or artefact area;
- out-of-distribution images;
- AI masks not corrected or approved under the defined workflow.

This keeps segmentation uncertainty from silently becoming biological signal.
