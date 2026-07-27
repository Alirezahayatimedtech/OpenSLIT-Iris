# OpenSLIT-Iris Collaborative Pilot Protocol

Protocol version: 1.1  
Selection seed: 20260726

## Objective

Establish whether routine frontal slit-lamp photographs can be graded and segmented consistently enough to support a larger OpenSLIT-Iris development dataset.

This pilot measures annotation feasibility and inter-ophthalmologist reproducibility. It does not validate diagnosis, clinical associations, repeatability, or quantitative iris features.

## Roles

- **Data custodian:** builds the blinded pilot, retains the private source key, provisions Google Drive and CVAT, freezes submissions, and runs automated comparisons.
- **Grader 01:** independently grades all 50 images and segments the locked double-annotation subset.
- **Grader 02:** performs the same work independently.
- **Senior ophthalmologist:** reviews disagreements only after the independent source submissions are frozen, then accepts one result, creates consensus, or requests a versioned revision.

The senior must not replace or erase either original independent submission.

## Sampling design

The pilot contains 50 unique participants and one selected image per participant.

### Distribution panel

Forty participants are sampled without replacement using NumPy's seeded random number generator. One image is sampled from each selected participant using the same generator.

This provides broad subject-level representation. It is not a formal population-prevalence sample because the source dataset itself may be selected.

### Technical challenge panel

Ten additional participants are selected from objective image statistics:

- two dark images;
- two bright images;
- two low-sharpness images;
- two highly clipped images;
- two highly overexposed images.

No participant may occur in both panels. Exact SHA-256 duplicate groups are excluded before sampling.

The technical challenge panel deliberately overrepresents difficult images. Its results must be reported separately from the distribution panel.

## Blinding

Shared images use aliases such as `PILOT-I001.jpg`. Shared patient groups use aliases such as `PILOT-P001`.

The following information is prohibited from grader material, Google Drive, and CVAT:

- original patient identifier;
- original path or filename;
- eye identifier;
- left/right laterality;
- center, nasal, or temporal view;
- historical view label;
- angle or closure label;
- diagnosis or clinical outcome.

Only the data custodian retains the private key connecting aliases to source files.

## Independent end-to-end grading

Both graders review the same 50 aliased images in separate private Google Sheets. Neither grader may view the other grader's Sheet, comments, completion status, CVAT project, masks, or exported files before both independent submissions are frozen.

Required judgments:

- acquisition eligibility;
- A/B/C/D quality grade;
- pupil visibility;
- outer iris visibility;
- segmentation feasibility;
- recommended inclusion for mask annotation;
- primary exclusion reason;
- confidence.

Optional structured detail:

- focus problem;
- exposure problem;
- reflection burden;
- slit-beam burden;
- eyelid/eyelash burden;
- off-axis problem;
- comments.

The graders may recommend whether an image should enter mask annotation, but those recommendations do not alter the locked independent double-annotation subset. Both graders must segment the same predetermined 20 images for valid mask comparison.

## Operational burden scale

For reflection, slit beam, and eyelid/eyelash burden, estimate the affected fraction of the visible iris:

- **none:** no visible burden over pupil or iris;
- **mild:** less than approximately 10%, with main boundaries still reliable;
- **moderate:** approximately 10–30%, or one relevant boundary segment is uncertain;
- **severe:** more than approximately 30%, or the burden prevents reliable delineation;
- **uncertain:** another category cannot be assigned confidently.

Percentages are visual categories, not pixel measurements. Automated burdens are calculated only after validated masks exist.

Focus, exposure, and off-axis problems use the same consequence-based logic:

- **none:** no relevant effect;
- **mild:** visible issue but boundaries remain reliable;
- **moderate:** part of a relevant boundary or region is uncertain;
- **severe:** reliable delineation or measurement is prevented;
- **uncertain:** another category cannot be assigned confidently.

## Provisional quality definitions

### Grade A

The pupil and most visible iris boundaries can be delineated reliably. The image is suitable for pixel geometry and exploratory color/texture measurements. Artefacts are absent or small enough to mask without removing a material fraction of the iris.

### Grade B

The image supports reliable geometry and coarse exploratory color/texture. Moderate artefact or incomplete peripheral iris visibility prevents complete version-1 measurements.

### Grade C

Only limited geometry is reliable. Major blur, occlusion, reflection, illumination artefact, or incomplete visibility prevents reliable texture or color analysis.

### Grade D

Reject. Pupil or iris tissue cannot be delineated with sufficient reliability, or the image is outside the intended acquisition scope.

These definitions remain provisional until the two graders and senior adjudicator review actual pilot disagreements and approve real visual examples.

## Quality-submission freeze

After a grader confirms completion, the custodian exports the Google Sheet as XLSX, validates all required responses, computes SHA-256 hashes, saves a versioned snapshot, and downgrades the grader's Sheet permission from writer to reader.

The adjudication folder remains hidden from the senior until both quality submissions are frozen.

## Mask annotation

Masks are created in two isolated self-hosted CVAT projects. Each grader receives only their own project and task.

The machine-readable source of truth is `configs/annotation_schema_v1.json`. Protocol v1 uses:

```text
0 = background / other ocular tissue
1 = pupil
2 = visible iris
3 = corneal / specular reflection
4 = slit-beam artefact
5 = eyelid occlusion
6 = eyelash occlusion
7 = uncertain / ungradable region
```

Background is implicit in CVAT. Annotators draw the seven non-background classes. Sclera and conjunctiva remain part of background in protocol v1.

Annotators label only visible structures. Hidden anatomy must never be inferred or synthetically completed beneath lids, lashes, reflections, slit-beam artefacts, or indeterminate boundaries.

Twenty images are predetermined for independent double mask annotation. The remaining mask-eligible pilot images can later receive one annotation plus independent expert review, but they are not used to estimate inter-annotator mask agreement in the first gate.

## Segmentation-submission freeze

After a grader completes CVAT:

1. export the task in CVAT Segmentation Mask format;
2. convert label colors or indices to OpenSLIT class IDs;
3. validate mask dimensions and allowed classes;
4. save a versioned indexed-PNG snapshot;
5. calculate SHA-256 hashes;
6. mark the local workflow state as frozen.

Do not edit a frozen task or overwrite its export. A requested correction becomes a new CVAT task and a new version.

## Agreement and adjudication

The software reports quality-grade agreement and, for masks:

- Dice coefficient for every class;
- Intersection over Union for every class;
- total disagreement fraction;
- pupil-center difference;
- visible-iris area difference;
- colored disagreement overlays.

The senior ophthalmologist selects one outcome for each disagreement:

```text
ACCEPT_A
ACCEPT_B
CREATE_CONSENSUS
REVISION_REQUESTED_FROM_A
REVISION_REQUESTED_FROM_B
REVISION_REQUESTED_FROM_BOTH
UNGRADABLE
PROTOCOL_CLARIFICATION_REQUIRED
```

Adjudication starts only after both independent quality submissions and both independent mask submissions are frozen.

## Versioned revision

A revision request must contain:

- image alias;
- requested grader or both graders;
- anatomical or artefact region;
- reason;
- protocol reference;
- revision version.

The software creates a correction task containing only disputed images and pre-populates it with the grader's latest frozen masks. The grader edits that copy. Original masks remain immutable.

## Gate to the next phase

Proceed to large-scale annotation only after:

1. both quality submissions pass validation and freeze;
2. both mask submissions pass validation and freeze;
3. every core disagreement is adjudicated or resolved through a versioned revision;
4. quality definitions have accepted visual examples;
5. each mask class has an operational boundary definition;
6. export, normalization, and class validation succeed;
7. failure patterns and uncertain cases are documented;
8. the research team freezes the next protocol version.

There is no universal kappa or Dice threshold that automatically proves adequacy. The team must examine class-specific disagreements and whether those differences materially alter derived measurements.
