# OpenSLIT-Iris Collaborative Pilot Protocol

Protocol version: 1.0
Selection seed: 20260726

## Objective

Establish whether routine slit-lamp photographs can be graded and segmented
consistently enough to support a larger OpenSLIT-Iris development dataset.

This phase measures annotation feasibility and intergrader reproducibility. It
does not validate diagnostic performance, clinical associations, repeatability,
or iris features.

## Sampling design

The pilot contains 50 unique participants and one selected image per
participant.

### Distribution panel

Forty participants are sampled without replacement using NumPy's seeded random
number generator. One image is sampled from each selected participant using the
same generator.

This panel provides broad subject-level representation. It is not a formal
population-prevalence sample because the source dataset itself may be selected.

### Technical challenge panel

Ten additional participants are selected from objective image statistics:

- two dark images;
- two bright images;
- two low-sharpness images;
- two highly clipped images;
- two highly overexposed images.

No participant may occur in both panels. Exact SHA-256 duplicate groups are
excluded before sampling.

The challenge panel deliberately overrepresents difficult images. Results from
this panel must be reported separately from distribution-panel results.

## Blinding

Shared images use identifiers such as `PILOT-I001.jpg`. Shared patient groups
use identifiers such as `PILOT-P001`.

The following information is prohibited from shared grader material:

- original patient identifier;
- original source path or filename;
- eye identifier;
- left/right laterality;
- center, nasal, or temporal view;
- historical view label;
- angle grade;
- closure label;
- clinical outcome.

The private key connects blinded identifiers to sources. Only the data
custodian retains it.

## Independent grading

Two graders independently review all 50 images. Each grader receives a separate
workbook and must not see the other grader's responses before submission.

Required judgments:

- acquisition eligibility;
- A/B/C/D quality grade;
- focus problem;
- exposure problem;
- reflection burden;
- slit-beam burden;
- eyelid/eyelash burden;
- off-axis problem;
- pupil visibility;
- outer iris visibility;
- segmentation feasibility;
- inclusion for mask annotation;
- primary exclusion reason;
- confidence.

Comments are optional. All other response fields are required.

## Operational burden scale

For reflection, slit beam, and eyelid/eyelash burden, estimate the affected
fraction of the visible iris:

- **none:** no visible burden over pupil or iris;
- **mild:** less than approximately 10%, with main boundaries still reliable;
- **moderate:** approximately 10–30%, or one relevant boundary segment is
  uncertain;
- **severe:** more than approximately 30%, or the burden prevents reliable
  delineation;
- **uncertain:** another category cannot be assigned confidently.

Percentages are visual categories, not pixel measurements. They are intended
to standardize the pilot review. Automated pixel burdens will be calculated
only after validated masks exist.

Focus, exposure, and off-axis problems use the same consequence-based logic:

- **none:** no relevant effect;
- **mild:** visible issue but boundaries remain reliable;
- **moderate:** part of a relevant boundary or region is uncertain;
- **severe:** reliable delineation or measurement is prevented;
- **uncertain:** another category cannot be assigned confidently.

## Provisional quality definitions

### Grade A

The pupil and most visible iris boundaries can be delineated reliably. The
image is suitable for pixel geometry and exploratory color/texture
measurements. Artefacts are absent or small enough to mask without removing a
material fraction of the iris.

### Grade B

The image supports reliable geometry and coarse exploratory color/texture.
Moderate artefact or incomplete peripheral iris visibility prevents complete
version-1 measurements.

### Grade C

Only limited geometry is reliable. Major blur, occlusion, reflection,
illumination artefact, or incomplete visibility prevents reliable texture or
color analysis.

### Grade D

Reject. Pupil or iris tissue cannot be delineated with sufficient reliability,
or the image is outside the intended acquisition scope.

These definitions remain provisional until the two graders and adjudicator
review actual pilot disagreements and approve visual examples.

## Mask annotation

Masks must be created in CVAT or an equivalent pixel-annotation tool, not in a
spreadsheet.

Required indexed classes:

```text
0 = background/non-eye
1 = pupil
2 = visible iris tissue
3 = sclera/conjunctiva
4 = eyelid
5 = eyelash or dense lash occlusion
6 = corneal/specular reflection
7 = slit-beam or overexposed illumination artefact
8 = uncertain/ungradable region
```

Annotators label only visible structures. Hidden anatomy must never be inferred
or synthetically completed.

Twenty pilot images are predetermined for independent double mask annotation.
The remaining mask-eligible pilot images can receive one annotation plus
independent expert review.

## Agreement and adjudication

The software reports:

- percent agreement for eligibility;
- percent agreement for quality grade;
- percent agreement for segmentation feasibility;
- percent agreement for mask inclusion;
- quadratic-weighted Cohen's kappa for quality grade;
- image-level disagreement queue.

Kappa must be accompanied by the raw grade table and percent agreement because
its value depends on category prevalence.

An adjudicator reviews every core disagreement after both independent
submissions are frozen. Adjudication does not replace the original grades.

## Gate to the next phase

Proceed to large-scale annotation only after:

1. Both submissions pass validation.
2. Every core disagreement is adjudicated.
3. Quality definitions have accepted visual examples.
4. Each mask class has an operational boundary definition.
5. Mask export and class validation succeed.
6. Failure patterns and uncertain cases are documented.
7. The research team freezes protocol version 2.0.

There is no universal kappa threshold that automatically proves adequacy. The
team must examine class-specific disagreements and whether derived measurements
would materially change.
