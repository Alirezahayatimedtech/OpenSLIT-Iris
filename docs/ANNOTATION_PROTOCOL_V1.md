# OpenSLIT-Iris Annotation Protocol v1.0

## Status

This document defines the first frozen annotation protocol for the OpenSLIT-Iris pilot. It is intended for ophthalmologists and trained reviewers creating semantic-segmentation masks from frontal slit-lamp photographs.

The protocol covers only annotation. It does not provide diagnosis, infer disease, or assign laterality, nasal/temporal orientation, or clinical outcomes.

## Objective

Create reproducible pixel-level labels for:

1. pupil;
2. visible iris;
3. corneal/specular reflection;
4. slit-beam artefact;
5. eyelid occlusion;
6. eyelash occlusion;
7. uncertain or ungradable regions.

The resulting masks will support later segmentation-model development and quantitative iris feature extraction.

## Supported images

Include:

- frontal or nearly frontal visible-light slit-lamp photographs;
- diffuse or broad-beam illumination;
- one eye per image;
- images in which the pupil and some iris tissue are visible.

Do not use this protocol for gonioscopy, retroillumination, narrow optical sections, AS-OCT, UBM, smartphone photographs, or images containing two eyes.

## Mask format

- Use one indexed PNG mask per image.
- The mask dimensions must exactly match the source image.
- Each pixel receives exactly one class.
- Do not draw inferred anatomy behind an occluder.
- Do not resize, crop, rotate, recolor, or enhance the source image during annotation.

## Class definitions

### 0. Background / other ocular tissue

Label as background:

- sclera and conjunctiva;
- cornea that is not represented by a reflection or slit-beam artefact;
- surrounding skin not included as an occluding eyelid;
- image borders and non-eye background;
- any other pixel not assigned to a target class.

### 1. Pupil

Label the visible pupillary aperture bounded by the pupillary margin.

Include:

- irregular visible pupil shapes;
- visible pupil in posterior synechiae or distorted pupils.

Exclude:

- iris tissue at the pupillary margin;
- reflections crossing the pupil;
- slit-beam pixels crossing the pupil;
- eyelash or eyelid occlusion;
- regions whose boundary cannot be determined confidently.

Do not force the pupil into a circle or ellipse.

### 2. Visible iris

Label visible iris tissue from the pupillary margin to the visible limbal boundary.

Include:

- visible crypts, furrows, collarette, pigment spots, atrophy, and lesions as iris tissue in protocol v1;
- abnormal iris tissue when its boundary remains visible.

Exclude:

- pupil;
- sclera and conjunctiva;
- tissue hidden by eyelids or eyelashes;
- reflection and slit-beam artefacts;
- uncertain regions.

Do not extrapolate the iris under the eyelid, eyelashes, glare, or an invisible limbal boundary.

### 3. Corneal / specular reflection

Label discrete bright reflections that obscure underlying ocular texture.

Typical examples:

- small circular or irregular white highlights;
- saturated reflections from the illumination or camera system;
- reflection clusters over the iris or pupil.

Do not label naturally light iris tissue as reflection. A bright region should be labelled reflection only when it behaves as a surface highlight and removes meaningful underlying texture.

### 4. Slit-beam artefact

Label the illumination band produced by the slit lamp when it materially changes or obscures the appearance of the iris or pupil.

Include:

- narrow or broad vertical, oblique, or curved bright bands;
- saturated central cores and visibly altered halo regions when underlying texture is no longer reliable.

Exclude diffuse illumination that does not obscure texture.

### 5. Eyelid occlusion

Label visible upper or lower eyelid tissue when it occludes the globe or defines a lid-covered region.

Include:

- lid margin and adjacent lid tissue crossing the ocular surface;
- lid tissue covering the peripheral iris.

Do not label all surrounding periocular skin. The aim is to identify clinically relevant occlusion.

### 6. Eyelash occlusion

Label individual or dense eyelashes crossing the ocular surface, iris, or pupil.

Include:

- isolated visible lashes;
- dense lash shadows when the underlying iris cannot be evaluated reliably.

Do not label lashes entirely outside the ocular region unless they contact or obscure the eye.

### 7. Uncertain / ungradable region

Use only when a pixel cannot be assigned confidently after applying the definitions above.

Examples:

- invisible limbal boundary caused by blur or haze;
- overlap between two artefacts that cannot be separated;
- severe saturation where reflection and slit beam cannot be distinguished;
- ambiguous eyelash shadow without a visible lash boundary.

Do not use uncertain as a substitute for careful annotation. Add a brief comment when the uncertain area is substantial.

## Class precedence

Because the mask permits one class per pixel, apply this order when classes overlap:

1. uncertain;
2. eyelash;
3. eyelid;
4. reflection;
5. slit beam;
6. pupil;
7. iris;
8. background.

This means that visible occlusion or artefact replaces the hidden anatomical class. For example, a corneal reflection over the pupil is labelled reflection, not pupil.

## Boundary rules

- Trace the visible boundary at the highest practical zoom.
- Follow real pathological irregularity.
- Do not smooth a distorted pupil into an ideal shape.
- Do not infer the outer iris boundary where it is hidden.
- When a boundary is blurred but still reasonably identifiable, place it at the midpoint of the transition zone.
- When a boundary cannot be identified reproducibly, use the uncertain class.
- Small isolated annotation holes should be corrected unless they represent a real excluded structure.

## Image-level gradability

Mark an image as gradable when both pupil and visible iris can be annotated sufficiently for protocol evaluation.

Mark it ungradable when:

- the pupil cannot be identified;
- almost no iris is visible;
- severe blur, saturation, or occlusion prevents reproducible boundaries;
- the image is not a supported slit-lamp acquisition.

Ungradable images remain in the audit and failure analysis. They must not be silently deleted.

## Pilot workflow

1. Select 20–30 representative pilot images.
2. Two ophthalmologists independently annotate the same images.
3. Each annotator works without viewing the other annotator's masks.
4. Validate mask files automatically.
5. Generate overlays and disagreement maps.
6. Review class-specific disagreement.
7. Adjudicate ambiguous cases with a senior ophthalmologist.
8. Add accepted examples and counterexamples to this manual.
9. Freeze the resulting document as protocol v1.0 before large-scale annotation.

The current collaborative pilot has 20 predetermined images for independent double-mask annotation.

## Required annotation manifest

Each annotation submission must include:

```text
image_id
image_file
mask_file
annotator_id
protocol_version
```

Optional fields:

```text
gradable
review_status
comments
```

Use `1.0.0` as the protocol version for this release.

Do not place patient names, medical-record numbers, source patient identifiers, laterality, diagnosis, or outcome fields in the shared annotation manifest.

## Quality assurance before submission

For every mask, verify:

- source image and mask dimensions match;
- only class IDs 0–7 are present;
- pupil and iris are present for gradable images;
- no hidden anatomy was inferred under occlusion;
- reflection and slit-beam regions are excluded from iris and pupil;
- filenames match the shared manifest;
- protocol version and annotator identity are complete;
- uncertain regions have comments when substantial.

Run:

```bash
openslit-validate-masks \
  --schema configs/annotation_schema_v1.json \
  --manifest annotations/annotation_manifest.csv \
  --images annotations/images \
  --masks annotations/masks \
  --report annotations/validation_report.csv
```

## Inter-annotator evaluation

After both submissions are frozen, calculate separately for each class:

- Dice coefficient;
- Intersection over Union;
- boundary disagreement;
- pupil-center difference;
- visible-iris area difference.

Disagreement does not automatically mean one annotator is wrong. It identifies rules that require clarification or adjudication.

## Protocol-change control

Any substantive change to class definitions, precedence, supported acquisitions, or boundary rules requires:

- a new protocol version;
- a written change log;
- revalidation of affected annotations;
- explicit separation from masks created under earlier versions.

Do not silently alter protocol v1.0 after the pilot is frozen.
