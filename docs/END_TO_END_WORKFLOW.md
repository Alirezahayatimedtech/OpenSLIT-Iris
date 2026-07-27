# End-to-end grader workflow

This workflow keeps two ophthalmologists independent from image-quality grading through pixel segmentation, then gives a senior ophthalmologist only the disagreements and frozen source submissions.

## Roles

| Role | Sees during independent work | Main responsibility |
|---|---|---|
| Data custodian | All operational resources and the private re-identification key | Build the pilot, provision access, freeze submissions, run comparisons |
| Grader 01 | Read-only aliased images, private quality Sheet, assigned CVAT project | Grade all pilot images and segment the predetermined double-annotation subset |
| Grader 02 | Read-only aliased images, private quality Sheet, assigned CVAT project | Perform the same work independently |
| Senior ophthalmologist | Shared images; adjudication folder only after both quality submissions freeze | Review disagreements, accept one result, create consensus, or request a versioned revision |

The private source key is never uploaded to Google Drive or CVAT.

## System responsibilities

- **GitHub:** code, controlled vocabularies, protocol versions, and reproducible configuration.
- **Google Drive:** aliased images, private grader Sheets, frozen submission snapshots, and senior-review reports.
- **CVAT Community:** pixel-level annotation in two isolated projects.
- **OpenSLIT workflow code:** gates, validation, hashes, state transitions, exports, disagreement metrics, revision records, and finalization.

## Google Drive folder layout

The bootstrap command creates:

```text
OpenSLIT-Iris Pilot v1/
├── 01_Aliased_Images/
├── 02_grader_01/
├── 03_grader_02/
├── 04_Adjudication/
└── 05_Final_Consensus/
```

Access policy:

- `01_Aliased_Images`: read-only for both graders and the senior.
- Grader folders: read-only visibility for the matching grader; the quality Sheet has direct edit permission while active.
- `04_Adjudication`: not shared with the senior until both quality submissions are frozen.
- `05_Final_Consensus`: controlled by the custodian and senior.

Because edit permission is granted directly on each Sheet rather than inherited from its folder, the freeze command can downgrade a grader from writer to reader without removing access to the record.

## Before running

1. Build the blinded pilot:

   ```bash
   python -m openslit.collaboration build \
     --config configs/pilot_slit_dataset.json
   ```

2. Install the integrated dependencies:

   ```bash
   python -m pip install -e '.[cvat,google]'
   ```

3. Start local CVAT:

   ```bash
   cp deployment/cvat/.env.example deployment/cvat/.env
   chmod +x deployment/cvat/cvat.sh
   deployment/cvat/cvat.sh up
   deployment/cvat/cvat.sh create-superuser
   ```

4. Create normal CVAT accounts for the two graders. They must not be staff or administrators.

5. Create a Google Drive parent folder and share it as Editor with the service-account email. Copy its folder ID.

6. Copy `deployment/google/.env.example` to a private local environment file and set `GOOGLE_APPLICATION_CREDENTIALS`.

7. Edit `configs/workflow_pilot_v1.json`:

   - both grader Google emails;
   - both CVAT usernames;
   - senior Google email;
   - senior CVAT username if used;
   - Google Drive parent folder ID.

8. Validate:

   ```bash
   openslit-workflow --config configs/workflow_pilot_v1.json check
   ```

## Stage 1: provision Drive and begin quality grading

```bash
openslit-workflow --config configs/workflow_pilot_v1.json bootstrap-drive
```

This command:

- creates the controlled folder structure;
- uploads aliased images once;
- grants read-only image access;
- inserts Drive image links into both workbooks;
- uploads and converts each workbook into a private Google Sheet;
- grants each grader edit access only to their own Sheet;
- writes the resource IDs to the local workflow state.

Each grader completes their own Sheet without seeing the other grader's answers.

## Stage 2: freeze both quality submissions

After a grader confirms completion:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-grading --grader grader_01
```

Repeat for `grader_02`.

The command exports the Google Sheet as XLSX, validates all required fields, computes SHA-256 hashes, saves a versioned snapshot, uploads the snapshot into the hidden adjudication folder, and removes the grader's edit permission. The senior receives access to the adjudication folder only after both graders are frozen.

Original independent submissions are never overwritten.

## Stage 3: create isolated CVAT projects

```bash
openslit-workflow --config configs/workflow_pilot_v1.json setup-cvat
```

This command refuses to run until both quality-grading submissions are frozen. It creates:

```text
OpenSLIT-Iris Pilot v1 - grader_01
OpenSLIT-Iris Pilot v1 - grader_02
```

Each project contains the same predetermined double-annotation images and one task assigned to the matching CVAT account. The background class remains implicit. The machine-readable annotation schema supplies the seven drawable labels.

The graders annotate independently and mark their task complete in CVAT.

## Stage 4: freeze both CVAT submissions

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-segmentation --grader grader_01
```

Repeat for `grader_02`.

The command exports CVAT's Segmentation Mask format, converts label colors or indices to the OpenSLIT class IDs, creates indexed PNG masks, validates dimensions and allowed classes, computes hashes, and saves an immutable versioned snapshot.

Do not edit the original CVAT task after its export is frozen. A revision must use a new version.

## Stage 5: build the senior package

```bash
openslit-workflow --config configs/workflow_pilot_v1.json build-adjudication
openslit-workflow --config configs/workflow_pilot_v1.json upload-adjudication
```

The package includes:

- the original image;
- both frozen masks;
- disagreement overlays;
- Dice and IoU for every class;
- pupil-center difference;
- visible-iris area difference;
- merged quality-grade disagreements;
- a senior adjudication Google Sheet.

The senior selects one outcome for every disputed image:

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

## Stage 6: request a revision

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  request-revision \
  --image-id PILOT-I017 \
  --from-grader grader_01 \
  --reason "Superior eyelash was labelled as iris" \
  --protocol-reference "Annotation Protocol v1.0, Eyelash class"
```

The request is appended to a permanent audit file and changes the affected grader's segmentation state to `REVISION_REQUESTED`. The frozen v1 mask remains untouched. Create a pre-populated correction task:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  create-revision-task --grader grader_01
```

The new task contains only the disputed images and imports the grader's latest frozen masks as the starting point. Revised work is exported as v2.

## Stage 7: finalize

Download the completed senior queue as CSV and run:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  finalize-adjudication \
  --queue /path/to/completed_senior_adjudication_queue.csv
```

Finalization fails when an image has no senior outcome or an unresolved revision request.

## Workflow states

Quality grading:

```text
NOT_STARTED -> IN_PROGRESS -> SUBMITTED -> FROZEN
                                      FROZEN -> REVISION_REQUESTED
```

Segmentation:

```text
LOCKED -> ASSIGNED -> IN_PROGRESS -> SUBMITTED -> FROZEN
                                               FROZEN -> REVISION_REQUESTED
```

Senior review:

```text
LOCKED -> READY -> IN_PROGRESS -> REVISION_REQUESTED -> FINALIZED
```

The local state file and generated submissions are under `collaboration_runs/` and are excluded from Git.

## Security and operational rules

- Use aliases only.
- Never place the private patient key in Drive or CVAT.
- Do not make graders CVAT staff or administrators.
- Do not share grader folders with each other.
- Do not expose disagreement reports until both source submissions are frozen.
- Do not overwrite a frozen XLSX, CVAT export, normalized mask, or final decision.
- Back up the CVAT volumes and the Google Drive root folder.
- Use HTTPS or a trusted VPN when CVAT is accessed outside the local network.
