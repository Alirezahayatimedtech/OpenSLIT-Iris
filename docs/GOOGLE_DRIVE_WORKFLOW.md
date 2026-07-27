# Google Drive and Grader Procedure

This procedure is automated by `openslit-workflow`. The complete architecture is described in [End-to-end grader workflow](END_TO_END_WORKFLOW.md).

## Data custodian

1. Run the pilot builder and confirm that `run_manifest.json` records the intended participant and image counts.
2. Keep `private/` and the patient re-identification key outside Google Drive and CVAT.
3. Configure both grader Google emails, both CVAT usernames, the senior email, and the parent Drive folder in `configs/workflow_pilot_v1.json`.
4. Authenticate locally with a Google service account that has Editor access to the parent folder.
5. Run:

   ```bash
   openslit-workflow --config configs/workflow_pilot_v1.json bootstrap-drive
   ```

The command uploads only aliased images, creates the private folder structure, inserts image links into the workbooks, converts them to Google Sheets, and applies least-privilege permissions.

## Access structure

```text
OpenSLIT-Iris Pilot v1/
├── 01_Aliased_Images/      read-only: both graders + senior
├── 02_grader_01/           visible only to grader 01 and custodian
├── 03_grader_02/           visible only to grader 02 and custodian
├── 04_Adjudication/        senior access only after both submissions freeze
└── 05_Final_Consensus/     senior + custodian
```

The grader folder is shared read-only. Edit permission is granted directly on the quality Sheet. This allows the freeze command to remove edit access without hiding the frozen record.

Disable public-link access. Share only with named Google accounts.

## Graders

Each grader completes the entire independent pathway:

1. Open only the assigned private Google Sheet.
2. Read `START HERE`.
3. Open `Review Images`.
4. Click `OPEN IMAGE`.
5. Complete the visible yellow dropdown cells.
6. Record the review date as `YYYY-MM-DD`.
7. Continue until `Remaining` is 0.
8. Grade only visible structures.
9. Do not infer laterality, center, nasal, or temporal orientation.
10. Do not discuss individual cases with the other grader before both submissions freeze.
11. Tell the custodian that the Sheet is complete.

Do not change blue reference cells, hidden machine headers, row count, aliases, or column names. Google Sheets can remove Excel worksheet protection, but the submission validator still detects changed reference fields.

## Freezing the quality Sheet

The custodian runs:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json \
  freeze-grading --grader grader_01
```

The command:

- exports the Sheet as XLSX;
- checks image coverage and controlled vocabulary;
- rejects missing required responses;
- calculates SHA-256 hashes;
- archives a versioned snapshot;
- downgrades the grader from writer to reader;
- hides the first completed submission from the senior until the second submission also freezes.

Repeat for `grader_02`.

## Segmentation

After both quality submissions are frozen, the custodian runs:

```bash
openslit-workflow --config configs/workflow_pilot_v1.json setup-cvat
```

Each grader receives a separate CVAT project with the same predetermined double-annotation images. A grader must not be a CVAT administrator or staff member.

Protocol v1 uses class IDs `0–7`:

```text
0 background / other ocular tissue
1 pupil
2 visible iris
3 reflection
4 slit-beam artefact
5 eyelid
6 eyelash
7 uncertain / ungradable
```

Background is implicit in CVAT. Annotators draw the seven non-background labels and label only visible anatomy.

## Senior review

The senior sees the adjudication folder only after both independent submissions are frozen. The folder contains:

- both frozen quality workbooks;
- both frozen mask sets;
- quality disagreement tables;
- class-level Dice and IoU;
- disagreement overlays;
- geometric differences;
- the editable senior adjudication Sheet.

The senior may accept A, accept B, create consensus, mark ungradable, request protocol clarification, or request a versioned revision from either or both graders.

## Privacy limitations

Aliasing reduces disclosure but is not complete anonymization. Eye photographs are sensitive research data. Use approved storage, named-account access, appropriate data-use authorization, institutional privacy controls, and encrypted backups.
