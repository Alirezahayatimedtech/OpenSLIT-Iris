# Contributing to OpenSLIT-Iris

## Image graders

Image graders do not need GitHub or local code.

1. Use only the Google Sheet assigned by the data custodian.
2. Read `START HERE`.
3. Complete `Review Images` independently.
4. Stop when `Remaining` is 0.
5. Notify the data custodian. Do not upload images or grades to GitHub.

## Data custodians

1. Keep source identifiers and `private/` outside shared Drive folders and Git.
2. Give each grader a separate workbook.
3. Freeze and archive each submission before comparing graders.
4. Validate both submissions.
5. Run the merge and adjudication workflow.
6. Record any protocol change in the documentation and use a new pilot version.

## Using another dataset

1. Copy `templates/source_manifest_template.csv`.
2. Enter one trusted participant identifier and image path per row.
3. Generate the image profile.
4. Copy `configs/pilot_template.json` and set the dataset paths and sample sizes.
5. Build the pilot into a new output directory.
6. Review `run_manifest.json`.
7. Upload only aliased images and separate grader workbooks.

Use one participant-level split unit. Do not use filenames, laterality, view
labels, disease labels, or outcomes to select the pilot.

## Code and protocol changes

1. Create a branch from `main`.
2. Make one scoped change.
3. Run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q openslit tests
```

4. Open a pull request describing the scientific or operational reason.
5. Require review before merging changes that affect selection, blinding,
   grading definitions, validation, agreement analysis, or mask classes.

Never commit source images, participant mappings, generated pilot runs,
credentials, completed grading files, or clinical data.
