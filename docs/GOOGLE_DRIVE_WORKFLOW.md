# Google Drive and Grader Procedure

## Data custodian

1. Run the pilot builder.
2. Confirm that `run_manifest.json` records 50 images and 50 participants.
3. Keep `private/` outside Google Drive.
4. Upload only the aliased files from `drive_upload/images/`.
5. Restrict access to named collaborators.
6. Disable public-link access.
7. Create a separate folder for each grader's workbook.
8. Do not place completed workbooks in a folder visible to other graders.
9. Back up every submitted workbook without overwriting the original.
10. Record submission date and file checksum.

## Adding clickable image links

After uploading images:

1. Enter each Google Drive URL in `shared/drive_links.csv`.
2. Match URLs by `blinded_image_id`.
3. Run `apply-links` against both clean workbooks.
4. Upload the updated workbooks.

The image alias in the sheet must match the alias displayed in the image
filename. Do not rename images after generating the pilot.

## Graders

1. Open only the assigned workbook.
2. Read `START HERE`.
3. Open `Review Images`.
4. Click `OPEN IMAGE`.
5. Complete the visible yellow cells using the dropdowns.
6. Record the date as `YYYY-MM-DD`.
7. Repeat until `Remaining` is 0.
8. Grade only visible structures.
9. Do not infer eye side, laterality, center, nasal, or temporal.
10. Do not discuss individual cases with the other grader before submission.
11. Return the completed file to the data custodian.

The grader identity is already assigned and hidden. Six detailed artifact
fields are hidden and optional: focus, exposure, reflection, slit beam,
eyelid/eyelash, and off-axis problems. The custodian can unhide them for
targeted review without changing the core submission.

Do not change blue reference cells, hidden machine headers, row count, image
identifiers, or column names. Google Sheets may remove Excel worksheet
protection; the validator still detects reference changes.

## Mask annotators

The spreadsheet tracks mask tasks but does not contain masks.

1. Use the CVAT project supplied by the data custodian.
2. Use only class IDs 0–8.
3. Label only visible tissue.
4. Mark genuinely ambiguous pixels as class 8.
5. Submit for review rather than editing another annotator's independent mask.
6. Export masks without resizing.
7. Name each mask `<blinded_image_id>_mask.png`.

## Submission control

The custodian validates each workbook before opening the other grader's
submission for comparison:

```bash
python3 -m openslit.collaboration validate \
  --submission completed_grader.xlsx \
  --index collaboration_runs/slit_pilot_v1/shared/pilot_image_index.csv
```

Invalid files are returned for correction. Original invalid submissions remain
archived for auditability.

## Privacy limitations

Aliasing reduces disclosure but is not complete anonymization. Eye photographs
are sensitive research data. Access control, approved storage, data-use
authorization, and institutional privacy requirements still apply.
