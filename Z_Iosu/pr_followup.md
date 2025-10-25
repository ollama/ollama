# Vision Regression Follow-up (Simple Guide)

Purpose: Track and close the vision regression efficiently with consistent evidence.

## 1. Fill Image Inventory
Run the PowerShell inventory script (already in `ejecutadocker.md`) to produce `image_inventory.csv`.
Then test each candidate image using `Test-Vision` and set `VisionStatus` (OK / FAIL / ?).

## 2. Determine Boundary
- LKG (Last Known Good) = last image with OK.
- FKB (First Known Bad) = first image after LKG with FAIL.
Add a row at bottom:
```
Boundary: LKG=<digest_short>  FKB=<digest_short>
```

## 3. Map to Commits
Check labels in CSV:
- If `OCI_Revision` populated: use those as `<LKG_commit>` and `<FKB_commit>`.
- If missing: rebuild future images with label or manually record `git rev-parse HEAD` right after build.

## 4. Focused Diff
```
git diff <LKG_commit>..<FKB_commit> -- \
  server/images.go server/routes.go api/types.go \
  model/models llm/server.go integration/llm_image_test.go
```
Inspect for:
- Capability / vision detection changes.
- Token / template markers (image placeholders).
- Pointer vs value transition for image inputs.

## 5. Validate Hypothesis
Confirm if regression aligns with:
- `gguf:"v,vision"` -> `gguf:"v"` change.
- Input slice type change (value -> pointer) affecting image counting.

## 6. Optional Quick Mitigation
(Temporary patch until root fix)
- Force vision capability if metadata shows vision projector (only locally).
- Adjust Modelfile template to ensure image marker expansion.
Document any patch applied.

## 7. (If Needed) Narrow Further via Bisect
```
git bisect start <FKB_commit> <LKG_commit>
# For each step:
#  - build image (tag with short commit)
#  - run Test-Vision
#  - git bisect good / bad
```
Stop once single commit isolated.

## 8. Prepare Fix PR (Later)
Include:
- Symptom (ignored images / missing vision tokens).
- Root cause (file + line diff snippet).
- Fix description.
- Test evidence (before/after chat response, ImageCount log).

## 9. Update Documentation
- Update the table with final LKG/FKB.
- Add a short "Regression Boundary" section to `ejecutadocker.md`.

## 10. Completion Checklist
- [ ] Inventory filled
- [ ] LKG & FKB identified
- [ ] Commits mapped
- [ ] Diff reviewed
- [ ] Hypothesis confirmed/adjusted
- [ ] (Optional) Bisect completed
- [ ] Fix patch drafted
- [ ] Validation logs captured
- [ ] Docs updated

---
Keep everything evidence-driven: no status without logs or test output.
