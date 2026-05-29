# Verification Report

## Initial Assessment

The workspace had a working reduced AlignPose reproduction and a prior
`core_claim` result, but final packaging needed verification after recent fixes.
The latest CPU verification job, `b00a7706-137`, completed successfully and
regenerated baseline outputs, method outputs, canonical scores, and train/test
scores through `scripts/run.sh`.

## Issues Found

- **Minor**: `scores.json` was stale before the final verification job and did
  not yet include the full integrity metric surface.
- **Minor**: `scripts/evaluate.sh` previously scored fixed artifact paths
  instead of discovering all method prediction files.
- **Cosmetic**: A few method-agnostic guidance files used wording that was more
  specific than necessary for from-scratch scientist agents.

## Fixes Applied

- Updated `scripts/evaluate.sh` to score all `outputs/*_predictions.json`
  artifacts through the reusable evaluator interface.
- Added label-stripped `data/synthetic_inputs.json` for method-visible inputs,
  leaving ground truth only in the evaluation benchmark file.
- Refined `briefing/evaluation.md`, `scoring/TARGETS.md`, and
  `scoring/CONSTRAINTS.md` to keep from-scratch guidance method-agnostic.
- Ran CPU verification job `b00a7706-137`; it refreshed `scoring/scores.json`,
  `scoring/scores_train.json`, and `scoring/scores_test.json`.
- Verified download URLs for the DINOv2-S files with `curl -I -L`; all resolved
  to HTTP 200.

## Final Judgment

The workspace is ready to serve as a research gym at `core_claim`. The
evaluation code is separated from the method, download scripts are
self-contained and free of committed `shared/` paths, and `python validate.py
--compare` passes. The executable target is a reduced synthetic benchmark, so
future agents should treat the full BOP results as reference context rather
than as already reproduced coverage.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic reduced benchmark only
