# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-29 03:13] - method_runs
- Implemented the reduced AlignPose pipeline with world-frame candidate aggregation, 3D NMS, multi-view feature-metric SE(3) refinement, and artifact-based evaluation.
- Verified that `scripts/reproduce.sh` runs end-to-end on the synthetic calibrated multi-view benchmark.

### [2026-05-29 03:30] - core_claim
- Scheduler job `90d08839-921` completed successfully and wrote `scoring/scores.json`.
- On the runnable 12-scene synthetic benchmark, AlignPose improved over the single-view candidate baseline: AR 88.89 vs 47.22, AP 91.67 vs 50.00, mean ADD error 0.0339 vs 0.0820, and final duplicate candidates 1.0 vs 12.0.
- The result supports the central claim at reduced scale: calibrated multi-view aggregation plus feature-metric refinement improves noisy single-view 6D pose candidates without depth input or object-specific training.

### [2026-05-29 03:45] - method_runs
- Continuing audit found that the previous prediction writer selected the final refined hypothesis using ground-truth ADD error. This leaked evaluation labels into method outputs.
- Replaced final hypothesis selection with the method's own feature-alignment confidence score after final NMS. The stale `core_claim` result now requires a fresh container run before being restored.

### [2026-05-29 04:16] - core_claim
- CPU test job `cba204d5-f6f` reran `scripts/reproduce.sh` successfully after the leakage fix.
- The regenerated artifact-based scores remained: single-view candidate AR 47.22 / AP 50.00, AlignPose AR 88.89 / AP 91.67.
- This restores `core_claim` with predictions selected only by method confidence, not ground-truth pose error.

## Stop Justification
- Completed at milestone `core_claim`.
