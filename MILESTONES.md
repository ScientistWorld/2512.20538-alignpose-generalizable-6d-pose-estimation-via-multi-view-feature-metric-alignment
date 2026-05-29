# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-29 03:22] - method_runs
- Previous reduced AlignPose pipeline ran end-to-end on the scheduler.
- After audit, revised the method to use local DINOv2-S patch features, rendered RGB-D registered features, and explicit scored protocol constraints.
- Submitting a CPU test job to validate the revised end-to-end pipeline before restoring any `core_claim` claim.

### [2026-05-29 03:27] - method_runs
- CPU test job `216bf4fb-60f` failed during DINO preprocessing setup because `AutoImageProcessor` required `torchvision`.
- Replaced `AutoImageProcessor` with direct local preprocessing from `preprocessor_config.json`, keeping offline DINOv2-S features and avoiding the missing dependency.
- Resubmitting the revised end-to-end reproduction job.

### [2026-05-29 03:35] - core_claim
- CPU test job `926ce860-7c2` completed successfully with the revised DINOv2-S query/render feature pipeline.
- On the 12-scene reduced benchmark, AlignPose improved over the single-view candidate baseline from AR 47.22 / AP 50.00 to AR 97.22 / AP 100.00.
- The score artifact includes the core constraints: four calibrated RGB views, no depth sensor, no object-specific training, frozen DINOv2 features, and duplicate suppression.

### [2026-05-29 03:58] - method_runs
- Packaging audit found that `scripts/run.sh` evaluated stale prediction artifacts and that canonical `scores.json` lacked the newer integrity metrics.
- Updated scheduler execution to regenerate baseline and AlignPose predictions before all scoring steps.
- Split method-visible synthetic inputs from evaluation labels by adding `data/synthetic_inputs.json`; submitting a CPU test job before restoring `core_claim`.

### [2026-05-29 04:08] - method_runs
- Final verification found that the last completed CPU job predated the patched scheduler script, so canonical `scores.json` remained stale.
- Updated `scripts/evaluate.sh` to discover all prediction artifacts through the reusable evaluator interface.
- Submitting a CPU test job to regenerate method outputs and all score files before final `core_claim` sign-off.

### [2026-05-29 04:25] - core_claim
- CPU verification job `b00a7706-137` completed successfully with exit code 0.
- `scripts/run.sh` regenerated baseline and AlignPose prediction artifacts, then refreshed canonical, train, and test score files.
- Final scores support the core claim on the reduced benchmark: AlignPose improved over the candidate baseline from AR 47.22 / AP 50.00 to AR 97.22 / AP 100.00 with no ground-truth pose copying.

### [2026-05-29 04:30] - method_runs
- Tightened the synthetic interface so method-visible candidates are camera-frame `T_CO` poses, matching the paper's per-view input contract.
- Updated AlignPose and the single-view baseline to transform candidates through known camera extrinsics before world-frame aggregation and scoring.
- Added an offline scheduler preflight so compute jobs verify cached DINOv2 files without attempting network access.
- Submitting a CPU verification job before restoring the `core_claim` milestone for this patched method path.

## Stop Justification
- Not stopped; CPU verification is pending for the camera-frame candidate interface.
