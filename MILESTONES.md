# Reproduction Milestones

**Current: method_runs**

Milestone levels:

- `none` - documentation/scoring scaffold only; no method has run
- `method_runs` - the paper's method executes end-to-end without errors
- `core_claim` - minimum experiment supports the paper's central claim
- `core_claim_plus` - core claim reproduced on additional settings
- `secondary_claims` - secondary results or contributions reproduced
- `majority` - more than half of reported results reproduced
- `near_complete` - most results reproduced, only minor gaps remain
- `full` - all reported results reproduced
- `not_possible` - faithful implementation is infeasible and no surrogate is claimed

## Progress Log

### [2026-05-29 03:07] - none
- Initialized method-agnostic documentation and scoring scaffold for AlignPose.

### [2026-05-29 03:12] - method_runs
- Implemented a reduced AlignPose pipeline with world-frame candidate aggregation, 3D NMS, multi-view feature-metric SE(3) refinement, and final NMS.
- Ran the end-to-end reproduction locally on the 12-scene synthetic benchmark; scheduler job is being submitted to verify the packaged environment.

## Stop Justification

- Not stopping. The next job verifies whether the reproduced results support `core_claim`.
