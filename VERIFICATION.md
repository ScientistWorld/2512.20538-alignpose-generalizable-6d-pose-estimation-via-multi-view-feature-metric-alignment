# Verification Report

## Initial Assessment

The workspace had a working reduced AlignPose reproduction and a prior
`core_claim` result, but the current HEAD has changed since the last successful
CPU verification job. The latest completed job, `b00a7706-137`, regenerated
baseline outputs, method outputs, canonical scores, and train/test scores
through the older scheduler path. A later patch tightened the method-visible
candidate interface from world-frame poses to camera-frame poses and changed
`scripts/run.sh`; the existing score artifacts predate those source files.

## Issues Found

- **Critical**: Current prediction and score artifacts are stale relative to
  `method/alignpose.py`, `data/synthetic_inputs.json`, and `scripts/run.sh`.
  The final `core_claim` sign-off must wait for a CPU verification job that
  reruns the patched camera-frame candidate path.
- **Minor**: The executable benchmark remains a reduced synthetic substitute,
  and the reproduced AP is near the ceiling. The baseline gap is still large
  enough to be useful, but this limits confidence beyond the core claim.
- **Cosmetic**: Unrelated system files in the workspace root are dirty
  (`README.md`, `manifest.json`, and removed `trajectory/*.jsonl`). They were
  not part of this verification and should not be staged by this agent.

## Fixes Applied

- Re-checked method/problem separation: `eval/` and `data/` do not import from
  `method/`, and method-agnostic briefing/scoring files do not expose the
  implementation details.
- Verified that `scripts/download.sh` is self-contained and free of `shared/`
  paths, with an offline preflight suitable for compute nodes.
- Verified the three DINOv2-S download URLs with `curl -I -L --max-time 15`;
  all resolved to HTTP 200.
- Removed generated Python cache directories under `eval/`.
- Left `action.yaml` as a CPU test submission so the scheduler can regenerate
  all prediction and score artifacts for the current HEAD.

## Final Judgment

The workspace is structurally close to ready, but it should not be finalized as
`core_claim` until the pending CPU verification job completes on the current
camera-frame candidate interface. The evaluation code is separated from the
method, download scripts are self-contained and free of committed `shared/`
paths, and the stale score files still pass `python validate.py --compare`; the
remaining requirement is to refresh those artifacts from current source.

- Milestone: method_runs
- Ready for gym use: partial
- Confidence: medium
- Key limitation: current scores are stale relative to the latest method/data
  interface patch
