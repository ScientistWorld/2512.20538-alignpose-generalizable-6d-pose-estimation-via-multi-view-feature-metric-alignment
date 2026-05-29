# Verification Report

## Initial Assessment

The workspace had a working reduced AlignPose reproduction and a prior
`core_claim` result, but the current HEAD had changed since the previous
successful CPU verification job. A later patch tightened the method-visible
candidate interface from world-frame poses to camera-frame poses and changed
`scripts/run.sh`, so stale artifacts had to be regenerated before final
sign-off.

## Issues Found

- **Resolved critical**: Prediction and score artifacts were stale relative to
  `method/alignpose.py`, `data/synthetic_inputs.json`, and `scripts/run.sh`.
  CPU verification job `0068b751-1e0` reran the patched camera-frame candidate
  path and refreshed all scoring outputs successfully.
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
- Reviewed CPU verification job `0068b751-1e0`; it exited with code 0 after a
  scheduler preemption and restart, regenerated canonical/train/test scores,
  and left score artifacts newer than the patched source files.
- Re-ran `python validate.py --compare`; validation and structural gym checks
  pass.

## Final Judgment

The workspace is ready to serve as a research gym at `core_claim`. The current
artifacts were regenerated from the patched method-visible camera-frame
candidate interface, evaluation code is separated from the method, download
scripts are self-contained and free of committed `shared/` paths, and
`python validate.py --compare` passes.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic reduced benchmark only
