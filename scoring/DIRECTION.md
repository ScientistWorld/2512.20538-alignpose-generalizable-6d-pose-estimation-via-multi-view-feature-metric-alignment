# Research Direction

## Research Problem

This gym studies model-based multi-view 6D pose estimation from calibrated RGB
views and known object geometry. The central question is how to turn noisy
single-view pose candidates into an accurate, globally consistent world-frame
pose.

## Approach Scope

Work should stay within calibrated multi-view object pose estimation. Valid
improvements may change aggregation, duplicate suppression, confidence scoring,
feature representation, robust loss, or pose optimization while preserving the
same inputs and output contract.

## Out of Bounds

- Using depth, RGB-D, captured point clouds, or extra sensors.
- Adding extra test-time views.
- Training on the benchmark's ground-truth poses.
- Hardcoding ground-truth poses or paper-reported scores.
- Reframing the task as 2D detection without reporting 6D pose metrics.
