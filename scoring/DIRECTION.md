# Research Direction

## Research Problem

This benchmark studies generalizable multi-view 6D pose estimation for rigid objects from calibrated RGB images and known object meshes. The central question is how to turn ambiguous, incomplete, or noisy evidence from several views into accurate object poses in a shared 3D frame, especially when the object instances are unseen during training.

## Core Contribution

The paper's contribution is in multi-view pose aggregation, scoring, and refinement for object pose estimation under an RGB-only, calibrated-camera setting. A scientist should improve the way a system combines multi-view evidence into final 6D poses while preserving the problem formulation and evaluation conditions.

## Approach Scope

Work should stay within model-based, calibrated multi-view object pose estimation. Improvements may change candidate aggregation, duplicate handling, confidence scoring, pose refinement, feature representations, optimization, robustness to missing views, or handling of occlusion and ambiguous objects. The system may use existing single-view candidate sources as inputs or replace them with another valid unseen-object RGB candidate generator, as long as the same input and evaluation constraints are respected.

## Out of Bounds

- Switching to a depth-based, RGB-D, point-cloud, or active-sensing pose-estimation problem.
- Training specifically on the unseen test object instances or using their test poses outside official evaluation.
- Beating the benchmark by adding more test-time views, new labeled datasets, manual pose corrections, or manual symmetry annotations not available under the unseen-object protocol.
- Reframing the task as category-level pose estimation, 2D detection only, mesh reconstruction, or object tracking without reporting the required 6D pose metrics.
- Reporting only localization recall or only detection precision when the relevant experiment requires both AR and AP.
