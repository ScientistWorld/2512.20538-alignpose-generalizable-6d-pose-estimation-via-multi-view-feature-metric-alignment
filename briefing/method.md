# Paper's Method

## Key Contribution

The paper introduces AlignPose, a generalizable RGB-only multi-view 6D pose estimation pipeline for known object meshes and calibrated cameras. Its core contribution is a multi-view feature-metric refinement objective that optimizes a single world-frame object pose against image evidence from all available views.

## Approach

The pipeline starts from per-view pose candidates produced by existing single-view unseen-object pose estimators. It transforms those candidates into a common world frame using the known camera extrinsics, then removes duplicate or overlapping hypotheses with 3D non-maximum suppression. Each remaining candidate is refined independently by comparing feature descriptors from rendered object views with feature maps extracted from the observed images. The optimized pose is the single world-frame pose that minimizes the sum of feature discrepancies across views, and the final confidence score is derived from average multi-view feature alignment quality.

## Main Claims

The primary claim is that multi-view feature-metric pose refinement improves AR and AP for unseen-object 6D pose estimation across BOP-style datasets, outperforming both single-view candidate estimates and CosyPose multi-view refinement. Secondary claims are that the method is useful for difficult industrial and household materials, does not require object-specific training or symmetry annotations, remains competitive on seen-object T-LESS comparisons, and benefits from its aggregation, duplicate-removal, refinement, and scoring choices.
