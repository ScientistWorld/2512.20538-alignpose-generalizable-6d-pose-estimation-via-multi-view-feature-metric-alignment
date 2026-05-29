# Evaluation Targets

## Primary: unseen_multiview_pose

Tests overall 6D pose localization and detection on unseen-object multi-view scenes across YCB-V, T-LESS, ITODD-MV, and HouseCat6D.
- **Metrics**: `ar`, `ap` (higher is better)
- **Constraint metrics**: `object_specific_training` and `symmetry_annotations_required` (lower is better)
- **How to run evaluation**: run the BOP-style multi-view evaluation on the same dataset/view-group settings and candidate-source conditions represented in `reference.json`.

## Constraint: challenging_housecat6d_materials

Tests whether performance holds on the difficult HouseCat6D material subsets: transparent glass and metallic cutlery.
- **Metrics**: `ar`, `ap` (higher is better)
- **Constraint metrics**: `object_specific_training` and `rgb_only` (object-specific training lower is better; RGB-only higher is better)
- **How to run evaluation**: filter HouseCat6D results to the glass and metallic category groups and compute the same AR/AP metrics.

## Constraint: seen_tless_pose

Tests whether the method remains competitive on a seen-object T-LESS setting while preserving a fixed four-view RGB evaluation setup.
- **Metrics**: `ar`, `ap` (higher is better)
- **Constraint metrics**: `num_views` and `rgb_only` (num_views should remain fixed; RGB-only higher is better)
- **How to run evaluation**: evaluate on the T-LESS seen-object comparison setting with the same view count and reported baseline rows.

## Ablation: component_contribution

Tests whether using the full multi-view pose-estimation pipeline improves over partial system variants under the same YCB-V and T-LESS ablation setting.
- **Metrics**: `ar`, `ap` (higher is better)
- **Constraint metrics**: `num_views` (fixed by row)
- **How to run evaluation**: run each ablation condition on the same scenes and report AR/AP with the same candidate source.
