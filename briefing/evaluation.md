# Evaluation

Evaluation should measure 6D object localization and detection quality under the same calibrated multi-view conditions used by the paper. This file describes only the task metrics and protocol, not any solution internals.

## Metrics

- **AR**: Average Recall for 6D object localization. It is the average of recall under the BOP VSD, MSSD, and MSPD pose-error functions across their thresholds. Higher is better.
- **AP**: Average Precision for 6D object detection. It follows BOP/COCO-style ranked detection evaluation using pose-error correctness rather than 2D IoU. The paper averages AP over MSSD and MSPD. Higher is better.
- **AR_VSD**: Recall based on Visible Surface Discrepancy, which compares rendered visible object surfaces and accounts for occlusion. Higher is better.
- **AR_MSSD / AP_MSSD**: Recall or precision based on maximum symmetry-aware 3D surface distance. Higher is better.
- **AR_MSPD / AP_MSPD**: Recall or precision based on maximum symmetry-aware 2D projection distance. Higher is better.

## Baselines and Targets

Reference targets are stored in `scoring/reference.json`. The main comparison points are:

- Single-view unseen-object pose estimators evaluated directly on each view group.
- CosyPose multi-view refinement applied to the same candidate sources.
- Published seen-object multi-view baselines on T-LESS where the paper reports a secondary comparison.
- Ablation rows that isolate whether multi-view use, duplicate suppression, and final pose improvement each contribute to the final score.

## Evaluation Protocol

Use the official BOP-style pose evaluation wherever possible. Evaluate on YCB-V, T-LESS, ITODD-MV, and HouseCat6D multi-view scenes with the camera calibration and object meshes supplied by each dataset. For YCB-V, T-LESS, and HouseCat6D, compare methods on the same sampled view groups; for ITODD-MV, use the provided four-view scenes. Report both AR and AP for the main results, and include the individual VSD/MSSD/MSPD components when available. Methods must output confidence scores so AP can be computed fairly.
