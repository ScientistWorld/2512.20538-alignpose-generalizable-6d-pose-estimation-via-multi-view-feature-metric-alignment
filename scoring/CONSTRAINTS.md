# Scientist Constraints

## Data

Use the same benchmark datasets and splits represented by this workspace: YCB-V, T-LESS, ITODD-MV, and HouseCat6D for the unseen-object evaluation; T-LESS for the seen-object comparison; HouseCat6D glass and metallic categories for the material subset evaluation. Do not add external labeled pose datasets, extra test-time scene views, or manually corrected labels. The benchmark is about better multi-view pose estimation under the paper's data conditions, not about expanding the data resource.

## Inputs

Use RGB images, known camera intrinsics, known camera extrinsics, and provided object meshes. Do not use depth images, point clouds captured by depth sensors, manually annotated object symmetries for the unseen-object setting, or additional calibration information unavailable to all methods. The paper's claim is specifically about calibrated multi-view RGB pose estimation for known meshes.

## View Protocol

Use the same view-grouping protocol as the reproduced evaluation. YCB-V, T-LESS, and HouseCat6D must use the sampled multi-view groups; ITODD-MV must use its predefined four views. Do not improve scores by increasing the number of test views beyond the benchmark setting.

## Training and Onboarding

For unseen-object experiments, do not train on the test object instances or their pose labels. A limited mesh-based onboarding/precomputation stage is allowed only if it uses the provided object model and no ground-truth test poses. The seen-object T-LESS experiment may use the same seen-object setting reported in the paper, but it must be reported separately from the unseen-object claim.

## Evaluation

Use BOP-style AR/AP metrics with VSD, MSSD, and MSPD pose-error functions as applicable. Methods must provide confidence scores for detection AP. Do not tune thresholds, view sampling, object subsets, or ranking rules on the test labels.

## Compliance Checklist

- [ ] Uses only the benchmark datasets and splits listed above
- [ ] Uses RGB, calibration, and object meshes; no depth sensor input
- [ ] Does not add extra test-time views beyond the benchmark protocol
- [ ] Does not train on unseen test object instances or their pose labels
- [ ] Does not require manually annotated symmetry metadata for unseen-object evaluation
- [ ] Reports AR/AP with BOP-style pose-error functions
- [ ] Keeps seen-object T-LESS results separate from unseen-object results
