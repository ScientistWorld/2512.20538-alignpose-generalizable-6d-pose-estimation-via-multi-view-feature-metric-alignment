# Experiment Overview

## unseen_multiview_pose

Tests whether a reproduced system solves the paper's central task: estimating 6D object poses for unseen objects from calibrated multi-view RGB scenes. The experiment compares against single-view candidate sources and a published multi-view baseline on the datasets used for the main claim.

## challenging_housecat6d_materials

Tests whether the main effect persists on difficult object materials in HouseCat6D. This experiment focuses on transparent glass and metallic objects because these categories expose failures that may be hidden by aggregate dataset scores.

## seen_tless_pose

Tests whether the system remains competitive in a seen-object industrial setting on T-LESS. The purpose is to check that the multi-view pose-estimation result is not limited to the unseen-object protocol, while keeping the comparison on the same benchmark family.

## component_contribution

Tests whether the complete system is better than partial multi-view variants under the paper's ablation setting. The purpose is to verify that the reported improvement is not explained only by adding more views or by producing more candidate detections.
