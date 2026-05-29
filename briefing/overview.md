# Overview

- **Paper ID:** 2512.20538
- **Title:** AlignPose: Generalizable 6D Pose Estimation via Multi-view Feature-metric Alignment
- **Domain:** Computer Vision / 6D Object Pose Estimation
- **TL;DR:** The paper targets RGB-only, multi-view 6D pose estimation for novel rigid objects using calibrated cameras and known object meshes.

## Short Summary

This paper studies model-based 6D object pose estimation from multiple calibrated RGB views. The target setting is challenging because the object instances may be unseen during training, single-view pose candidates can be noisy or ambiguous, and scenes may contain clutter, occlusion, textureless surfaces, metallic objects, or transparent objects. Evaluation follows BOP-style localization and detection metrics on YCB-V, T-LESS, ITODD-MV, and HouseCat6D. The reported results show large gains over single-view estimates and the published CosyPose multi-view baseline, especially on industrial and difficult-material settings.

## Key Results

- On unseen-object multi-view evaluation, the paper reports higher AR and AP than both single-view inputs and CosyPose multi-view refinement across YCB-V, T-LESS, ITODD-MV, and HouseCat6D.
- On HouseCat6D glass and metallic subsets, the reported gains are largest for transparent glass objects and meaningful but harder for thin metallic cutlery.
- On seen-object T-LESS evaluation, the reported system remains competitive with methods trained on the test objects while using the multi-view RGB evaluation setup.
