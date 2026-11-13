# Problem Definition

This benchmark concerns multi-view 6D pose estimation for rigid objects. A solver receives four RGB images of the same scene, the camera intrinsics, the camera extrinsics in a shared world frame, a textured 3D mesh for every object that may appear, and, for every view, a set of noisy pose candidates from a single-view pose estimator. It must output, for every scene, the objects present, a world-frame 6D pose for each, and a confidence score for each.

The objects at test time are new: the hidden test scenes use an object library that does not appear in the visible train split. Only the meshes are available for them, so a solver may not train on the test objects; a short mesh-only onboarding at run time is allowed. No depth input is provided.

The single-view candidates are imperfect in the ways real single-view estimators are: their distance from the camera is uncertain, their orientation can be off by several degrees and occasionally grossly wrong, an object can be missed in some views (especially when occluded) and spurious detections occur, including wrong object identities. Some objects are detected in only one or two of the four views.

## Scene regimes

The benchmark has three regimes, scored separately:

- **textureless_clutter**: plain, weakly textured objects among several distractors, with frequent occlusion (a T-LESS-like setting).
- **textured_household**: richly textured objects with lighter clutter (a YCB-V-like setting).
- **reflective_industrial**: grayscale images of specular, weakly textured parts on a dark background (an ITODD-like setting).

## Research Question

Can a system accurately estimate 6D poses of previously unseen objects from calibrated multi-view RGB images and noisy single-view candidates, without object-specific training or depth input, and do better than both the single-view candidates and a geometric fusion of those candidates across views?

## Why It Matters

Accurate 6D poses are needed for robot manipulation, inspection, augmented reality and scene understanding. Calibrated multi-camera RGB setups are common in industrial cells; they can resolve the depth ambiguity, occlusions and appearance ambiguities that limit single-view estimators, without the cost and failure modes of depth sensors on reflective objects.

## Success Criteria

A successful solution finds the objects that are present (recall), localizes them precisely in 3D and in every image (MSSD and MSPD errors), ranks correct estimates above wrong or duplicate ones (precision), and does so in all three regimes, including the textureless and reflective ones.
