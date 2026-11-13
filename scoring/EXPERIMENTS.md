# Experiment Overview

All three experiments use the same task: calibrated four-view RGB scenes, object meshes for an unseen object library, and noisy single-view pose candidates per view. Methods must return world-frame poses with confidences. Scoring is per image, BOP-style (see `briefing/evaluation.md`). The hidden test scenes are generated at test time; the visible train split has the same regimes, other scenes and other objects.

## textureless_clutter (weight 0.40)

Can a method localize plain, weakly textured objects in cluttered, occluded scenes better than the single-view candidates and a purely geometric fusion of them across views? This is where single-view estimates are least reliable (occlusion, missed detections, weak appearance cues) and where the paper reports its largest gains on a BOP dataset (T-LESS). AR measures how many objects are localized and how precisely; AP measures whether correct estimates are ranked above duplicates and spurious detections.

## textured_household (weight 0.25)

The same question on richly textured objects with lighter clutter (a YCB-V-like setting). Candidates are better here, so the margin between methods is smaller and precision at tight thresholds matters more.

## reflective_industrial (weight 0.35)

The same question on grayscale images of specular, weakly textured industrial parts (an ITODD-like setting), where appearance changes with viewpoint and per-view detection is least reliable. This is the hardest regime for every method.

## What the scores establish together

The benefit is localization accuracy (AR) in all three regimes. The constraint is that it must not come from flooding the output with hypotheses or from uncalibrated confidences, which AP penalizes; and from the protocol, that the method uses only the supplied RGB views, calibration, meshes and candidates, on objects it has not seen before. A method that fails or writes invalid output receives zero in every experiment.
