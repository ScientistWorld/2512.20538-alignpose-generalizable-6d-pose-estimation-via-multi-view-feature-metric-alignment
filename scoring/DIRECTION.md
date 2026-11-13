# Research Direction

## Research Problem

Model-based multi-view 6D pose estimation of unseen objects from calibrated RGB views: turning noisy, partly missing and partly wrong single-view pose candidates into accurate, globally consistent world-frame poses with calibrated confidences, using the images and the object meshes.

## Approach Scope

Any approach that keeps the inputs and the output contract: how candidates from different views are combined, which hypotheses are kept and refined, how image evidence from all views is used, which image features or losses are used, how poses are optimized, how confidences are computed, and how objects missed by the candidates could be recovered.

## Where there is room

The reference leaves room in every regime: coarse hypotheses outside the convergence range of a local refinement, wrong hypothesis selection inside a group of duplicates, objects detected in few views, specular and textureless appearance, and confidence ranking. Gains should hold in all three regimes, not only in one.

## Out of Bounds

- Depth, extra views or other sensors.
- Training on test objects or test scenes, or reading the test code or answers.
- Hardcoding poses, scene-specific tables or paper-reported numbers.
- Exploiting the scorer instead of improving poses (e.g. flooding hypotheses or manipulating confidences).
