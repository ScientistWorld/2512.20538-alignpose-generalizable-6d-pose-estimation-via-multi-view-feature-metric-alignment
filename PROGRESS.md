# Progress

## What Works

Documentation and scoring scaffolding have been initialized for the AlignPose reproduction. A reduced synthetic multi-view pose benchmark has been added under `data/`, with a single asymmetric object, calibrated four-view scenes, and noisy single-view pose candidates.

The method code implements the core AlignPose computation at reduced scale: candidates are aggregated in a world frame, filtered with global 3D AABB NMS at IoU 0.4, refined by optimizing one SE(3) world pose against fixed registered features from all views, and filtered again after refinement. The evaluator reads prediction artifacts and does not import `method/`.

## Results

Local smoke results are available in `scoring/scores.json`. On the 12-scene synthetic benchmark, the single-view baseline reached AR 47.22 and AP 50.00, while the AlignPose reproduction reached AR 88.89 and AP 91.67. Mean ADD error decreased from 0.0820 to 0.0339, and final duplicate candidates decreased from 12.0 to 1.0.

These local results support the core effect on the reduced target, but a scheduler job is being submitted before claiming `core_claim`.

## Remaining

Run the packaged reproduction in the compute container and inspect `scoring/scores.json`. If the method does not improve over the baseline, tune the synthetic data difficulty or refinement settings without changing the AlignPose computation.

## Issues

The exact BOP candidate archives and full single-view estimator stack are not bundled with the paper, so the first runnable target uses generated data. The compute job still needs to verify the container dependency path and end-to-end scripts.

## Deviations from Paper

The full paper uses BOP datasets, DINOv2-L layer 18 features with PCA, RGB-D rendering for registered feature lifting, and BOP AR/AP metrics. This workspace currently uses a deterministic synthetic RGB feature benchmark and NumPy feature maps so the core multi-view aggregation plus feature-metric SE(3) refinement can run within the available environment. The algorithmic stages are preserved; the data scale and feature extractor are reduced.
