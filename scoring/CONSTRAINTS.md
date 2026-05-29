# Scientist Constraints

## Data

Use `data/synthetic_benchmark.json` for the executable gym target. Do not add
extra labeled scenes, alter ground-truth poses, or tune on hidden labels. The
paper's full BOP datasets are documented as context, but they are not required
for the current runnable target.

## Inputs

Use only RGB images, known camera intrinsics, known camera extrinsics, the
provided object model, and the supplied candidate hypotheses. Do not use depth sensor input,
external point clouds, manually corrected poses, or manually annotated
symmetry metadata.

## View Protocol

Keep the four-view calibrated setup fixed. Do not improve scores by adding
test-time views or changing the camera geometry in the dataset.

## Training and Onboarding

The executable target requires no object-specific training. Mesh-only
precomputation is allowed, but it must not use ground-truth test poses.

## Evaluation

Methods must write prediction artifacts consumed by `scripts/evaluate.sh`.
Evaluation code must remain independent of `method/`.
