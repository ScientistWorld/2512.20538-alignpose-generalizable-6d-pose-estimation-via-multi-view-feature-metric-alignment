# Scientist Constraints

## Inputs

Use only what a submission receives in `INPUT_DIR`: the RGB images, camera intrinsics and extrinsics, object meshes with their vertex colours, and the per-view single-view candidates. Do not use depth, extra views, external labels or annotated symmetries.

## Objects and training

Test objects are unseen. Mesh-only onboarding at run time is allowed (rendering the supplied meshes, computing statistics from those renders). Training or tuning on the visible train split (`data/train`, labels in `eval/train/train_gt.json`) is allowed; training on, or looking up, test objects or test scenes is not.

## Evaluation code and answers

Do not read, import, copy or modify `eval/test/`, and do not try to locate the hidden scenes' answers or another submission's outputs while running. The evaluator scores only the `detections` you write; any other field is ignored.

## Output

One script per method in `submissions/<name>.sh`, called as `bash submissions/<name>.sh INPUT_DIR OUTPUT_JSON`, finishing within 4 hours on the evaluation node. Every detection must be a valid rotation, a finite translation in metres, an object from the scene's library and a numeric score; at most 100 detections per scene. Otherwise the whole run is scored as failed.

## Code organisation

`eval/` must not import from `method/`; shared data utilities belong in `data/` or the method itself.
