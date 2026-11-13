# Evaluation

This file describes the task metrics and the protocol only.

## Interface

A method is an executable script `submissions/<name>.sh`, called as

```
bash submissions/<name>.sh INPUT_DIR OUTPUT_JSON
```

`INPUT_DIR` has the layout described in `data/README.md` (images, calibration, per-view candidates, object meshes; no labels). The script writes

```json
{"detections": [{"scene_id": "...", "obj_id": "...", "R": [[...],[...],[...]], "t": [x, y, z], "score": 0.9}]}
```

with world-frame object poses (`x_world = R x_obj + t`, metres) and a confidence score used for ranking.

- `scripts/evaluate_train.sh` runs every `submissions/*.sh` on a fresh copy of the visible train split (`data/train/`) and writes `scoring/scores_train.json`.
- `scripts/evaluate_test.sh` (no arguments) generates the hidden test scenes at test time, runs every `submissions/*.sh` on its own fresh copy of them, and writes `scoring/scores_test.json`. The hidden test uses new scenes and a new object library in the same three regimes and the same format. The test's code and answers are not readable while submissions run, and a run may take at most 4 hours.

## Validity

The whole run is scored as failed, AR = AP = 0 in every experiment, if the script exits non-zero or times out, writes no file or invalid JSON, or any detection has: an unknown scene id, an object id outside the scene's object library, a non-finite number, a rotation matrix that is not orthonormal with determinant 1 (tolerance 1e-3), a translation above 1000 m, or a non-numeric score; or if a scene has more than 100 detections. Nothing besides `detections` is read.

## Metrics

Evaluation follows the BOP protocol used by the paper, **per image**: every scene detection is an estimate in each of the scene's four images (its pose expressed in that camera). A ground-truth instance is an (image, object) pair whose visible fraction in that image is at least 0.1; estimates of an object that is present but not visible in an image are ignored there. The objects are asymmetric.

- **MSSD**: maximum distance between model points under the estimated and true pose, relative to the object diameter. Thresholds 0.05, 0.10, ..., 0.50.
- **MSPD**: maximum distance between projected model points in the image, in pixels. Thresholds 5, 10, ..., 50 px at 640 px image width, scaled to the actual width (2–20 px at 256 px).
- **AR** (localization, higher is better): for each ground-truth instance the highest-scoring estimate of that object in the scene is evaluated; recall is averaged over the 10 thresholds of each error, over instances, and AR = (AR_MSSD + AR_MSPD) / 2. A missed object scores 0. The paper also averages a VSD term, which is not computed here.
- **AP** (detection, higher is better): BOP/COCO-style 101-point interpolated average precision. All estimates of an object are ranked by score; the top estimate of an object in a scene is a true positive in an image if its error is below the threshold, and every other estimate (duplicates, wrong objects, objects not in the scene) is a false positive. AP is averaged over the 10 thresholds, over objects, and AP = (AP_MSSD + AP_MSPD) / 2. AP therefore penalizes duplicate and spurious hypotheses and badly calibrated confidences, which AR does not.

Each regime (`textureless_clutter`, `textured_household`, `reflective_industrial`) is a separate experiment with AR and AP; weights and coefficients are in `scoring/reference.json`.

## Baselines

The baselines in `submissions/` use only the supplied candidates, no images:

- `single_view_candidate`: the top hypothesis of every detection in the first view only.
- `multiview_aggregation`: every candidate of every view, mapped to the world frame.
- `multiview_aggregation_nms`: the same pool after removing overlapping duplicates in 3D.
- `multiview_consensus`: a geometric multi-view fusion of mutually consistent candidates (a reprojection fit across views), comparable in spirit to the published multi-view refinement the paper compares against.

`alignpose` is the reproduced reference method. The paper's real-data numbers are in `scoring/reference.json` under `_paper_*` keys for context; they are not on this benchmark's scale.
