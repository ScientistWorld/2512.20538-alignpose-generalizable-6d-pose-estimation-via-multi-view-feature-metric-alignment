# Data

## Visible train split: `data/train/`

Generated once and tracked in git. Its labels live in `eval/train/train_gt.json`.
The hidden test split uses the same format, the same three scene regimes and a
**disjoint object library**; it is generated only at test time and is never on
disk beforehand.

```
data/train/
  index.json                       # {"scenes": [...], "objects": [...], "datasets": {regime: {"scenes", "objects"}}}
  objects/<obj_id>.json            # object model given to the solver
  scenes/<scene_id>/scene.json     # calibrated views + single-view candidates
  scenes/<scene_id>/view_<k>.jpg   # RGB image of view k (256 x 256)
```

### Object model (`objects/<obj_id>.json`)

| key | meaning |
|---|---|
| `vertices` | N x 3 vertex positions in the object frame, metres |
| `faces` | M x 3 vertex indices (triangles) |
| `vertex_colors` | N x 3 RGB in [0, 1] (the model's texture; it approximates, but is not identical to, the observed appearance) |

### Scene (`scenes/<scene_id>/scene.json`)

| key | meaning |
|---|---|
| `dataset` | regime: `textureless_clutter`, `textured_household` or `reflective_industrial` |
| `object_library` | object ids that may appear in this scene (each appears at most once per scene) |
| `views[k].image` | image file name |
| `views[k].K` | 3 x 3 intrinsics; pixel (row r, col c) has centre (c + 0.5, r + 0.5) |
| `views[k].T_CW` | `{"R", "t"}` world-to-camera extrinsics (`x_cam = R x_world + t`) |
| `views[k].candidates` | single-view pose candidates for this view: `obj_id`, `det_id` (2D detection the hypothesis belongs to), `T_CO` (`{"R","t"}`, object-to-camera), `score` |

Candidates come from a simulated single-view estimator: they contain depth and
rotation errors, occasional gross orientation errors, missed detections and
false positives (wrong object ids, spurious detections).

## Output format (all splits)

A submission is an executable `submissions/<name>.sh` called as
`bash submissions/<name>.sh INPUT_DIR OUTPUT_JSON`. It must write

```json
{"detections": [{"scene_id": "...", "obj_id": "...", "R": [[...],[...],[...]], "t": [x, y, z], "score": 0.9}]}
```

with world-frame object poses (`x_world = R x_obj + t`, metres). See
`briefing/evaluation.md` for validity rules and metrics.

## Downloads: `data/downloads/`

Not tracked; populated by `scripts/download.sh` (frozen DINOv2-S weights used by
the reference method).
