"""Visible train-slice scorer for pose prediction artifacts.

This evaluator is method-agnostic: it reads generated prediction JSON files
from an outputs directory and never imports method code.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


EXPERIMENT_NAME = "tless_coop_ablation"
SLICE_NAME = "train"
KNOWN_METHOD_NAMES = {
    "baseline_predictions": "single_view_candidate",
    "alignpose_predictions": "alignpose",
}


def pose(raw):
    return {
        "R": np.asarray(raw["R"], dtype=np.float64),
        "t": np.asarray(raw["t"], dtype=np.float64),
    }


def transform(points, transform_pose):
    return points @ transform_pose["R"].T + transform_pose["t"][None, :]


def add_error(points, pred, gt):
    return float(np.mean(np.linalg.norm(transform(points, pred) - transform(points, gt), axis=1)))


def best_candidate_add_error(points, scene):
    gt = pose(scene["T_WO_gt"])
    errors = []
    for view in scene["views"]:
        for candidate in view["candidates"]:
            errors.append(add_error(points, pose(candidate["T_WO"]), gt))
    if not errors:
        return None
    return min(errors)


def rotation_error_deg(pred, gt):
    rel = pred["R"] @ gt["R"].T
    cosine = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def select_scenes(dataset):
    scenes = sorted(dataset["scenes"], key=lambda scene: scene["scene_id"])
    selected = [scene for index, scene in enumerate(scenes) if index % 2 == 0]
    if not selected:
        raise ValueError("train slice is empty; dataset must contain at least one scene")
    return selected


def method_name_for(path):
    stem = path.stem
    if stem in KNOWN_METHOD_NAMES:
        return KNOWN_METHOD_NAMES[stem]
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    if not name:
        raise ValueError(f"could not infer method name from {path}")
    return name


def discover_prediction_files(outputs_dir):
    prediction_files = sorted(Path(outputs_dir).glob("*_predictions.json"))
    methods = {}
    for path in prediction_files:
        name = method_name_for(path)
        if name in methods:
            raise ValueError(f"multiple prediction artifacts map to method '{name}'")
        methods[name] = path
    if not methods:
        raise FileNotFoundError(f"no prediction artifacts found under {outputs_dir}")
    return methods


def evaluate_one(dataset, scenes, pred_path):
    pred_data = json.loads(Path(pred_path).read_text())
    metadata = pred_data.get("metadata", {})
    preds = {entry["scene_id"]: entry for entry in pred_data["predictions"]}
    points = np.asarray([point["xyz"] for point in dataset["object_model"]], dtype=np.float64)
    diameter = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    add_errors = []
    rotation_errors = []
    translation_errors = []
    nms_counts = []
    candidate_gains = []
    exact_copies = []
    complete_count = 0
    for scene in scenes:
        gt = pose(scene["T_WO_gt"])
        candidate_error = best_candidate_add_error(points, scene)
        pred_entry = preds.get(scene["scene_id"])
        if pred_entry is None:
            add_errors.append(diameter)
            rotation_errors.append(180.0)
            translation_errors.append(diameter)
            nms_counts.append(0.0)
            if candidate_error is not None:
                candidate_gains.append(100.0 * (candidate_error - diameter) / diameter)
            exact_copies.append(0.0)
            continue
        pred = pose(pred_entry["T_WO"])
        scene_add_error = add_error(points, pred, gt)
        add_errors.append(scene_add_error)
        rotation_errors.append(rotation_error_deg(pred, gt))
        translation_errors.append(float(np.linalg.norm(pred["t"] - gt["t"])))
        nms_counts.append(float(pred_entry.get("num_after_final_nms", 1)))
        if candidate_error is not None:
            candidate_gains.append(100.0 * (candidate_error - scene_add_error) / diameter)
        exact_copies.append(float(scene_add_error <= 1e-12))
        complete_count += 1

    if complete_count == 0:
        raise ValueError(f"{pred_path} has no predictions for the {SLICE_NAME} slice")

    add_arr = np.asarray(add_errors, dtype=np.float64)
    thresholds = [0.05 * diameter, 0.10 * diameter, 0.15 * diameter]
    recalls = [float(np.mean(add_arr < threshold)) for threshold in thresholds]
    return {
        "AR": 100.0 * float(np.mean(recalls)),
        "AP": 100.0 * float(np.mean(add_arr < 0.10 * diameter)),
        "complete_prediction_rate": 100.0 * complete_count / len(scenes),
        "candidate_add_gain_pct": float(np.mean(candidate_gains)) if candidate_gains else 0.0,
        "exact_pose_copy_rate": 100.0 * float(np.mean(exact_copies)),
        "mean_add_error": float(np.mean(add_arr)),
        "median_translation_error": float(np.median(translation_errors)),
        "median_rotation_error_deg": float(np.median(rotation_errors)),
        "mean_final_candidates": float(np.mean(nms_counts)),
        "num_views": float(metadata.get("num_views", len(dataset["scenes"][0]["views"]))),
        "uses_depth_sensor": float(bool(metadata.get("uses_depth_sensor", False))),
        "object_specific_training": float(bool(metadata.get("object_specific_training", False))),
        "uses_dinov2_features": float(bool(metadata.get("uses_dinov2_features", False))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic_benchmark.json")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text())
    scenes = select_scenes(dataset)
    methods = discover_prediction_files(args.outputs_dir)
    results = {
        method_name: evaluate_one(dataset, scenes, pred_path)
        for method_name, pred_path in sorted(methods.items())
    }

    payload = {"experiments": {EXPERIMENT_NAME: {"results": results}}}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
