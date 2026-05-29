"""Method-agnostic pose metrics for generated prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pose(p):
    return {"R": np.asarray(p["R"], dtype=np.float64), "t": np.asarray(p["t"], dtype=np.float64)}


def transform(points, p):
    return points @ p["R"].T + p["t"][None, :]


def add(points, pred, gt):
    return float(np.mean(np.linalg.norm(transform(points, pred) - transform(points, gt), axis=1)))


def rot_deg(pred, gt):
    rel = pred["R"] @ gt["R"].T
    c = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def evaluate_one(dataset, pred_path):
    pred_data = json.loads(Path(pred_path).read_text())
    metadata = pred_data.get("metadata", {})
    preds = {p["scene_id"]: p for p in pred_data["predictions"]}
    points = np.asarray([p["xyz"] for p in dataset["object_model"]], dtype=np.float64)
    diameter = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    add_errors, rot_errors, trans_errors, nms_counts = [], [], [], []
    for scene in dataset["scenes"]:
        if scene["scene_id"] not in preds:
            continue
        pred = pose(preds[scene["scene_id"]]["T_WO"])
        gt = pose(scene["T_WO_gt"])
        add_errors.append(add(points, pred, gt))
        rot_errors.append(rot_deg(pred, gt))
        trans_errors.append(float(np.linalg.norm(pred["t"] - gt["t"])))
        nms_counts.append(float(preds[scene["scene_id"]].get("num_after_final_nms", 1)))
    add_arr = np.asarray(add_errors, dtype=np.float64)
    thresholds = [0.05 * diameter, 0.10 * diameter, 0.15 * diameter]
    recalls = [float(np.mean(add_arr < th)) for th in thresholds]
    metrics = {
        "AR": 100.0 * float(np.mean(recalls)),
        "AP": 100.0 * float(np.mean(add_arr < 0.10 * diameter)),
        "mean_add_error": float(np.mean(add_arr)),
        "median_translation_error": float(np.median(trans_errors)),
        "median_rotation_error_deg": float(np.median(rot_errors)),
        "mean_final_candidates": float(np.mean(nms_counts)),
    }
    metrics.update({
        "num_views": float(metadata.get("num_views", len(dataset["scenes"][0]["views"]))),
        "uses_depth_sensor": float(bool(metadata.get("uses_depth_sensor", False))),
        "object_specific_training": float(bool(metadata.get("object_specific_training", False))),
        "uses_dinov2_features": float(bool(metadata.get("uses_dinov2_features", False))),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic_benchmark.json")
    parser.add_argument("--baseline", default="outputs/baseline_predictions.json")
    parser.add_argument("--method", default="outputs/alignpose_predictions.json")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text())
    results = {}
    if Path(args.baseline).exists():
        results["single_view_candidate"] = evaluate_one(dataset, args.baseline)
    if Path(args.method).exists():
        results["alignpose"] = evaluate_one(dataset, args.method)
    # The reduced benchmark is a scale substitute for the paper's T-LESS Co-op
    # ablation: one noisy single-view candidate source versus the full
    # aggregation + NMS + feature-metric refinement pipeline.
    payload = {"experiments": {"tless_coop_ablation": {"results": results}}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
