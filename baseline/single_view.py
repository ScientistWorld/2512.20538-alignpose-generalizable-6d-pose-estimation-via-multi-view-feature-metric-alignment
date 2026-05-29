"""Single-view candidate baseline for the synthetic AlignPose benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pose_from_json(raw):
    return {"R": np.asarray(raw["R"], dtype=float), "t": np.asarray(raw["t"], dtype=float)}


def pose_to_json(pose):
    return {"R": pose["R"].tolist(), "t": pose["t"].tolist()}


def compose(a, b):
    return {"R": a["R"] @ b["R"], "t": a["R"] @ b["t"] + a["t"]}


def invert(pose):
    r_inv = pose["R"].T
    return {"R": r_inv, "t": -(r_inv @ pose["t"])}


def candidate_world_pose(candidate, view_pose):
    if "T_CO" in candidate:
        return compose(invert(view_pose), pose_from_json(candidate["T_CO"]))
    return pose_from_json(candidate["T_WO"])


def run_baseline(dataset_path, output_path, max_scenes=None):
    data = json.loads(Path(dataset_path).read_text())
    scenes = data["scenes"][:max_scenes] if max_scenes else data["scenes"]
    n_views = len(data["scenes"][0]["views"]) if data["scenes"] else 0
    out = {
        "metadata": {
            "method": "best_single_view_candidate",
            "num_views": n_views,
            "uses_depth_sensor": False,
            "object_specific_training": False,
            "uses_dinov2_features": False,
        },
        "predictions": [],
    }
    for scene in scenes:
        candidates = []
        for view in scene["views"]:
            t_cw = pose_from_json(view["T_CW"])
            for candidate in view["candidates"]:
                converted = dict(candidate)
                converted["T_WO"] = pose_to_json(candidate_world_pose(candidate, t_cw))
                candidates.append(converted)
        best = max(candidates, key=lambda c: c["score"])
        out["predictions"].append({
            "scene_id": scene["scene_id"],
            "T_WO": best["T_WO"],
            "score": best["score"],
            "num_initial_candidates": len(candidates),
            "num_after_first_nms": len(candidates),
            "num_after_final_nms": len(candidates),
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic_inputs.json")
    parser.add_argument("--output", default="outputs/baseline_predictions.json")
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    run_baseline(args.dataset, args.output, args.max_scenes)


if __name__ == "__main__":
    main()
