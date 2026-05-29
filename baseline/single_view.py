"""Single-view candidate baseline for the synthetic AlignPose benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_baseline(dataset_path, output_path, max_scenes=None):
    data = json.loads(Path(dataset_path).read_text())
    scenes = data["scenes"][:max_scenes] if max_scenes else data["scenes"]
    out = {"metadata": {"method": "best_single_view_candidate"}, "predictions": []}
    for scene in scenes:
        candidates = []
        for view in scene["views"]:
            candidates.extend(view["candidates"])
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
    parser.add_argument("--dataset", default="data/synthetic_benchmark.json")
    parser.add_argument("--output", default="outputs/baseline_predictions.json")
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    run_baseline(args.dataset, args.output, args.max_scenes)


if __name__ == "__main__":
    main()
