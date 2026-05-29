"""AlignPose reproduction on the synthetic multi-view benchmark.

This implements the paper's core computational stages at reduced scale:
single-view candidates are transformed into a shared world frame, filtered by
3D NMS, then independently refined by minimizing a multi-view feature-metric
loss over one world-frame SE(3) pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from method.geometry import compose, pose_from_json, pose_to_json, project, se3_left_update, transform


def load_dataset(path):
    data = json.loads(Path(path).read_text())
    model_points = np.asarray([p["xyz"] for p in data["object_model"]], dtype=np.float64)
    model_features = np.asarray([p["feature"] for p in data["object_model"]], dtype=np.float64)
    return data, model_points, model_features


def feature_image(image):
    arr = np.asarray(image, dtype=np.float64)
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    gx = np.zeros((h, w, 1), dtype=np.float64)
    gy = np.zeros((h, w, 1), dtype=np.float64)
    gray = arr.mean(axis=2, keepdims=True)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    pos = np.stack([xx / max(1, w - 1), yy / max(1, h - 1)], axis=2)
    return np.concatenate([arr, gx, gy, 0.15 * pos], axis=2)


def sample_bilinear(fmap, uv):
    h, w, c = fmap.shape
    u = uv[:, 0]
    v = uv[:, 1]
    valid = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    u0 = np.floor(np.clip(u, 0, w - 1)).astype(int)
    v0 = np.floor(np.clip(v, 0, h - 1)).astype(int)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    du = (np.clip(u, 0, w - 1) - u0)[:, None]
    dv = (np.clip(v, 0, h - 1) - v0)[:, None]
    top = fmap[v0, u0] * (1 - du) + fmap[v0, u1] * du
    bot = fmap[v1, u0] * (1 - du) + fmap[v1, u1] * du
    val = top * (1 - dv) + bot * dv
    val[~valid] = 0.0
    return val, valid


def registered_features(model_points, model_features, coarse_t_wo, t_cw, intrinsics, image_size, max_points):
    t_co = compose(t_cw, coarse_t_wo)
    pts_c = transform(model_points, t_co)
    uvz = project(pts_c, intrinsics)
    valid = (uvz[:, 2] > 0) & (uvz[:, 0] >= 1) & (uvz[:, 0] < image_size - 2) & (uvz[:, 1] >= 1) & (uvz[:, 1] < image_size - 2)
    idx = np.nonzero(valid)[0]
    if len(idx) > max_points:
        idx = idx[np.linspace(0, len(idx) - 1, max_points).astype(int)]
    base = model_features[idx]
    # Fixed descriptors mirror the query feature channels: RGB, gradients, and weak position channels.
    zeros = np.zeros((len(idx), 2), dtype=np.float64)
    uv_norm = np.stack([uvz[idx, 0] / (image_size - 1), uvz[idx, 1] / (image_size - 1)], axis=1) * 0.15
    desc = np.concatenate([base, zeros, uv_norm], axis=1)
    return model_points[idx], desc


def barron_weights(residual, alpha=-5.0, c=0.5):
    norm = np.linalg.norm(residual, axis=1)
    # Robust down-weighting matching the paper's intent: large feature residuals
    # contribute less strongly than in L2.
    return (1.0 + (norm / c) ** 2 / max(1e-6, abs(alpha - 2.0))) ** (alpha / 2.0 - 1.0)


def residual_vector(pose, view_reps, intrinsics):
    chunks = []
    losses = []
    for rep in view_reps:
        t_co = compose(rep["T_CW"], pose)
        pts_c = transform(rep["points"], t_co)
        uvz = project(pts_c, intrinsics)
        sampled, valid = sample_bilinear(rep["query"], uvz[:, :2])
        diff = rep["features"] - sampled
        diff[~valid] = 1.0
        weights = np.sqrt(barron_weights(diff))[:, None]
        chunks.append((diff * weights).reshape(-1))
        losses.append(float(np.mean(np.linalg.norm(diff, axis=1))))
    return np.concatenate(chunks), float(np.mean(losses))


def lm_refine(initial_pose, view_reps, intrinsics, max_iters=30):
    pose = {"R": initial_pose["R"].copy(), "t": initial_pose["t"].copy()}
    damping = 1e-3
    eps = 1e-4
    residual, loss = residual_vector(pose, view_reps, intrinsics)
    for _ in range(max_iters):
        jac = np.zeros((residual.size, 6), dtype=np.float64)
        for k in range(6):
            delta = np.zeros(6, dtype=np.float64)
            delta[k] = eps
            r2, _ = residual_vector(se3_left_update(pose, delta), view_reps, intrinsics)
            jac[:, k] = (r2 - residual) / eps
        lhs = jac.T @ jac + damping * np.eye(6)
        rhs = -(jac.T @ residual)
        try:
            step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(step) < 1e-5:
            break
        proposal = se3_left_update(pose, step)
        new_residual, new_loss = residual_vector(proposal, view_reps, intrinsics)
        if new_loss < loss:
            pose, residual, loss = proposal, new_residual, new_loss
            damping = max(1e-6, damping * 0.5)
        else:
            damping = min(1e2, damping * 4.0)
    return pose, loss


def aabb_for_pose(points, pose):
    p = transform(points, pose)
    return p.min(axis=0), p.max(axis=0)


def iou_aabb(a, b):
    amin, amax = a
    bmin, bmax = b
    inter_min = np.maximum(amin, bmin)
    inter_max = np.minimum(amax, bmax)
    inter = np.prod(np.maximum(0.0, inter_max - inter_min))
    av = np.prod(np.maximum(0.0, amax - amin))
    bv = np.prod(np.maximum(0.0, bmax - bmin))
    return float(inter / max(1e-12, av + bv - inter))


def nms(candidates, model_points, threshold=0.4):
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []
    boxes = []
    for cand in ordered:
        box = aabb_for_pose(model_points, cand["pose"])
        if all(iou_aabb(box, prev) <= threshold for prev in boxes):
            kept.append(cand)
            boxes.append(box)
    return kept


def run_alignpose(dataset_path, output_path, max_scenes=None, max_points=90):
    data, model_points, model_features = load_dataset(dataset_path)
    intrinsics = data["intrinsics"]
    image_size = data["image_size"]
    results = {"metadata": {"method": "alignpose_synthetic", "nms_iou": 0.4, "max_points_per_view": max_points}, "predictions": []}
    scenes = data["scenes"][:max_scenes] if max_scenes else data["scenes"]
    for scene in scenes:
        candidates = []
        for view in scene["views"]:
            for cand in view["candidates"]:
                candidates.append({"pose": pose_from_json(cand["T_WO"]), "score": float(cand["score"]), "source_view": cand["view_id"]})
        aggregated = nms(candidates, model_points, threshold=0.4)
        refined = []
        for cand in aggregated:
            reps = []
            for view in scene["views"]:
                t_cw = pose_from_json(view["T_CW"])
                pts, desc = registered_features(model_points, model_features, cand["pose"], t_cw, intrinsics, image_size, max_points)
                reps.append({"T_CW": t_cw, "query": feature_image(view["image"]), "points": pts, "features": desc})
            pose, loss = lm_refine(cand["pose"], reps, intrinsics, max_iters=30)
            refined.append({"pose": pose, "score": 1.0 - loss, "feature_loss": loss, "source_view": cand["source_view"]})
        refined = nms(refined, model_points, threshold=0.4)
        best = max(refined, key=lambda c: c["score"])
        results["predictions"].append({
            "scene_id": scene["scene_id"],
            "T_WO": pose_to_json(best["pose"]),
            "score": best["score"],
            "num_initial_candidates": len(candidates),
            "num_after_first_nms": len(aggregated),
            "num_after_final_nms": len(refined),
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic_benchmark.json")
    parser.add_argument("--output", default="outputs/alignpose_predictions.json")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-points", type=int, default=90)
    args = parser.parse_args()
    run_alignpose(args.dataset, args.output, args.max_scenes, args.max_points)


if __name__ == "__main__":
    main()
