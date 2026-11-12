"""Image-free baselines built only from the supplied single-view candidates.

Modes (mirroring the rows of the paper's component ablation, plus a geometric
multi-view consensus comparable in spirit to CosyPose's multi-view stage with
known cameras):

* ``single_view``      -- top hypothesis of every detection in the first view only.
* ``aggregation``      -- all hypotheses of all views mapped to the world frame, no NMS.
* ``aggregation_nms``  -- the same pool after global 3D NMS (IoU 0.4) on world AABBs.
* ``consensus``        -- NMS seeds; every seed gathers mutually consistent candidates
                          (same object, close translation and rotation), and one world
                          pose is fitted to the 2D reprojections of all inlier
                          candidates in their own views (reprojection bundle fit).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def so3_exp(w):
    th = float(np.linalg.norm(w))
    k = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + k
    k = k / th
    return np.eye(3) + math.sin(th) * k + (1.0 - math.cos(th)) * (k @ k)


def orthonormalize(r):
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vt
    return out


def rot_angle(ra, rb):
    c = (np.trace(ra @ rb.T) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def load_points(input_dir, obj_id, cache):
    if obj_id not in cache:
        raw = json.loads((Path(input_dir) / "objects" / f"{obj_id}.json").read_text())
        v = np.asarray(raw["vertices"], dtype=np.float64)
        sel = np.linspace(0, len(v) - 1, min(600, len(v))).astype(np.int64)
        sub = v[sel]
        diff = sub[:, None, :] - sub[None, :, :]
        cache[obj_id] = (sub, float(np.sqrt((diff ** 2).sum(-1).max())))
    return cache[obj_id]


def world_candidates(scene):
    out = []
    for v in scene["views"]:
        r_cw = np.asarray(v["T_CW"]["R"], dtype=np.float64)
        t_cw = np.asarray(v["T_CW"]["t"], dtype=np.float64)
        r_wc = r_cw.T
        t_wc = -r_wc @ t_cw
        for c in v["candidates"]:
            r_co = np.asarray(c["T_CO"]["R"], dtype=np.float64)
            t_co = np.asarray(c["T_CO"]["t"], dtype=np.float64)
            out.append({"obj_id": c["obj_id"], "det_id": c.get("det_id", -1), "view": v["view_id"],
                        "R": r_wc @ r_co, "t": r_wc @ t_co + t_wc, "score": float(c["score"]),
                        "R_CO": r_co, "t_CO": t_co, "K": np.asarray(v["K"], dtype=np.float64),
                        "R_CW": r_cw, "t_CW": t_cw})
    return out


def aabb(pts, r, t):
    p = pts @ r.T + t[None, :]
    return p.min(0), p.max(0)


def iou(a, b):
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    inter = float(np.prod(np.maximum(0.0, hi - lo)))
    va = float(np.prod(np.maximum(0.0, a[1] - a[0])))
    vb = float(np.prod(np.maximum(0.0, b[1] - b[0])))
    return inter / max(1e-12, va + vb - inter)


def nms(cands, points, thr=0.4):
    order = sorted(range(len(cands)), key=lambda i: (-cands[i]["score"], i))
    boxes = [aabb(points[c["obj_id"]][0], c["R"], c["t"]) for c in cands]
    kept = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) <= thr for k in kept):
            kept.append(i)
    return [cands[i] for i in kept]


def project(pts_w, k, r_cw, t_cw):
    xc = pts_w @ r_cw.T + t_cw[None, :]
    z = np.maximum(xc[:, 2], 1e-6)
    return np.stack([k[0, 0] * xc[:, 0] / z + k[0, 2], k[1, 1] * xc[:, 1] / z + k[1, 2]], 1)


def reprojection_fit(r, t, inliers, pts, iters=15):
    pts = pts[np.linspace(0, len(pts) - 1, min(64, len(pts))).astype(np.int64)]
    targets = []
    for c in inliers:
        obs = project(pts @ c["R_CO"].T + c["t_CO"][None, :], c["K"], np.eye(3), np.zeros(3))
        targets.append((c, obs))

    def residual(rr, tt):
        out = []
        pw = pts @ rr.T + tt[None, :]
        for c, obs in targets:
            out.append((project(pw, c["K"], c["R_CW"], c["t_CW"]) - obs).ravel())
        return np.concatenate(out)

    lam = 1e-3
    res = residual(r, t)
    cost = float(res @ res)
    for _ in range(iters):
        jac = np.zeros((res.size, 6))
        for k in range(6):
            d = np.zeros(6)
            d[k] = 1e-5
            jac[:, k] = (residual(so3_exp(d[:3]) @ r, t + d[3:]) - res) / 1e-5
        h = jac.T @ jac
        g = jac.T @ res
        try:
            step = -np.linalg.solve(h + lam * np.diag(np.diag(h)) + 1e-9 * np.eye(6), g)
        except np.linalg.LinAlgError:
            break
        r2, t2 = so3_exp(step[:3]) @ r, t + step[3:]
        res2 = residual(r2, t2)
        c2 = float(res2 @ res2)
        if c2 < cost:
            r, t, res, cost = r2, t2, res2, c2
            lam = max(lam / 10, 1e-7)
        else:
            lam *= 10
            if lam > 1e6:
                break
    return r, t


def consensus(cands, points, n_views):
    seeds = nms(cands, points)
    out = []
    for s in seeds:
        pts, diam = points[s["obj_id"]]
        pool = [c for c in cands if c["obj_id"] == s["obj_id"]
                and np.linalg.norm(c["t"] - s["t"]) < 0.5 * diam]
        best = None
        for h in pool:
            inl = [c for c in pool if np.linalg.norm(c["t"] - h["t"]) < 0.25 * diam
                   and rot_angle(c["R"], h["R"]) < math.radians(30)]
            support = (len({c["view"] for c in inl}), sum(c["score"] for c in inl))
            if best is None or support > best[0]:
                best = (support, h, inl)
        (n_v, s_sum), hyp, inliers = best
        per_view = {}
        for c in inliers:
            if c["view"] not in per_view or c["score"] > per_view[c["view"]]["score"]:
                per_view[c["view"]] = c
        chosen = list(per_view.values())
        r, t = reprojection_fit(hyp["R"], hyp["t"], chosen, pts)
        score = max(c["score"] for c in chosen) * (0.5 + 0.5 * len(chosen) / n_views)
        out.append({"obj_id": s["obj_id"], "R": r, "t": t, "score": score})
    return nms(out, points)


def run(mode, input_dir, output):
    input_dir = Path(input_dir)
    index = json.loads((input_dir / "index.json").read_text())
    cache = {}
    dets = []
    for sid in index["scenes"]:
        scene = json.loads((input_dir / "scenes" / sid / "scene.json").read_text())
        cands = world_candidates(scene)
        points = {c["obj_id"]: load_points(input_dir, c["obj_id"], cache) for c in cands}
        if mode == "single_view":
            first = scene["views"][0]["view_id"]
            best = {}
            for c in cands:
                key = (c["view"], c["det_id"])
                if c["view"] == first and (key not in best or c["score"] > best[key]["score"]):
                    best[key] = c
            chosen = list(best.values())
        elif mode == "aggregation":
            chosen = cands
        elif mode == "aggregation_nms":
            chosen = nms(cands, points)
        elif mode == "consensus":
            chosen = consensus(cands, points, len(scene["views"]))
        else:
            raise ValueError(mode)
        for c in chosen:
            dets.append({"scene_id": sid, "obj_id": c["obj_id"], "R": orthonormalize(c["R"]).tolist(),
                         "t": np.asarray(c["t"]).tolist(), "score": float(c["score"])})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"method": mode, "detections": dets}))
    print(f"{mode}: wrote {len(dets)} detections for {len(index['scenes'])} scenes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["single_view", "aggregation", "aggregation_nms", "consensus"])
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.mode, a.input_dir, a.output)


if __name__ == "__main__":
    main()
