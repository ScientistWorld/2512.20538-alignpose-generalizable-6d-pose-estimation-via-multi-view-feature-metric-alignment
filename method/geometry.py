"""SE(3) helpers, input loading and 3D NMS shared by the AlignPose reproduction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(w):
    w = np.asarray(w, dtype=np.float64)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / th)
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
    return float(math.acos(max(-1.0, min(1.0, c))))


class Mesh:
    def __init__(self, obj_id, vertices, faces, colors):
        self.obj_id = obj_id
        self.V = np.asarray(vertices, dtype=np.float64)
        self.F = np.asarray(faces, dtype=np.int64)
        self.C = np.asarray(colors, dtype=np.float64)
        e1 = self.V[self.F[:, 1]] - self.V[self.F[:, 0]]
        e2 = self.V[self.F[:, 2]] - self.V[self.F[:, 0]]
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        self.face_normals = n / np.maximum(norm, 1e-12)
        sel = np.linspace(0, len(self.V) - 1, min(600, len(self.V))).astype(np.int64)
        self.V_sub = self.V[sel]
        diff = self.V_sub[:, None, :] - self.V_sub[None, :, :]
        self.diameter = float(np.sqrt((diff ** 2).sum(-1).max()))


def load_index(input_dir):
    return json.loads((Path(input_dir) / "index.json").read_text())


def load_mesh(input_dir, obj_id):
    raw = json.loads((Path(input_dir) / "objects" / f"{obj_id}.json").read_text())
    return Mesh(obj_id, raw["vertices"], raw["faces"], raw["vertex_colors"])


def load_scene(input_dir, scene_id):
    sdir = Path(input_dir) / "scenes" / scene_id
    raw = json.loads((sdir / "scene.json").read_text())
    views = []
    for v in raw["views"]:
        img = np.asarray(Image.open(sdir / v["image"]).convert("RGB"), dtype=np.float32) / 255.0
        cands = [{"obj_id": c["obj_id"], "det_id": int(c.get("det_id", -1)),
                  "R": np.asarray(c["T_CO"]["R"], dtype=np.float64), "t": np.asarray(c["T_CO"]["t"], dtype=np.float64),
                  "score": float(c["score"])} for c in v["candidates"]]
        views.append({"view_id": v["view_id"], "image": img, "K": np.asarray(v["K"], dtype=np.float64),
                      "R_CW": np.asarray(v["T_CW"]["R"], dtype=np.float64),
                      "t_CW": np.asarray(v["T_CW"]["t"], dtype=np.float64), "candidates": cands})
    return {"scene_id": raw["scene_id"], "dataset": raw.get("dataset"), "views": views,
            "library": list(raw.get("object_library", []))}


def world_candidates(scene):
    """Transform every per-view candidate into the shared world frame (T_WO = T_WC T_CO)."""
    out = []
    for v in scene["views"]:
        r_wc = v["R_CW"].T
        t_wc = -r_wc @ v["t_CW"]
        for c in v["candidates"]:
            out.append({"obj_id": c["obj_id"], "R": r_wc @ c["R"], "t": r_wc @ c["t"] + t_wc,
                        "score": c["score"], "view_id": v["view_id"], "det_id": c["det_id"]})
    return out


def aabb(mesh, r, t):
    p = mesh.V_sub @ r.T + t[None, :]
    return p.min(0), p.max(0)


def iou_aabb(a, b):
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    inter = float(np.prod(np.maximum(0.0, hi - lo)))
    va = float(np.prod(np.maximum(0.0, a[1] - a[0])))
    vb = float(np.prod(np.maximum(0.0, b[1] - b[0])))
    return inter / max(1e-12, va + vb - inter)


def nms(cands, meshes, threshold=0.4):
    """Global (inter-class) 3D NMS on world axis-aligned boxes.

    Returns the kept candidates (highest score first) and, for each kept
    candidate, the list of candidates it suppressed (itself first).
    """
    order = sorted(range(len(cands)), key=lambda i: (-cands[i]["score"], i))
    boxes = [aabb(meshes[c["obj_id"]], c["R"], c["t"]) for c in cands]
    kept, clusters = [], []
    for i in order:
        hit = None
        for k_idx, k in enumerate(kept):
            if iou_aabb(boxes[i], boxes[k]) > threshold:
                hit = k_idx
                break
        if hit is None:
            kept.append(i)
            clusters.append([i])
        else:
            clusters[hit].append(i)
    return [cands[i] for i in kept], [[cands[i] for i in cl] for cl in clusters]


def detection_json(scene_id, obj_id, r, t, score):
    return {"scene_id": scene_id, "obj_id": obj_id, "R": orthonormalize(r).tolist(),
            "t": np.asarray(t, dtype=np.float64).tolist(), "score": float(score)}
