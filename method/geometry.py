"""Small SE(3) and projection utilities used by the AlignPose reproduction."""

from __future__ import annotations

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def so3_exp(w):
    theta = float(np.linalg.norm(w))
    wx = skew(w)
    if theta < 1e-10:
        return np.eye(3) + wx
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * wx + b * (wx @ wx)


def se3_left_update(pose, delta):
    r_delta = so3_exp(delta[:3])
    t_delta = delta[3:]
    return {"R": r_delta @ pose["R"], "t": r_delta @ pose["t"] + t_delta}


def compose(a, b):
    return {"R": a["R"] @ b["R"], "t": a["R"] @ b["t"] + a["t"]}


def transform(points, pose):
    return points @ pose["R"].T + pose["t"][None, :]


def project(points_c, intrinsics):
    z = np.maximum(points_c[:, 2], 1e-9)
    u = intrinsics["fx"] * points_c[:, 0] / z + intrinsics["cx"]
    v = intrinsics["fy"] * points_c[:, 1] / z + intrinsics["cy"]
    return np.stack([u, v, points_c[:, 2]], axis=1)


def pose_from_json(p):
    return {"R": np.asarray(p["R"], dtype=np.float64), "t": np.asarray(p["t"], dtype=np.float64)}


def pose_to_json(p):
    return {"R": p["R"].tolist(), "t": p["t"].tolist()}


def rotation_error_deg(a, b):
    rel = a["R"] @ b["R"].T
    c = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def add_error(points, pred, gt):
    return float(np.mean(np.linalg.norm(transform(points, pred) - transform(points, gt), axis=1)))
