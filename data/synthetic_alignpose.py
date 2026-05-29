"""Synthetic multi-view 6D pose benchmark for AlignPose-style refinement.

The generated scenes are deliberately small, but they preserve the problem
structure required by the paper: calibrated RGB views, a known asymmetric 3D
object, noisy per-view pose candidates, and ground-truth world poses.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def euler_matrix(rx, ry, rz):
    return _matmul(_rot_z(rz), _matmul(_rot_y(ry), _rot_x(rx)))


def transform_points(points, r, t):
    out = []
    for x, y, z in points:
        out.append([
            r[0][0] * x + r[0][1] * y + r[0][2] * z + t[0],
            r[1][0] * x + r[1][1] * y + r[1][2] * z + t[1],
            r[2][0] * x + r[2][1] * y + r[2][2] * z + t[2],
        ])
    return out


def make_object_model(samples_per_face=9):
    """Create an asymmetric cuboid-plus-nub model with deterministic surface colors."""
    points = []
    dims = [(0.55, 0.35, 0.28, [0.80, 0.18, 0.12]), (0.18, 0.18, 0.18, [0.12, 0.55, 0.90])]
    offsets = [(0.0, 0.0, 0.0), (0.25, 0.16, 0.18)]
    for (sx, sy, sz, base), (ox, oy, oz) in zip(dims, offsets):
        xs = [(-0.5 + i / (samples_per_face - 1)) * sx + ox for i in range(samples_per_face)]
        ys = [(-0.5 + i / (samples_per_face - 1)) * sy + oy for i in range(samples_per_face)]
        zs = [(-0.5 + i / (samples_per_face - 1)) * sz + oz for i in range(samples_per_face)]
        faces = []
        for x in (xs[0], xs[-1]):
            faces += [(x, y, z) for y in ys for z in zs]
        for y in (ys[0], ys[-1]):
            faces += [(x, y, z) for x in xs for z in zs]
        for z in (zs[0], zs[-1]):
            faces += [(x, y, z) for x in xs for y in ys]
        for x, y, z in faces:
            color = [
                min(1.0, max(0.0, base[0] + 0.18 * math.sin(11 * x + 3 * z))),
                min(1.0, max(0.0, base[1] + 0.18 * math.cos(7 * y - 2 * x))),
                min(1.0, max(0.0, base[2] + 0.18 * math.sin(5 * z + 4 * y))),
            ]
            points.append({"xyz": [x, y, z], "feature": color})
    return points


def look_at(camera_position, target=(0.0, 0.0, 0.0)):
    cx, cy, cz = camera_position
    tx, ty, tz = target
    z_axis = _normalize([tx - cx, ty - cy, tz - cz])
    x_axis = _normalize([z_axis[1], -z_axis[0], 0.0])
    y_axis = [
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    ]
    r_cw = [x_axis, y_axis, z_axis]
    t_cw = [-sum(r_cw[i][j] * camera_position[j] for j in range(3)) for i in range(3)]
    return r_cw, t_cw


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) + 1e-12
    return [x / n for x in v]


def _compose(a, b):
    ra, ta = a["R"], a["t"]
    rb, tb = b["R"], b["t"]
    r = _matmul(ra, rb)
    t = [sum(ra[i][j] * tb[j] for j in range(3)) + ta[i] for i in range(3)]
    return {"R": r, "t": t}


def _invert(pose):
    r = pose["R"]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    t = [-sum(rt[i][j] * pose["t"][j] for j in range(3)) for i in range(3)]
    return {"R": rt, "t": t}


def _perturb_pose(pose, rng, rot_sigma, trans_sigma):
    dr = euler_matrix(rng.gauss(0, rot_sigma), rng.gauss(0, rot_sigma), rng.gauss(0, rot_sigma))
    dt = [rng.gauss(0, trans_sigma) for _ in range(3)]
    return {"R": _matmul(dr, pose["R"]), "t": [pose["t"][i] + dt[i] for i in range(3)]}


def _project(point, intrinsics):
    x, y, z = point
    if z <= 1e-6:
        return None
    return [
        intrinsics["fx"] * x / z + intrinsics["cx"],
        intrinsics["fy"] * y / z + intrinsics["cy"],
        z,
    ]


def render_points(model, t_co, intrinsics, size):
    image = [[[0.03, 0.035, 0.04] for _ in range(size)] for _ in range(size)]
    depth = [[1e9 for _ in range(size)] for _ in range(size)]
    pts = [p["xyz"] for p in model]
    transformed = transform_points(pts, t_co["R"], t_co["t"])
    for p_model, p_cam in zip(model, transformed):
        proj = _project(p_cam, intrinsics)
        if proj is None:
            continue
        u, v, z = proj
        ui, vi = int(round(u)), int(round(v))
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                x, y = ui + du, vi + dv
                if 0 <= x < size and 0 <= y < size and z < depth[y][x]:
                    depth[y][x] = z
                    image[y][x] = p_model["feature"]
    return image


def generate_dataset(output, n_scenes=24, n_views=4, seed=251220538):
    rng = random.Random(seed)
    size = 96
    intrinsics = {"fx": 82.0, "fy": 82.0, "cx": size / 2, "cy": size / 2}
    model = make_object_model()
    scenes = []
    for sid in range(n_scenes):
        gt = {
            "R": euler_matrix(rng.uniform(-0.4, 0.4), rng.uniform(-0.7, 0.7), rng.uniform(-math.pi, math.pi)),
            "t": [rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04)],
        }
        views = []
        for vid in range(n_views):
            angle = 2 * math.pi * vid / n_views + 0.25 * (sid % 3)
            cam_pos = [1.45 * math.cos(angle), 1.45 * math.sin(angle), 0.62 + 0.08 * ((sid + vid) % 2)]
            r_cw, t_cw = look_at(cam_pos)
            t_cw_pose = {"R": r_cw, "t": t_cw}
            t_co = _compose(t_cw_pose, gt)
            image = render_points(model, t_co, intrinsics, size)
            candidates = []
            for hyp in range(3):
                noisy = _perturb_pose(t_co, rng, rot_sigma=0.12 + 0.06 * hyp, trans_sigma=0.035 + 0.015 * hyp)
                world = _compose(_invert(t_cw_pose), noisy)
                candidates.append({
                    "T_CO": noisy,
                    "T_WO": world,
                    "score": max(0.05, 1.0 - 0.22 * hyp + rng.gauss(0, 0.04)),
                    "view_id": vid,
                })
            views.append({"T_CW": t_cw_pose, "image": image, "candidates": candidates})
        scenes.append({"scene_id": sid, "T_WO_gt": gt, "views": views})
    payload = {"image_size": size, "intrinsics": intrinsics, "object_model": model, "scenes": scenes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_benchmark.json")
    parser.add_argument("--scenes", type=int, default=24)
    parser.add_argument("--views", type=int, default=4)
    args = parser.parse_args()
    generate_dataset(Path(args.output), args.scenes, args.views)


if __name__ == "__main__":
    main()
