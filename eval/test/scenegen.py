"""Hidden multi-view scene generator for the held-out test split.

Everything that can regenerate the test answers lives in this file: the
procedural object library, the scene layout, the photometric image formation
(shading, shadows, clutter, sensor noise) and the single-view candidate
simulator.  The test evaluator calls :func:`generate_split` at test time with
the hidden test seed, so test images, candidates and ground truth never exist
on disk before the evaluation job starts.

The same code was used once, offline and with a different seed and a disjoint
object library, to produce the visible train split under ``data/train`` and
its labels in ``eval/train/train_gt.json`` (see ``--make-train``).
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 256
BASE_SIZE = 256          # focal length and pixel thresholds are defined at this size and scaled
BASE_FOCAL = 280.0
SUPERSAMPLE = 2
N_VIEWS = 4
MESH_SPACING = 0.015
EVAL_VERTICES = 1500
VISIB_MIN = 0.1
JPEG_QUALITY = 90
OBJ_BIAS_FRACTION = 0.6

DATASETS = ("textureless_clutter", "textured_household", "reflective_industrial")
SHORT = {"textureless_clutter": "tl", "textured_household": "tx", "reflective_industrial": "ri"}

TEST_SEED = 912_377_455_061
TEST_SCENES_PER_DATASET = 60
TEST_OBJECTS_PER_DATASET = 8
TRAIN_SEED = 20_251_220
TRAIN_SCENES_PER_DATASET = 24
TRAIN_OBJECTS_PER_DATASET = 8

CONFIG = {
    "textureless_clutter": dict(
        appearance="plain", grayscale=False, n_targets=3, n_distractors=(3, 5), plane="cloth",
        sigma_2d=0.03, sigma_depth_view=0.03, sigma_depth_det=0.04, sigma_rot_deg=6.0, p_gross=0.2,
        det_base=0.15, det_max=0.92, fp_rate=1.2, n_hyp=3, light=(0.55, 0.85), ambient=(0.25, 0.40),
    ),
    "textured_household": dict(
        appearance="rich", grayscale=False, n_targets=3, n_distractors=(1, 3), plane="wood",
        sigma_2d=0.025, sigma_depth_view=0.025, sigma_depth_det=0.035, sigma_rot_deg=4.5, p_gross=0.12,
        det_base=0.25, det_max=0.92, fp_rate=0.7, n_hyp=3, light=(0.50, 0.80), ambient=(0.30, 0.45),
    ),
    "reflective_industrial": dict(
        appearance="metal", grayscale=True, n_targets=3, n_distractors=(2, 4), plane="dark",
        sigma_2d=0.035, sigma_depth_view=0.035, sigma_depth_det=0.05, sigma_rot_deg=7.0, p_gross=0.25,
        det_base=0.12, det_max=0.92, fp_rate=1.0, n_hyp=3, light=(0.70, 1.00), ambient=(0.15, 0.30),
    ),
}


# --------------------------------------------------------------------------- math


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(w):
    w = np.asarray(w, dtype=np.float64)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / th)
    return np.eye(3) + math.sin(th) * k + (1.0 - math.cos(th)) * (k @ k)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_angle(ra, rb):
    c = (np.trace(ra @ rb.T) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(1e-12, float(np.linalg.norm(v)))


# --------------------------------------------------------------------------- objects


def prim_extent(prim):
    if prim["kind"] == "box":
        return np.asarray(prim["dims"], dtype=np.float64)
    r, hh = prim["dims"][0], prim["dims"][1]
    return np.array([r, r, hh])


def box_mesh(h, spacing):
    verts, faces = [], []
    count = 0
    for axis in range(3):
        b, c = [a for a in range(3) if a != axis]
        nb = max(1, int(math.ceil(2 * h[b] / spacing)))
        nc = max(1, int(math.ceil(2 * h[c] / spacing)))
        gb = np.linspace(-h[b], h[b], nb + 1)
        gc = np.linspace(-h[c], h[c], nc + 1)
        bb, cc = np.meshgrid(gb, gc, indexing="ij")
        for sign in (-1.0, 1.0):
            v = np.zeros((bb.size, 3))
            v[:, axis] = sign * h[axis]
            v[:, b] = bb.ravel()
            v[:, c] = cc.ravel()
            idx = np.arange(bb.size).reshape(nb + 1, nc + 1) + count
            a0 = idx[:-1, :-1].ravel()
            a1 = idx[1:, :-1].ravel()
            a2 = idx[1:, 1:].ravel()
            a3 = idx[:-1, 1:].ravel()
            faces.append(np.concatenate([np.stack([a0, a1, a2], 1), np.stack([a0, a2, a3], 1)]))
            verts.append(v)
            count += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def cyl_mesh(r, hh, spacing, nseg=32):
    ang = np.linspace(0.0, 2 * math.pi, nseg, endpoint=False)
    nz = max(1, int(math.ceil(2 * hh / spacing)))
    zs = np.linspace(-hh, hh, nz + 1)
    zz, aa = np.meshgrid(zs, ang, indexing="ij")
    side = np.stack([r * np.cos(aa).ravel(), r * np.sin(aa).ravel(), zz.ravel()], 1)
    idx = np.arange(side.shape[0]).reshape(nz + 1, nseg)
    nxt = np.roll(idx, -1, axis=1)
    a0 = idx[:-1].ravel()
    a1 = nxt[:-1].ravel()
    a2 = nxt[1:].ravel()
    a3 = idx[1:].ravel()
    verts = [side]
    faces = [np.concatenate([np.stack([a0, a1, a2], 1), np.stack([a0, a2, a3], 1)])]
    count = side.shape[0]
    nr = max(1, int(math.ceil(r / spacing)))
    radii = np.linspace(0.0, r, nr + 1)[1:]
    for sign in (-1.0, 1.0):
        center = np.array([[0.0, 0.0, sign * hh]])
        rr, aa2 = np.meshgrid(radii, ang, indexing="ij")
        ring = np.stack([rr.ravel() * np.cos(aa2).ravel(), rr.ravel() * np.sin(aa2).ravel(),
                         np.full(rr.size, sign * hh)], 1)
        c_idx = count
        ring_idx = np.arange(ring.shape[0]).reshape(nr, nseg) + count + 1
        ring_nxt = np.roll(ring_idx, -1, axis=1)
        fan = np.stack([np.full(nseg, c_idx), ring_idx[0], ring_nxt[0]], 1)
        f = [fan]
        if nr > 1:
            b0 = ring_idx[:-1].ravel()
            b1 = ring_nxt[:-1].ravel()
            b2 = ring_nxt[1:].ravel()
            b3 = ring_idx[1:].ravel()
            f.append(np.stack([b0, b1, b2], 1))
            f.append(np.stack([b0, b2, b3], 1))
        verts += [center, ring]
        faces.append(np.concatenate(f))
        count += 1 + ring.shape[0]
    return np.concatenate(verts), np.concatenate(faces)


def tessellate(prims, spacing=MESH_SPACING):
    verts, faces, pidx = [], [], []
    count = 0
    for i, p in enumerate(prims):
        if p["kind"] == "box":
            v, f = box_mesh(p["dims"], spacing)
        else:
            v, f = cyl_mesh(p["dims"][0], p["dims"][1], spacing)
        verts.append(v @ p["R"].T + p["c"])
        faces.append(f + count)
        pidx.append(np.full(len(v), i, dtype=np.int64))
        count += len(v)
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(pidx)


def random_waves(rng, n, lam_min, lam_max, amp, channels_equal=False):
    ks, phases, amps = [], [], []
    for _ in range(n):
        lam = rng.uniform(lam_min, lam_max)
        ks.append(unit(rng.normal(size=3)) * (2 * math.pi / lam))
        phases.append(rng.uniform(0, 2 * math.pi))
        if channels_equal:
            amps.append(np.full(3, amp * rng.uniform(0.5, 1.0)))
        else:
            amps.append(amp * rng.uniform(-1.0, 1.0, 3))
    return dict(k=np.array(ks).reshape(-1, 3), phase=np.array(phases), amp=np.array(amps).reshape(-1, 3))


def make_appearance(rng, kind, n_prims):
    if kind == "plain":
        base = rng.uniform(0.5, 0.8)
        tint = rng.normal(0.0, 0.025, 3)
        cols = [np.clip(base + tint + rng.normal(0.0, 0.04) + rng.normal(0.0, 0.01, 3), 0.05, 0.95)
                for _ in range(n_prims)]
        waves = random_waves(rng, 3, 0.10, 0.25, 0.025)
        decals = []
        spec, shin = rng.uniform(0.08, 0.25), rng.uniform(6.0, 20.0)
    elif kind == "rich":
        cols = [np.array(colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.4, 0.9), rng.uniform(0.45, 0.95)))
                for _ in range(n_prims)]
        waves = random_waves(rng, 5, 0.035, 0.10, 0.16)
        decals = []
        for _ in range(int(rng.integers(3, 7))):
            decals.append(dict(
                n=unit(rng.normal(size=3)), d=rng.uniform(-0.05, 0.05), w=rng.uniform(0.008, 0.03),
                color=np.array(colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.0, 1.0), rng.uniform(0.05, 1.0))),
            ))
        spec, shin = rng.uniform(0.08, 0.30), rng.uniform(10.0, 40.0)
    elif kind == "metal":
        base = rng.uniform(0.35, 0.6)
        cols = [np.full(3, np.clip(base + rng.normal(0.0, 0.05), 0.1, 0.9)) for _ in range(n_prims)]
        waves = random_waves(rng, 2, 0.012, 0.03, 0.04, channels_equal=True)
        decals = []
        for _ in range(int(rng.integers(0, 3))):
            decals.append(dict(n=unit(rng.normal(size=3)), d=rng.uniform(-0.04, 0.04), w=rng.uniform(0.004, 0.012),
                               color=np.full(3, rng.uniform(0.05, 0.2))))
        spec, shin = rng.uniform(0.55, 0.95), rng.uniform(25.0, 80.0)
    else:
        raise ValueError(kind)
    return dict(cols=np.array(cols), waves=waves, decals=decals, spec=float(spec), shin=float(shin))


def albedo(app, pts, prim_idx):
    a = app["cols"][prim_idx].astype(np.float64).copy()
    w = app["waves"]
    if len(w["k"]):
        s = np.sin(pts @ w["k"].T + w["phase"][None, :])
        a += s @ w["amp"]
    for dec in app["decals"]:
        m = pts @ dec["n"] - dec["d"]
        inside = (m >= 0.0) & (m < dec["w"])
        a[inside] = 0.15 * a[inside] + 0.85 * dec["color"][None, :]
    return np.clip(a, 0.02, 0.98)


def make_object(rng, appearance, obj_id=None, n_range=(2, 5), size_scale=1.0):
    prims = []
    n = int(rng.integers(n_range[0], n_range[1]))
    for k in range(n):
        kind = "box" if rng.random() < 0.6 else "cyl"
        s = size_scale * (1.0 if k == 0 else rng.uniform(0.35, 0.75))
        if kind == "box":
            dims = rng.uniform(0.035, 0.09, 3) * s
        else:
            dims = np.array([rng.uniform(0.025, 0.06) * s, rng.uniform(0.03, 0.09) * s, 0.0])
        if k == 0:
            c, rot = np.zeros(3), np.eye(3)
        else:
            parent = prims[int(rng.integers(len(prims)))]
            d = unit(rng.normal(size=3))
            local = d / np.max(np.abs(d) / prim_extent(parent))
            c = parent["c"] + parent["R"] @ local
            rot = random_rotation(rng)
        prims.append(dict(kind=kind, c=c, R=rot, dims=dims))
    verts, faces, pidx = tessellate(prims)
    center = 0.5 * (verts.min(0) + verts.max(0))
    for p in prims:
        p["c"] = p["c"] - center
    verts = verts - center
    app = make_appearance(rng, appearance, len(prims))
    sel = np.linspace(0, len(verts) - 1, min(EVAL_VERTICES, len(verts))).astype(np.int64)
    eval_pts = verts[sel]
    diff = eval_pts[:, None, :] - eval_pts[None, :, :]
    diameter = float(np.sqrt((diff ** 2).sum(-1).max()))
    gain = 1.0 + rng.uniform(-0.04, 0.04, 3)
    if appearance == "metal":
        gain[:] = gain[0]
    vcol = np.clip(albedo(app, verts, pidx) * gain[None, :] + rng.normal(0.0, 0.01, (len(verts), 1 if appearance == "metal" else 3)), 0.0, 1.0)
    return dict(obj_id=obj_id, prims=prims, app=app, V=verts, F=faces, vertex_colors=vcol,
                eval_pts=eval_pts, diameter=diameter)


def object_model_json(obj):
    return {
        "obj_id": obj["obj_id"],
        "units": "meters",
        "vertices": np.round(obj["V"], 5).tolist(),
        "faces": obj["F"].astype(int).tolist(),
        "vertex_colors": np.round(obj["vertex_colors"], 3).tolist(),
    }


# --------------------------------------------------------------------------- ray casting


def world_prims(entities):
    out = []
    for ei, e in enumerate(entities):
        for pi, p in enumerate(e["prims"]):
            rw = e["R"] @ p["R"]
            cw = e["R"] @ p["c"] + e["t"]
            if p["kind"] == "box":
                rad = float(np.linalg.norm(p["dims"]))
            else:
                rad = float(math.hypot(p["dims"][0], p["dims"][1]))
            out.append(dict(kind=p["kind"], R=rw, c=cw, dims=np.asarray(p["dims"], dtype=np.float64),
                            rad=rad, ent=ei, prim=pi))
    return out


def intersect(prim, origins, dirs, eps=1e-6):
    n_rays = dirs.shape[0]
    t_out = np.full(n_rays, np.inf)
    n_out = np.zeros((n_rays, 3))
    oc = prim["c"][None, :] - origins
    proj = (oc * dirs).sum(1)
    d2 = (oc * oc).sum(1) - proj * proj
    cand = np.nonzero((d2 <= prim["rad"] ** 2) & (proj + prim["rad"] > eps))[0]
    if cand.size == 0:
        return t_out, n_out
    o = (origins[cand] - prim["c"][None, :]) @ prim["R"]
    d = dirs[cand] @ prim["R"]
    rows = np.arange(len(cand))
    if prim["kind"] == "box":
        h = prim["dims"]
        dd = np.where(np.abs(d) < 1e-12, 1e-12, d)
        inv = 1.0 / dd
        t1 = (-h[None, :] - o) * inv
        t2 = (h[None, :] - o) * inv
        tn = np.minimum(t1, t2)
        tf = np.maximum(t1, t2)
        tnear = tn.max(1)
        tfar = tf.min(1)
        hit = (tnear <= tfar) & (tnear > eps)
        ax = tn.argmax(1)
        nl = np.zeros_like(o)
        nl[rows, ax] = -np.sign(dd[rows, ax])
        tl = np.where(hit, tnear, np.inf)
    else:
        r, hh = prim["dims"][0], prim["dims"][1]
        ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - r * r
        disc = b * b - 4.0 * a * c
        ok = (a > 1e-12) & (disc >= 0.0)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        a_safe = np.where(a > 1e-12, a, 1.0)
        ts = (-b - sq) / (2.0 * a_safe)
        zs = oz + ts * dz
        side = ok & (ts > eps) & (np.abs(zs) <= hh)
        ts = np.where(side, ts, np.inf)
        dz_safe = np.where(np.abs(dz) < 1e-12, 1e-12, dz)
        best_cap = np.full(len(o), np.inf)
        cap_sign = np.zeros(len(o))
        for sgn in (1.0, -1.0):
            tc = (sgn * hh - oz) / dz_safe
            x = ox + tc * dx
            y = oy + tc * dy
            okc = (tc > eps) & (x * x + y * y <= r * r)
            tc = np.where(okc, tc, np.inf)
            better = tc < best_cap
            best_cap = np.where(better, tc, best_cap)
            cap_sign = np.where(better, sgn, cap_sign)
        tl = np.minimum(ts, best_cap)
        use_side = ts <= best_cap
        fin = np.isfinite(tl)
        p = o + np.where(fin, tl, 0.0)[:, None] * d
        nl = np.zeros_like(o)
        nl[:, 0] = np.where(use_side, p[:, 0] / r, 0.0)
        nl[:, 1] = np.where(use_side, p[:, 1] / r, 0.0)
        nl[:, 2] = np.where(use_side, 0.0, cap_sign)
    t_out[cand] = tl
    n_out[cand] = nl @ prim["R"].T
    return t_out, n_out


def cast(prims, origins, dirs):
    t_best = np.full(dirs.shape[0], np.inf)
    n_best = np.zeros_like(dirs)
    id_best = np.full(dirs.shape[0], -1, dtype=np.int64)
    for i, p in enumerate(prims):
        t, n = intersect(p, origins, dirs)
        better = t < t_best
        t_best = np.where(better, t, t_best)
        n_best[better] = n[better]
        id_best[better] = i
    return t_best, n_best, id_best


def camera_rays(cam, size, ss):
    k = np.asarray(cam["K"], dtype=np.float64)
    coords = (np.arange(size * ss) + 0.5) / ss
    uu, vv = np.meshgrid(coords, coords)
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 1)
    dc = pix @ np.linalg.inv(k).T
    dc /= np.linalg.norm(dc, axis=1, keepdims=True)
    dw = dc @ cam["R_CW"]
    ow = -cam["R_CW"].T @ cam["t_CW"]
    return np.broadcast_to(ow, dw.shape).copy(), dw


def plane_albedo(pa, xy):
    a = np.repeat(pa["base"][None, :], len(xy), 0)
    if len(pa["k"]):
        s = np.sin(xy @ pa["k"].T + pa["phase"][None, :])
        a = a + s @ pa["amp"]
    return np.clip(a, 0.02, 0.98)


def make_plane(rng, kind):
    if kind == "cloth":
        base = np.clip(rng.uniform(0.25, 0.55) + rng.normal(0.0, 0.06, 3), 0.05, 0.9)
        n, lam, amp = 4, (0.01, 0.06), 0.06
    elif kind == "wood":
        base = np.array([0.50, 0.33, 0.20]) * rng.uniform(0.8, 1.2)
        n, lam, amp = 4, (0.008, 0.08), 0.07
    else:
        base = np.full(3, rng.uniform(0.12, 0.25))
        n, lam, amp = 3, (0.01, 0.05), 0.03
    ks, ph, am = [], [], []
    for _ in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        ks.append(np.array([math.cos(ang), math.sin(ang)]) * 2 * math.pi / rng.uniform(*lam))
        ph.append(rng.uniform(0, 2 * math.pi))
        am.append(amp * (rng.uniform(-1, 1, 3) if kind != "dark" else np.full(3, rng.uniform(-1, 1))))
    return dict(base=base, k=np.array(ks), phase=np.array(ph), amp=np.array(am))


def render_view(scene, cam, rng, cfg, size):
    ss = SUPERSAMPLE
    ents = scene["entities"]
    prims = scene["prims"]
    origins, dirs = camera_rays(cam, size, ss)
    t_obj, n_obj, id_obj = cast(prims, origins, dirs)
    t_pl = np.where(dirs[:, 2] < -1e-9, -origins[:, 2] / np.where(dirs[:, 2] < -1e-9, dirs[:, 2], -1.0), np.inf)
    t_pl = np.where(t_pl > 1e-6, t_pl, np.inf)
    plane_hit = t_pl < t_obj
    t = np.minimum(t_obj, t_pl)
    fin = np.isfinite(t)
    pts = origins + np.where(fin, t, 0.0)[:, None] * dirs
    normal = np.where(plane_hit[:, None], np.array([0.0, 0.0, 1.0])[None, :], n_obj)
    ent_of_prim = np.array([p["ent"] for p in prims], dtype=np.int64)
    sub_of_prim = np.array([p["prim"] for p in prims], dtype=np.int64)
    label = np.full(len(t), -1, dtype=np.int64)
    obj_pix = fin & ~plane_hit
    label[obj_pix] = ent_of_prim[id_obj[obj_pix]]

    alb = np.tile(np.array([0.18, 0.18, 0.2]), (len(t), 1))
    ks = np.full(len(t), 0.04)
    shin = np.full(len(t), 5.0)
    pl = fin & plane_hit
    alb[pl] = plane_albedo(scene["plane"], pts[pl, :2])
    for ei, e in enumerate(ents):
        m = label == ei
        if not m.any():
            continue
        p_obj = (pts[m] - e["t"][None, :]) @ e["R"]
        alb[m] = albedo(e["app"], p_obj, sub_of_prim[id_obj[m]])
        ks[m] = e["app"]["spec"]
        shin[m] = e["app"]["shin"]

    light = scene["light"]
    ldir = light["dir"]
    shadow = np.zeros(len(t), dtype=bool)
    idx = np.nonzero(fin)[0]
    if idx.size:
        so = pts[idx] + normal[idx] * 1e-4
        sd = np.broadcast_to(ldir, so.shape).copy()
        ts, _, _ = cast(prims, so, sd)
        shadow[idx] = np.isfinite(ts)
    ndl = np.clip((normal * ldir[None, :]).sum(1), 0.0, None) * (~shadow)
    view = -dirs
    half = ldir[None, :] + view
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    ndh = np.clip((normal * half).sum(1), 0.0, None)
    spec = ks * ndh ** shin * (~shadow)
    col = alb * (light["ambient"] + light["intensity"] * ndl)[:, None] + (light["intensity"] * spec)[:, None]
    col[~fin] = np.array([0.2, 0.2, 0.22])
    img = col.reshape(size, ss, size, ss, 3).mean(axis=(1, 3))

    gain = rng.uniform(0.85, 1.15) * (1.0 + rng.uniform(-0.06, 0.06, 3))
    if cfg["grayscale"]:
        gain[:] = gain[0]
    img = img * gain[None, None, :]
    if cfg["grayscale"]:
        g = img @ np.array([0.299, 0.587, 0.114])
        img = np.repeat(g[..., None], 3, axis=2)
    img = img + rng.normal(0.0, rng.uniform(0.008, 0.02), img.shape[:2] + ((1,) if cfg["grayscale"] else (3,)))
    img = np.clip(img, 0.0, 1.0)
    label_img = label.reshape(size * ss, size * ss)
    return (img * 255.0 + 0.5).astype(np.uint8), label_img, (origins, dirs)


# --------------------------------------------------------------------------- scenes


def look_at(pos, target, roll):
    z = unit(target - pos)
    up = np.array([0.0, 0.0, 1.0])
    x = unit(np.cross(z, up))
    y = np.cross(z, x)
    rr = so3_exp(z * roll)
    x, y = rr @ x, rr @ y
    r_cw = np.stack([x, y, z])
    return r_cw, -r_cw @ pos


def place_entity(rng, pts, placed, radius, strict):
    last = None
    for _ in range(120):
        rot = random_rotation(rng)
        q = pts @ rot.T
        foot = 0.8 * float(np.max(np.linalg.norm(q[:, :2], axis=1)))
        while True:
            xy = rng.uniform(-radius, radius, 2)
            if np.linalg.norm(xy) <= radius:
                break
        cand = (rot, np.array([xy[0], xy[1], -q[:, 2].min() + 5e-4]), foot)
        last = cand
        if all(np.linalg.norm(xy - p[1][:2]) > 0.9 * (foot + p[2]) for p in placed):
            return cand
    return last if strict else None


def project_points(pts_w, cam):
    xc = pts_w @ cam["R_CW"].T + cam["t_CW"]
    k = cam["K"]
    z = xc[:, 2]
    return np.stack([k[0, 0] * xc[:, 0] / z + k[0, 2], k[1, 1] * xc[:, 1] / z + k[1, 2]], 1), z


def simulate_candidates(rng, cfg, cam, targets, visib, alone_px, library, objects, depth_bias, obj_bias, size):
    k = cam["K"]
    cands = []
    det_id = 0
    visible_targets = []
    min_px = 60.0 * (size / BASE_SIZE) ** 2
    for j, tg in enumerate(targets):
        v = visib[j]
        if alone_px[j] < min_px or v < VISIB_MIN:
            continue
        visible_targets.append(j)
        p_det = cfg["det_base"] + (cfg["det_max"] - cfg["det_base"]) * v ** 1.5
        if rng.random() > p_det:
            continue
        obj = objects[tg["obj_id"]]
        r_co = cam["R_CW"] @ tg["R"]
        t_co = cam["R_CW"] @ tg["t"] + cam["t_CW"]
        z = t_co[2]
        u0 = k[0, 0] * t_co[0] / z + k[0, 2]
        v0 = k[1, 1] * t_co[1] / z + k[1, 2]
        size_px = 0.5 * (k[0, 0] + k[1, 1]) * obj["diameter"] / z
        occl = 1.0 + (1.0 - v)
        du_det = rng.normal(0.0, cfg["sigma_2d"] * size_px * occl, 2)
        dz_det = depth_bias + rng.normal(0.0, cfg["sigma_depth_det"] * occl)
        srot = math.radians(cfg["sigma_rot_deg"]) * occl
        w_det = rng.normal(0.0, srot, 3)
        for h in range(cfg["n_hyp"]):
            du = du_det + rng.normal(0.0, 0.25 * cfg["sigma_2d"] * size_px, 2)
            dz = dz_det + rng.normal(0.0, 0.25 * cfg["sigma_depth_det"])
            w = w_det + rng.normal(0.0, 0.4 * srot, 3)
            r_bias = so3_exp(obj_bias[j])
            r_est = so3_exp(w) @ r_co @ r_bias
            if rng.random() < cfg["p_gross"]:
                axis = np.eye(3)[int(rng.integers(3))]
                ang = float(rng.choice([math.pi / 2, math.pi, -math.pi / 2]))
                r_est = so3_exp(w) @ r_co @ r_bias @ so3_exp(axis * ang)
            z_new = max(0.05, z * (1.0 + dz))
            ray = np.array([(u0 + du[0] - k[0, 2]) / k[0, 0], (v0 + du[1] - k[1, 2]) / k[1, 1], 1.0])
            t_est = ray * z_new
            quality = math.exp(-rot_angle(r_est, r_co) / 0.5) * math.exp(-abs(dz) / 0.08)
            score = 0.3 + 0.35 * v + 0.25 * quality + rng.normal(0.0, 0.08) - 0.04 * h
            cands.append(dict(obj_id=tg["obj_id"], det_id=det_id, R=r_est, t=t_est,
                              score=float(np.clip(score, 0.02, 0.98))))
        det_id += 1
    n_fp = int(rng.poisson(cfg["fp_rate"]))
    for _ in range(n_fp):
        if visible_targets and rng.random() < 0.5:
            j = visible_targets[int(rng.integers(len(visible_targets)))]
            wrong = [o for o in library if o != targets[j]["obj_id"]]
            oid = wrong[int(rng.integers(len(wrong)))]
            base_t = cam["R_CW"] @ targets[j]["t"] + cam["t_CW"]
            base_t = base_t * (1.0 + rng.normal(0.0, 0.05)) + rng.normal(0.0, 0.01, 3)
        else:
            oid = library[int(rng.integers(len(library)))]
            for _ in range(20):
                uv = rng.uniform(0.15 * size, 0.85 * size, 2)
                ray_c = np.array([(uv[0] - k[0, 2]) / k[0, 0], (uv[1] - k[1, 2]) / k[1, 1], 1.0])
                ray_w = cam["R_CW"].T @ ray_c
                origin = -cam["R_CW"].T @ cam["t_CW"]
                if ray_w[2] < -1e-3:
                    break
            tt = -origin[2] / min(ray_w[2], -1e-3)
            pw = origin + tt * ray_w + np.array([0.0, 0.0, 0.5 * objects[oid]["diameter"] * 0.5])
            base_t = cam["R_CW"] @ pw + cam["t_CW"]
        r_fp = random_rotation(rng)
        n_h = int(rng.integers(1, cfg["n_hyp"] + 1))
        fp_score = rng.uniform(0.1, 0.55)
        for h in range(n_h):
            r_h = so3_exp(rng.normal(0.0, 0.1, 3)) @ r_fp
            t_h = base_t * (1.0 + rng.normal(0.0, 0.02))
            cands.append(dict(obj_id=oid, det_id=det_id, R=r_h, t=t_h,
                              score=float(np.clip(fp_score - 0.04 * h + rng.normal(0.0, 0.03), 0.02, 0.98))))
        det_id += 1
    order = rng.permutation(len(cands))
    return [cands[i] for i in order]


def generate_scene(task):
    (seed, d_idx, s_idx, dataset, scene_id, library, objects, out_dir, cfg, size) = task
    # independent random streams: scene layout, per-view image formation and
    # per-view candidate simulation (so changing one never reshuffles the others)
    rng = np.random.default_rng([seed, d_idx, s_idx, 7])
    tgt_ids = [library[i] for i in rng.choice(len(library), size=cfg["n_targets"], replace=False)]
    entities, placed = [], []
    for oid in tgt_ids:
        obj = objects[oid]
        rot, t, foot = place_entity(rng, obj["eval_pts"], placed, 0.20, strict=True)
        placed.append((rot, t, foot))
        entities.append(dict(kind="target", obj_id=oid, prims=obj["prims"], app=obj["app"], R=rot, t=t))
    n_d = int(rng.integers(cfg["n_distractors"][0], cfg["n_distractors"][1] + 1))
    for _ in range(n_d):
        dobj = make_object(rng, cfg["appearance"], n_range=(1, 3), size_scale=rng.uniform(0.6, 1.0))
        res = place_entity(rng, dobj["eval_pts"], placed, 0.27, strict=False)
        if res is None:
            continue
        rot, t, foot = res
        placed.append(res)
        entities.append(dict(kind="distractor", obj_id=None, prims=dobj["prims"], app=dobj["app"], R=rot, t=t))
    scene = dict(entities=entities, prims=world_prims(entities), plane=make_plane(rng, cfg["plane"]))
    el = rng.uniform(math.radians(35), math.radians(80))
    az = rng.uniform(0, 2 * math.pi)
    scene["light"] = dict(dir=np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]),
                          intensity=rng.uniform(*cfg["light"]), ambient=rng.uniform(*cfg["ambient"]))
    center = np.array([0.0, 0.0, 0.05]) + np.append(rng.normal(0.0, 0.02, 2), 0.0)
    az0 = rng.uniform(0, 2 * math.pi)
    cams = []
    for v in range(N_VIEWS):
        a = az0 + v * math.pi / 2 + rng.uniform(-0.3, 0.3)
        e = rng.uniform(0.45, 1.0)
        dist = rng.uniform(0.62, 0.85)
        pos = center + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        r_cw, t_cw = look_at(pos, center + rng.normal(0.0, 0.03, 3), rng.uniform(-0.09, 0.09))
        px = size / BASE_SIZE
        f = BASE_FOCAL * px * rng.uniform(0.97, 1.03)
        kmat = np.array([[f, 0.0, size / 2 + px * rng.uniform(-2, 2)], [0.0, f, size / 2 + px * rng.uniform(-2, 2)],
                         [0.0, 0.0, 1.0]])
        cams.append(dict(K=kmat, R_CW=r_cw, t_CW=t_cw))
    targets = [dict(obj_id=e["obj_id"], R=e["R"], t=e["t"]) for e in entities if e["kind"] == "target"]
    target_ent = [i for i, e in enumerate(entities) if e["kind"] == "target"]
    # single-view estimators err consistently on a given object (shape/template
    # ambiguity): an orientation bias in the object frame shared by every view
    srot0 = math.radians(cfg["sigma_rot_deg"])
    obj_bias = [rng.normal(0.0, OBJ_BIAS_FRACTION * srot0, 3) for _ in targets]
    scene_dir = Path(out_dir) / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    views_json = []
    visib_all = []
    for v, cam in enumerate(cams):
        rng_img = np.random.default_rng([seed, d_idx, s_idx, 7, 100 + v])
        rng_cand = np.random.default_rng([seed, d_idx, s_idx, 7, 200 + v])
        img, label, (origins, dirs) = render_view(scene, cam, rng_img, cfg, size)
        visib, alone = [], []
        for ei in target_ent:
            own = [p for p in scene["prims"] if p["ent"] == ei]
            t_alone, _, _ = cast(own, origins, dirs)
            n_alone = int(np.isfinite(t_alone).sum())
            n_vis = int((label == ei).sum())
            alone.append(n_alone / SUPERSAMPLE ** 2)
            visib.append(0.0 if n_alone == 0 else n_vis / n_alone)
        visib_all.append(visib)
        depth_bias = rng_cand.normal(0.0, cfg["sigma_depth_view"])
        cands = simulate_candidates(rng_cand, cfg, cam, targets, visib, alone, library, objects, depth_bias, obj_bias,
                                    size)
        Image.fromarray(img).save(scene_dir / f"view_{v}.jpg", quality=JPEG_QUALITY)
        views_json.append({
            "view_id": v,
            "image": f"view_{v}.jpg",
            "K": cam["K"].tolist(),
            "T_CW": {"R": cam["R_CW"].tolist(), "t": cam["t_CW"].tolist()},
            "candidates": [{"obj_id": c["obj_id"], "det_id": int(c["det_id"]),
                            "T_CO": {"R": c["R"].tolist(), "t": c["t"].tolist()}, "score": c["score"]}
                           for c in cands],
        })
    scene_json = {
        "scene_id": scene_id,
        "dataset": dataset,
        "image_size": [size, size],
        "object_library": list(library),
        "views": views_json,
    }
    (scene_dir / "scene.json").write_text(json.dumps(scene_json))
    gt = {
        "scene_id": scene_id,
        "dataset": dataset,
        "image_size": [size, size],
        "cameras": [{"K": c["K"].tolist(), "R_CW": c["R_CW"].tolist(), "t_CW": c["t_CW"].tolist()} for c in cams],
        "targets": [{"obj_id": tg["obj_id"], "R": tg["R"].tolist(), "t": tg["t"].tolist(),
                     "visib": [float(visib_all[v][j]) for v in range(N_VIEWS)]} for j, tg in enumerate(targets)],
    }
    return gt


def _worker_count(requested=None):
    if requested:
        return max(1, int(requested))
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, n)


def generate_split(split, out_dir, scenes_per_dataset=None, workers=None, *, seed=None, image_size=None,
                   overrides=None):
    """Write method-visible inputs to ``out_dir`` and return the ground truth.

    The returned dictionary holds labels, cameras and evaluation geometry; it is
    never written inside ``out_dir``.  ``seed``/``image_size``/``overrides``
    exist only for offline calibration studies on a separate ``dev`` split; the
    train and test splits always use the module settings.
    """
    if split == "test":
        base_seed, n_scenes, n_obj, prefix = TEST_SEED, TEST_SCENES_PER_DATASET, TEST_OBJECTS_PER_DATASET, "test"
    elif split == "train":
        base_seed, n_scenes, n_obj, prefix = TRAIN_SEED, TRAIN_SCENES_PER_DATASET, TRAIN_OBJECTS_PER_DATASET, "train"
    elif split == "dev":
        if seed is None:
            raise ValueError("the dev split needs an explicit seed")
        base_seed, n_scenes, n_obj, prefix = int(seed), TRAIN_SCENES_PER_DATASET, TRAIN_OBJECTS_PER_DATASET, "dev"
    else:
        raise ValueError(split)
    if split != "dev" and (seed is not None or image_size is not None or overrides):
        raise ValueError("train/test splits are generated with the module settings only")
    seed = base_seed
    size = int(image_size or IMAGE_SIZE)
    configs = {d: dict(CONFIG[d], **((overrides or {}).get(d, {}))) for d in DATASETS}
    if scenes_per_dataset is not None:
        n_scenes = int(scenes_per_dataset)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "objects").mkdir(parents=True)
    (out_dir / "scenes").mkdir(parents=True)
    tasks, libraries, all_objects = [], {}, {}
    index = {"split": prefix, "scenes": [], "objects": [], "datasets": {}}
    for d_idx, dataset in enumerate(DATASETS):
        cfg = configs[dataset]
        library = []
        objects = {}
        for j in range(n_obj):
            oid = f"{prefix}_{SHORT[dataset]}_obj{j:02d}"
            obj = make_object(np.random.default_rng([seed, d_idx, 10_000 + j]), cfg["appearance"], obj_id=oid)
            objects[oid] = obj
            library.append(oid)
            (out_dir / "objects" / f"{oid}.json").write_text(json.dumps(object_model_json(obj)))
        libraries[dataset] = library
        all_objects.update(objects)
        slim = {oid: dict(prims=o["prims"], app=o["app"], eval_pts=o["eval_pts"], diameter=o["diameter"])
                for oid, o in objects.items()}
        scene_ids = [f"{prefix}_{SHORT[dataset]}_{s:03d}" for s in range(n_scenes)]
        index["datasets"][dataset] = {"scenes": scene_ids, "objects": library}
        index["scenes"] += scene_ids
        index["objects"] += library
        for s_idx, sid in enumerate(scene_ids):
            tasks.append((seed, d_idx, s_idx, dataset, sid, library, slim, str(out_dir), cfg, size))
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    n_workers = min(_worker_count(workers), len(tasks))
    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            gts = pool.map(generate_scene, tasks, chunksize=1)
    else:
        gts = [generate_scene(t) for t in tasks]
    gt = {
        "split": prefix,
        "datasets": {d: {"objects": libraries[d], "scenes": index["datasets"][d]["scenes"]} for d in DATASETS},
        "objects": {oid: {"eval_points": np.round(o["eval_pts"], 6).tolist(), "diameter": o["diameter"]}
                    for oid, o in all_objects.items()},
        "scenes": gts,
    }
    return gt


def main():
    parser = argparse.ArgumentParser(description="Offline generation of the visible train split.")
    parser.add_argument("--make-train", action="store_true")
    parser.add_argument("--inputs-out", default="data/train")
    parser.add_argument("--labels-out", default="eval/train/train_gt.json")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if not args.make_train:
        parser.error("only --make-train is supported from the command line")
    gt = generate_split("train", args.inputs_out, workers=args.workers)
    Path(args.labels_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.labels_out).write_text(json.dumps(gt))
    print(f"wrote {len(gt['scenes'])} train scenes to {args.inputs_out} and labels to {args.labels_out}")


if __name__ == "__main__":
    main()
