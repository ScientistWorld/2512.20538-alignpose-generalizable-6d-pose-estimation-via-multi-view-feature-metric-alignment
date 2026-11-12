"""Vectorized z-buffer rasterizer and perspective crops for AlignPose.

The rasterizer renders a textured (per-vertex colour) triangle mesh with a
simple head-light shading and returns colour and depth.  Pixel (row r, col c)
covers continuous image coordinates [c, c+1) x [r, r+1); its centre is
(c + 0.5, r + 0.5).
"""

from __future__ import annotations

import numpy as np


def rasterize(mesh, r_co, t_co, k, width, height, near=0.01):
    vc = mesh.V @ r_co.T + t_co[None, :]
    z = vc[:, 2]
    depth = np.full(height * width, np.inf)
    color = np.zeros((height * width, 3))
    zf = z[mesh.F]
    keep = np.all(zf > near, axis=1)
    if not keep.any():
        return color.reshape(height, width, 3), depth.reshape(height, width)
    faces = mesh.F[keep]
    fids = np.nonzero(keep)[0]
    zs = np.where(z > near, z, 1.0)
    u = k[0, 0] * vc[:, 0] / zs + k[0, 2]
    v = k[1, 1] * vc[:, 1] / zs + k[1, 2]
    tu = u[faces]
    tv = v[faces]
    xmin = np.ceil(tu.min(1) - 0.5).astype(np.int64)
    xmax = np.floor(tu.max(1) - 0.5).astype(np.int64)
    ymin = np.ceil(tv.min(1) - 0.5).astype(np.int64)
    ymax = np.floor(tv.max(1) - 0.5).astype(np.int64)
    xmin = np.clip(xmin, 0, width - 1)
    xmax = np.clip(xmax, -1, width - 1)
    ymin = np.clip(ymin, 0, height - 1)
    ymax = np.clip(ymax, -1, height - 1)
    wf = xmax - xmin + 1
    hf = ymax - ymin + 1
    ok = (wf > 0) & (hf > 0)
    area = (tu[:, 1] - tu[:, 0]) * (tv[:, 2] - tv[:, 0]) - (tu[:, 2] - tu[:, 0]) * (tv[:, 1] - tv[:, 0])
    ok &= np.abs(area) > 1e-12
    if not ok.any():
        return color.reshape(height, width, 3), depth.reshape(height, width)
    faces, fids, tu, tv, area = faces[ok], fids[ok], tu[ok], tv[ok], area[ok]
    xmin, ymin, wf, hf = xmin[ok], ymin[ok], wf[ok], hf[ok]
    counts = wf * hf
    total = int(counts.sum())
    tri = np.repeat(np.arange(len(faces)), counts)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    local = np.arange(total) - start
    px = xmin[tri] + local % wf[tri]
    py = ymin[tri] + local // wf[tri]
    pcx = px + 0.5
    pcy = py + 0.5
    ax, ay = tu[tri, 0], tv[tri, 0]
    bx, by = tu[tri, 1], tv[tri, 1]
    cx, cy = tu[tri, 2], tv[tri, 2]
    ar = area[tri]
    w0 = ((bx - pcx) * (cy - pcy) - (cx - pcx) * (by - pcy)) / ar
    w1 = ((cx - pcx) * (ay - pcy) - (ax - pcx) * (cy - pcy)) / ar
    w2 = 1.0 - w0 - w1
    eps = -1e-7
    inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
    if not inside.any():
        return color.reshape(height, width, 3), depth.reshape(height, width)
    tri, px, py, w0, w1, w2 = tri[inside], px[inside], py[inside], w0[inside], w1[inside], w2[inside]
    fz = z[faces[tri]]
    b0 = w0 / fz[:, 0]
    b1 = w1 / fz[:, 1]
    b2 = w2 / fz[:, 2]
    inv = b0 + b1 + b2
    zpix = 1.0 / inv
    pid = py * width + px
    order = np.lexsort((zpix, pid))
    pid_s = pid[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = pid_s[1:] != pid_s[:-1]
    sel = order[first]
    depth[pid[sel]] = zpix[sel]
    fv = faces[tri[sel]]
    bb = np.stack([b0[sel], b1[sel], b2[sel]], 1) / inv[sel][:, None]
    col = (bb[:, :, None] * mesh.C[fv]).sum(1)
    n_c = mesh.face_normals[fids[tri[sel]]] @ r_co.T
    cen = vc[fv].mean(1)
    view = -cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    shade = 0.35 + 0.65 * np.abs((n_c * view).sum(1))
    color[pid[sel]] = np.clip(col * shade[:, None], 0.0, 1.0)
    return color.reshape(height, width, 3), depth.reshape(height, width)


def crop_camera(k, r_cw, t_cw, r_wo, t_wo, mesh, width, height, crop, pad):
    """Perspective crop around the projection of a coarse pose (FoundPose/MegaPose style).

    The crop camera C' shares the optical centre of C, looks along the ray through
    the centre of the projected object box, and has a focal length chosen so the
    padded box fills the ``crop`` x ``crop`` image.
    """
    r_co = r_cw @ r_wo
    t_co = r_cw @ t_wo + t_cw
    x = mesh.V_sub @ r_co.T + t_co[None, :]
    if np.any(x[:, 2] < 0.02):
        return None
    u = k[0, 0] * x[:, 0] / x[:, 2] + k[0, 2]
    v = k[1, 1] * x[:, 1] / x[:, 2] + k[1, 2]
    umin, umax, vmin, vmax = u.min(), u.max(), v.min(), v.max()
    if umax < 0 or vmax < 0 or umin > width or vmin > height:
        return None
    cu, cv = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
    size = max(umax - umin, vmax - vmin, 4.0) * pad
    ray = np.linalg.solve(k, np.array([cu, cv, 1.0]))
    zc = ray / np.linalg.norm(ray)
    xc = np.array([1.0, 0.0, 0.0]) - zc * zc[0]
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    r_pc = np.stack([xc, yc, zc])
    f = 0.5 * (k[0, 0] + k[1, 1]) * crop / size
    kp = np.array([[f, 0.0, crop / 2.0], [0.0, f, crop / 2.0], [0.0, 0.0, 1.0]])
    return {"K": kp, "R_pc": r_pc, "R_pw": r_pc @ r_cw, "t_pw": r_pc @ t_cw}


def bilinear_image(img, x, y):
    """Sample ``img`` (H, W, C) at array coordinates (x=col, y=row); zero outside."""
    h, w = img.shape[:2]
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)))
    xp = np.clip(x + 1.0, 0.0, w + 1 - 1e-6)
    yp = np.clip(y + 1.0, 0.0, h + 1 - 1e-6)
    x0 = np.floor(xp).astype(np.int64)
    y0 = np.floor(yp).astype(np.int64)
    x1 = np.minimum(x0 + 1, w + 1)
    y1 = np.minimum(y0 + 1, h + 1)
    ax = (xp - x0)[:, None]
    ay = (yp - y0)[:, None]
    top = padded[y0, x0] * (1 - ax) + padded[y0, x1] * ax
    bot = padded[y1, x0] * (1 - ax) + padded[y1, x1] * ax
    return top * (1 - ay) + bot * ay


def warp_crop(img, k, cam, crop):
    g = np.arange(crop) + 0.5
    uu, vv = np.meshgrid(g, g)
    kp = cam["K"]
    rays_p = np.stack([(uu - kp[0, 2]) / kp[0, 0], (vv - kp[1, 2]) / kp[1, 1], np.ones_like(uu)], -1).reshape(-1, 3)
    rays_c = rays_p @ cam["R_pc"]
    p = rays_c @ k.T
    u = p[:, 0] / p[:, 2]
    v = p[:, 1] / p[:, 2]
    return bilinear_image(img, u - 0.5, v - 0.5).reshape(crop, crop, img.shape[2]).astype(np.float32)


def look_at_origin(direction, distance):
    """Camera on the unit ``direction`` at ``distance`` looking at the object origin."""
    zc = -direction / np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0]) if abs(zc[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    r_co = np.stack([xc, yc, zc])
    return r_co, np.array([0.0, 0.0, distance])


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
