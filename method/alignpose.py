"""AlignPose reproduction: multi-view aggregation + feature-metric refinement.

Pipeline (paper Sec. 3 and App. A):
  1. per-view candidates T_CO are mapped to the world frame, T_WO = T_WC T_CO;
  2. global (inter-class) 3D NMS on world axis-aligned boxes, IoU 0.4;
  3. for every kept candidate and every view: a perspective crop around the
     coarse pose gives the query feature map; the object is rendered (RGB-D)
     from the crop camera at the coarse pose, patch descriptors centred on the
     object are lifted to object coordinates through the rendered depth
     (registered features); features come from a frozen DINOv2 layer with the
     final LayerNorm, projected on a per-object PCA basis computed from
     rendered templates (mesh-only onboarding) and L2-normalised;
  4. one world pose T_WO is optimised with Levenberg-Marquardt (<= 30
     iterations) on sum_C sum_i rho(|| p_i - F_q(pi_C(T_CW T_WO x_i)) ||) with
     the Barron robust loss (alpha = -5, c = 0.5);
  5. score s = 1 - mean_C L_FE^C (per-view mean robust loss, normalised to
     [0, 1]); a second NMS on the refined poses removes converged duplicates.

Usage:  python -m method.alignpose --input-dir INPUTS --output predictions.json
(420 x 420 perspective crops, DINOv2-S layer 9, PCA 128, 40 onboarding templates)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from method.geometry import detection_json, load_index, load_mesh, load_scene, nms, so3_exp, world_candidates

DEFAULT_MODEL_DIR = "data/downloads/dinov2-small"

_G = {}


# --------------------------------------------------------------------------- robust loss


def barron_rho(x, alpha, c):
    z = (x / c) ** 2
    if alpha == 2:
        return 0.5 * z
    if alpha == 0:
        return np.log(0.5 * z + 1.0)
    b = abs(alpha - 2.0)
    return (b / alpha) * ((z / b + 1.0) ** (alpha / 2.0) - 1.0)


def barron_weight(x, alpha, c):
    """rho'(x) / x, the IRLS weight of the squared residual norm."""
    z = (x / c) ** 2
    if alpha == 2:
        return np.full_like(x, 1.0 / c ** 2)
    if alpha == 0:
        return (2.0 / c ** 2) / (z + 2.0)
    b = abs(alpha - 2.0)
    return (1.0 / c ** 2) * (z / b + 1.0) ** (alpha / 2.0 - 1.0)


# --------------------------------------------------------------------------- features


def make_extractor(cfg):
    from method.features import DinoFeatures, PixelPatchFeatures

    if cfg["features"] == "dino":
        return DinoFeatures(cfg["model_dir"], cfg["layer"])
    if cfg["features"] == "pixels":
        return PixelPatchFeatures(patch=14)
    raise ValueError(cfg["features"])


def build_view_reps(scene, mesh, pca, r_wo, t_wo, cfg):
    """Fixed per-view representations for one coarse pose: query maps + registered features."""
    from method.features import patch_centers, project
    from method.render import crop_camera, rasterize, warp_crop

    crop = cfg["crop"]
    extractor = _G["extractor"]
    patch = extractor.patch
    centers = patch_centers(crop, patch)
    ii, jj = np.meshgrid(centers, centers, indexing="ij")
    crops, renders, metas = [], [], []
    for v in scene["views"]:
        h, w = v["image"].shape[:2]
        cam = crop_camera(v["K"], v["R_CW"], v["t_CW"], r_wo, t_wo, mesh, w, h, crop, cfg["pad"])
        if cam is None:
            continue
        query = warp_crop(v["image"], v["K"], cam, crop)
        r_po = cam["R_pw"] @ r_wo
        t_po = cam["R_pw"] @ t_wo + cam["t_pw"]
        rgb, depth = rasterize(mesh, r_po, t_po, cam["K"], crop, crop)
        crops.append(query)
        renders.append(rgb.astype(np.float32))
        metas.append((cam, depth, r_po, t_po))
    if not crops:
        return []
    feats = extractor(np.stack(crops + renders))
    n = len(crops)
    fq_all = project(feats[:n], pca)
    fr_all = project(feats[n:], pca)
    reps = []
    for i, (cam, depth, r_po, t_po) in enumerate(metas):
        d = depth[ii, jj]
        m = np.isfinite(d)
        if m.sum() < 3:
            continue
        kp = cam["K"]
        u = jj[m] + 0.5
        vv = ii[m] + 0.5
        pc = np.stack([(u - kp[0, 2]) / kp[0, 0] * d[m], (vv - kp[1, 2]) / kp[1, 1] * d[m], d[m]], 1)
        pts_obj = (pc - t_po[None, :]) @ r_po
        reps.append({"R_pw": cam["R_pw"], "t_pw": cam["t_pw"], "K": kp, "Fq": fq_all[i], "pts": pts_obj,
                     "desc": fr_all[i][m], "patch": float(patch)})
    return reps


# --------------------------------------------------------------------------- optimisation


def _sample(fq, gu, gv):
    g_h, g_w = fq.shape[:2]
    x0 = np.clip(np.floor(gu), 0, g_w - 2).astype(np.int64)
    y0 = np.clip(np.floor(gv), 0, g_h - 2).astype(np.int64)
    ax = (gu - x0)[:, None]
    ay = (gv - y0)[:, None]
    f00 = fq[y0, x0]
    f01 = fq[y0, x0 + 1]
    f10 = fq[y0 + 1, x0]
    f11 = fq[y0 + 1, x0 + 1]
    val = (1 - ay) * ((1 - ax) * f00 + ax * f01) + ay * ((1 - ax) * f10 + ax * f11)
    dgu = (1 - ay) * (f01 - f00) + ay * (f11 - f10)
    dgv = (1 - ax) * (f10 - f00) + ax * (f11 - f01)
    return val, dgu, dgv


def _view_terms(r, t, rep, cfg, with_jac):
    pts = rep["pts"]
    rx = pts @ r.T
    xw = rx + t[None, :]
    xc = xw @ rep["R_pw"].T + rep["t_pw"][None, :]
    z = xc[:, 2]
    k = rep["K"]
    patch = rep["patch"]
    g_h, g_w = rep["Fq"].shape[:2]
    zs = np.where(z > 1e-4, z, 1.0)
    u = k[0, 0] * xc[:, 0] / zs + k[0, 2]
    v = k[1, 1] * xc[:, 1] / zs + k[1, 2]
    gu = (u - patch / 2.0) / patch
    gv = (v - patch / 2.0) / patch
    inside = (z > 1e-4) & (gu >= 0) & (gu <= g_w - 1) & (gv >= 0) & (gv <= g_h - 1)
    rho_max = barron_rho(np.array([2.0]), cfg["alpha"], cfg["scale"])[0]
    rho = np.full(len(pts), rho_max)
    if not inside.any():
        return rho, None, None
    idx = np.nonzero(inside)[0]
    val, dgu, dgv = _sample(rep["Fq"], gu[idx], gv[idx])
    res = rep["desc"][idx] - val
    x = np.linalg.norm(res, axis=1)
    rho[idx] = barron_rho(x, cfg["alpha"], cfg["scale"])
    if not with_jac:
        return rho, None, None
    w = barron_weight(x, cfg["alpha"], cfg["scale"])
    zi = z[idx]
    xi = xc[idx]
    n = len(idx)
    dudx = np.zeros((n, 2, 3))
    dudx[:, 0, 0] = k[0, 0] / zi
    dudx[:, 0, 2] = -k[0, 0] * xi[:, 0] / zi ** 2
    dudx[:, 1, 1] = k[1, 1] / zi
    dudx[:, 1, 2] = -k[1, 1] * xi[:, 1] / zi ** 2
    dudx /= patch
    a = rx[idx]
    dxw = np.zeros((n, 3, 6))
    # d(X_w)/d(omega) = -[R x]_x ; d(X_w)/d(v) = I   (update R <- exp(omega) R, t <- t + v)
    dxw[:, 0, 1], dxw[:, 0, 2] = a[:, 2], -a[:, 1]
    dxw[:, 1, 0], dxw[:, 1, 2] = -a[:, 2], a[:, 0]
    dxw[:, 2, 0], dxw[:, 2, 1] = a[:, 1], -a[:, 0]
    dxw[:, 0, 3] = dxw[:, 1, 4] = dxw[:, 2, 5] = 1.0
    dxc = np.einsum("ij,njk->nik", rep["R_pw"], dxw)
    duv = np.einsum("nij,njk->nik", dudx, dxc)
    dfeat = np.stack([dgu, dgv], axis=2)
    jac = -np.einsum("ndi,nik->ndk", dfeat, duv)
    hess = np.einsum("ndk,ndl,n->kl", jac, jac, w)
    grad = np.einsum("ndk,nd,n->k", jac, res, w)
    return rho, hess, grad


def total_cost(r, t, reps, cfg):
    return float(sum(_view_terms(r, t, rep, cfg, False)[0].sum() for rep in reps))


def pose_score(r, t, reps, cfg):
    if not reps:
        return 0.0
    rho_max = barron_rho(np.array([2.0]), cfg["alpha"], cfg["scale"])[0]
    losses = [float(_view_terms(r, t, rep, cfg, False)[0].mean() / rho_max) for rep in reps]
    return float(np.clip(1.0 - np.mean(losses), 0.0, 1.0))


def lm_refine(r, t, reps, cfg):
    r, t = r.copy(), t.copy()
    if not reps:
        return r, t
    cost = total_cost(r, t, reps, cfg)
    lam = cfg["lm_lambda"]
    for _ in range(cfg["max_iters"]):
        hess = np.zeros((6, 6))
        grad = np.zeros(6)
        for rep in reps:
            _, h_v, g_v = _view_terms(r, t, rep, cfg, True)
            if h_v is not None:
                hess += h_v
                grad += g_v
        if not np.any(hess):
            break
        a = hess + lam * np.diag(np.diag(hess)) + 1e-9 * np.eye(6)
        try:
            delta = -np.linalg.solve(a, grad)
        except np.linalg.LinAlgError:
            break
        r_new = so3_exp(delta[:3]) @ r
        t_new = t + delta[3:]
        cost_new = total_cost(r_new, t_new, reps, cfg)
        if cost_new < cost:
            r, t, cost = r_new, t_new, cost_new
            lam = max(lam / 10.0, 1e-6)
            if np.linalg.norm(delta) < 1e-7:
                break
        else:
            lam *= 10.0
            if lam > 1e6:
                break
    return r, t


# --------------------------------------------------------------------------- scene processing


def get_mesh(input_dir, obj_id):
    cache = _G.setdefault("meshes", {})
    if obj_id not in cache:
        cache[obj_id] = load_mesh(input_dir, obj_id)
    return cache[obj_id]


def refine_candidate(scene, cand, input_dir, cfg):
    mesh = get_mesh(input_dir, cand["obj_id"])
    pca = _G["pcas"][cand["obj_id"]]
    r, t = cand["R"], cand["t"]
    reps = build_view_reps(scene, mesh, pca, r, t, cfg)
    if cfg["enumerate"]:
        return r, t, pose_score(r, t, reps, cfg)
    r, t = lm_refine(r, t, reps, cfg)
    for _ in range(cfg["stages"] - 1):
        reps = build_view_reps(scene, mesh, pca, r, t, cfg)
        r, t = lm_refine(r, t, reps, cfg)
    return r, t, pose_score(r, t, reps, cfg)


def process_scene(input_dir, scene_id, cfg):
    scene = load_scene(input_dir, scene_id)
    cands = world_candidates(scene)
    meshes = {c["obj_id"]: get_mesh(input_dir, c["obj_id"]) for c in cands}
    if cfg["nms"]:
        kept, clusters = nms(cands, meshes, cfg["nms_iou"])
    else:
        kept, clusters = cands, [[c] for c in cands]
    if not cfg["refine"]:
        return [detection_json(scene_id, c["obj_id"], c["R"], c["t"], c["score"]) for c in kept]
    refined = []
    for keep, members in zip(kept, clusters):
        hyps = members if cfg["enumerate"] else members[: cfg["hypotheses"]]
        best = None
        for h in hyps:
            r, t, s = refine_candidate(scene, h, input_dir, cfg)
            if best is None or s > best["score"]:
                best = {"obj_id": h["obj_id"], "R": r, "t": t, "score": s}
        refined.append(best)
    if cfg["nms"]:
        refined, _ = nms(refined, meshes, cfg["nms_iou"])
    return [detection_json(scene_id, c["obj_id"], c["R"], c["t"], c["score"]) for c in refined]


# --------------------------------------------------------------------------- workers


def _init_worker(cfg, pcas):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _G.clear()
    _G["cfg"] = cfg
    _G["pcas"] = pcas
    if cfg["refine"]:
        import torch

        torch.set_num_threads(1)
        _G["extractor"] = make_extractor(cfg)


def _onboard_task(args):
    from method.features import onboard_pca

    input_dir, obj_id = args
    cfg = _G["cfg"]
    mesh = get_mesh(input_dir, obj_id)
    return onboard_pca(mesh, _G["extractor"], cfg["crop"], cfg["n_templates"], cfg["pca_dim"])


def _scene_task(args):
    input_dir, scene_id = args
    t0 = time.time()
    dets = process_scene(input_dir, scene_id, _G["cfg"])
    return scene_id, dets, time.time() - t0


def worker_count(requested):
    if requested:
        return max(1, requested)
    env = os.environ.get("ALIGNPOSE_WORKERS")
    if env:
        return max(1, int(env))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _digest(*parts):
    h = hashlib.sha1()
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
        h.update(b"|")
    return h.hexdigest()


def run(cfg, input_dir, output, workers=None, max_scenes=None):
    """Run the pipeline on every scene of ``input_dir``.

    If ``ALIGNPOSE_CACHE_DIR`` is set, per-object PCA bases and per-scene results
    are cached there, keyed by the configuration and the input file contents, so
    an interrupted run resumes where it stopped (results are identical).
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    index = load_index(input_dir)
    scene_ids = index["scenes"][:max_scenes] if max_scenes else index["scenes"]
    n_workers = worker_count(workers)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    cache = Path(os.environ["ALIGNPOSE_CACHE_DIR"]) if os.environ.get("ALIGNPOSE_CACHE_DIR") else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    objects = list(index["objects"])
    obj_bytes = {o: (Path(input_dir) / "objects" / f"{o}.json").read_bytes() for o in objects}
    pcas = {}
    if cfg["refine"]:
        onboard_cfg = json.dumps({k: cfg[k] for k in ("features", "model_dir", "layer", "crop", "n_templates", "pca_dim")},
                                 sort_keys=True)
        keys = {o: _digest(onboard_cfg, obj_bytes[o]) for o in objects}
        todo = []
        for o in objects:
            f = cache / f"pca_{keys[o]}.npz" if cache else None
            if f is not None and f.is_file():
                with np.load(f) as z:
                    pcas[o] = {"mean": z["mean"], "components": z["components"]}
            else:
                todo.append(o)
        if todo:
            with ctx.Pool(min(n_workers, len(todo)), initializer=_init_worker, initargs=(cfg, {})) as pool:
                for o, pca in zip(todo, pool.map(_onboard_task, [(input_dir, x) for x in todo], chunksize=1)):
                    pcas[o] = pca
                    if cache:
                        np.savez(cache / f"pca_{keys[o]}.npz", mean=pca["mean"], components=pca["components"])
        print(f"onboarded {len(pcas)} objects ({len(todo)} computed) in {time.time() - t0:.1f}s", flush=True)
    run_cfg = json.dumps({k: v for k, v in cfg.items() if k != "name"}, sort_keys=True)
    lib_key = _digest(*[obj_bytes[o] for o in sorted(objects)])
    results, scene_keys, tasks = {}, {}, []
    for sid in scene_ids:
        sdir = Path(input_dir) / "scenes" / sid
        scene_keys[sid] = _digest(run_cfg, lib_key, *[f.read_bytes() for f in sorted(sdir.iterdir()) if f.is_file()])
        f = cache / f"scene_{scene_keys[sid]}.json" if cache else None
        if f is not None and f.is_file():
            results[sid] = json.loads(f.read_text())
        else:
            tasks.append((input_dir, sid))
    if results:
        print(f"reused {len(results)} cached scenes", flush=True)
    if tasks:
        with ctx.Pool(min(n_workers, len(tasks)), initializer=_init_worker, initargs=(cfg, pcas)) as pool:
            for i, (sid, dets, dt) in enumerate(pool.imap(_scene_task, tasks, chunksize=1)):
                results[sid] = dets
                if cache:
                    tmp = cache / f"scene_{scene_keys[sid]}.json.tmp"
                    tmp.write_text(json.dumps(dets))
                    tmp.replace(cache / f"scene_{scene_keys[sid]}.json")
                if i % 20 == 0 or i == len(tasks) - 1:
                    print(f"[{i + 1}/{len(tasks)}] {sid}: {len(dets)} detections, {dt:.1f}s "
                          f"(elapsed {time.time() - t0:.0f}s)", flush=True)
    detections = [d for sid in scene_ids for d in results[sid]]
    payload = {"method": cfg["name"], "config": {k: v for k, v in cfg.items() if k != "name"},
               "detections": detections}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload))
    print(f"wrote {len(detections)} detections for {len(scene_ids)} scenes to {output} "
          f"in {time.time() - t0:.0f}s", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", default="alignpose")
    p.add_argument("--features", choices=["dino", "pixels"], default="dino")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--crop", type=int, default=420)
    p.add_argument("--pad", type=float, default=1.4)
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--n-templates", type=int, default=40)
    p.add_argument("--nms-iou", type=float, default=0.4)
    p.add_argument("--no-nms", action="store_true")
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--alpha", type=float, default=-5.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--lm-lambda", type=float, default=1e-2)
    p.add_argument("--stages", type=int, default=1, help="re-render registered features and refine again")
    p.add_argument("--hypotheses", type=int, default=1, help="refine the top-k members of each NMS cluster")
    p.add_argument("--enumerate", action="store_true", help="score every cluster member without optimisation")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-scenes", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = {
        "name": a.name, "features": a.features, "model_dir": a.model_dir, "layer": a.layer, "crop": a.crop,
        "pad": a.pad, "pca_dim": a.pca_dim, "n_templates": a.n_templates, "nms_iou": a.nms_iou,
        "nms": not a.no_nms, "refine": not a.no_refine or a.enumerate, "alpha": a.alpha, "scale": a.scale,
        "max_iters": a.max_iters, "lm_lambda": a.lm_lambda, "stages": max(1, a.stages),
        "hypotheses": max(1, a.hypotheses), "enumerate": a.enumerate,
    }
    run(cfg, a.input_dir, a.output, a.workers, a.max_scenes)


if __name__ == "__main__":
    sys.exit(main())
