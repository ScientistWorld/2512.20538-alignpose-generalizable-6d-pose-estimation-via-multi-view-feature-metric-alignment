"""Validity checks and BOP-style pose metrics for the visible train split (independent copy of the test-side scorer).

All numbers are recomputed here from the raw submitted poses and confidence
scores; nothing a submission writes besides ``detections`` is read.

Evaluation is per image, as in the BOP protocol the paper follows: each scene
detection (a world-frame pose) is an estimate in every view of its scene, and
each (image, object) pair whose visible fraction is at least 0.1 is one
ground-truth instance.

Errors (objects are asymmetric, so no symmetry handling is needed):
  * MSSD: maximum 3D distance between model points under the estimated and the
    ground-truth pose, relative to the object diameter.
  * MSPD: maximum 2D distance between the projected model points in the image.

AR follows the BOP localisation protocol (mean recall over 10 thresholds per
error, averaged over MSSD and MSPD; VSD is not computed). Each object appears at
most once per scene, so only the highest-scoring estimate of an object in a
scene is matched; further estimates of the same object are false positives in
AP. AP follows the BOP/COCO detection protocol (101-point interpolated average
precision over the same thresholds, per object, averaged over objects and over
MSSD and MSPD).
"""

from __future__ import annotations

import math

import numpy as np

MSSD_FRACTIONS = np.arange(1, 11) * 0.05
MSPD_PIXELS_AT_640 = np.arange(1, 11) * 5.0
VISIB_MIN = 0.1
MAX_DETECTIONS_PER_SCENE = 100
ROTATION_TOLERANCE = 1e-3
MAX_TRANSLATION = 1e3

METRICS = ("AR", "AP")


class InvalidSubmission(Exception):
    pass


def worst_metrics():
    return {"AR": 0.0, "AP": 0.0}


def _array(value, shape, what):
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidSubmission(f"{what} is not numeric: {exc}") from exc
    if arr.shape != shape:
        raise InvalidSubmission(f"{what} has shape {arr.shape}, expected {shape}")
    if not np.all(np.isfinite(arr)):
        raise InvalidSubmission(f"{what} contains non-finite values")
    return arr


def parse_detections(payload, scene_libraries):
    """Validate a submission; return {scene_id: [(score, index, obj_id, R, t), ...]}.

    Any impossible value invalidates the whole run.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("detections"), list):
        raise InvalidSubmission("output must be a JSON object with a 'detections' list")
    per_scene = {sid: [] for sid in scene_libraries}
    for i, det in enumerate(payload["detections"]):
        if not isinstance(det, dict):
            raise InvalidSubmission(f"detection {i} is not an object")
        sid = det.get("scene_id")
        oid = det.get("obj_id")
        if not isinstance(sid, str) or sid not in scene_libraries:
            raise InvalidSubmission(f"detection {i} has unknown scene_id {sid!r}")
        if not isinstance(oid, str) or oid not in scene_libraries[sid]:
            raise InvalidSubmission(f"detection {i} has obj_id {oid!r} outside the scene's object library")
        rot = _array(det.get("R"), (3, 3), f"detection {i} R")
        trans = _array(det.get("t"), (3,), f"detection {i} t")
        score = det.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise InvalidSubmission(f"detection {i} has a non-finite or non-numeric score")
        if np.abs(rot.T @ rot - np.eye(3)).max() > ROTATION_TOLERANCE or abs(np.linalg.det(rot) - 1.0) > ROTATION_TOLERANCE:
            raise InvalidSubmission(f"detection {i} R is not a rotation matrix")
        if np.abs(trans).max() > MAX_TRANSLATION:
            raise InvalidSubmission(f"detection {i} translation is implausibly large")
        per_scene[sid].append((float(score), i, oid, rot, trans))
        if len(per_scene[sid]) > MAX_DETECTIONS_PER_SCENE:
            raise InvalidSubmission(f"more than {MAX_DETECTIONS_PER_SCENE} detections for scene {sid}")
    return per_scene


def _mssd(r_e, t_e, r_g, t_g, pts):
    diff = pts @ (r_e - r_g).T + (t_e - t_g)[None, :]
    return float(np.sqrt((diff ** 2).sum(1)).max())


def _mspd_view(r_e, t_e, r_g, t_g, pts, cam):
    """Maximum 2D distance between projected model points in one image."""
    pe = (pts @ r_e.T + t_e[None, :]) @ cam["R_CW"].T + cam["t_CW"][None, :]
    pg = (pts @ r_g.T + t_g[None, :]) @ cam["R_CW"].T + cam["t_CW"][None, :]
    if np.any(pe[:, 2] <= 1e-6) or np.any(pg[:, 2] <= 1e-6):
        return math.inf
    k = cam["K"]
    du = k[0, 0] * (pe[:, 0] / pe[:, 2] - pg[:, 0] / pg[:, 2])
    dv = k[1, 1] * (pe[:, 1] / pe[:, 2] - pg[:, 1] / pg[:, 2])
    return float(np.sqrt(du * du + dv * dv).max())


def ap101(tp, n_gt):
    if n_gt <= 0:
        return None
    if len(tp) == 0:
        return 0.0
    tp = np.asarray(tp, dtype=bool)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    recall = ctp / float(n_gt)
    precision = ctp / np.maximum(ctp + cfp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    levels = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, levels, side="left")
    vals = np.where(idx < len(recall), envelope[np.minimum(idx, len(recall) - 1)], 0.0)
    return float(vals.mean())


def prepare_gt(gt):
    """Convert the ground-truth dictionary into numpy structures."""
    objects = {oid: (np.asarray(o["eval_points"], dtype=np.float64), float(o["diameter"]))
               for oid, o in gt["objects"].items()}
    scenes = {}
    for sc in gt["scenes"]:
        cams = [{"K": np.asarray(c["K"], dtype=np.float64), "R_CW": np.asarray(c["R_CW"], dtype=np.float64),
                 "t_CW": np.asarray(c["t_CW"], dtype=np.float64)} for c in sc["cameras"]]
        targets = [{"obj_id": t["obj_id"], "R": np.asarray(t["R"], dtype=np.float64),
                    "t": np.asarray(t["t"], dtype=np.float64), "visib": list(t["visib"])} for t in sc["targets"]]
        scenes[sc["scene_id"]] = {"scene_id": sc["scene_id"], "dataset": sc["dataset"], "cameras": cams,
                                  "targets": targets, "width": int(sc["image_size"][0])}
    libraries = {}
    for dataset, info in gt["datasets"].items():
        for sid in info["scenes"]:
            libraries[sid] = set(info["objects"])
    return {"objects": objects, "scenes": scenes, "datasets": gt["datasets"], "libraries": libraries}


def score_dataset(prepared, dataset, dets_by_scene):
    """Per-image BOP evaluation of world-frame scene detections.

    Every scene detection is an estimate in every view of the scene (its pose
    expressed in that camera).  A ground-truth instance is an (image, object)
    pair with visible fraction >= 0.1 in that image; estimates of an object that
    is present but not visible in an image are ignored there.
    """
    info = prepared["datasets"][dataset]
    objects = prepared["objects"]
    library = list(info["objects"])
    ap_entries = {oid: [] for oid in library}
    n_gt = {oid: 0 for oid in library}
    per_target = []
    for s_order, sid in enumerate(info["scenes"]):
        scene = prepared["scenes"][sid]
        width = scene["width"]
        mspd_thr = MSPD_PIXELS_AT_640 * width / 640.0
        gt_by_obj = {t["obj_id"]: t for t in scene["targets"]}
        dets = sorted(dets_by_scene.get(sid, []), key=lambda d: (-d[0], d[1]))
        # errors of the highest-scoring estimate of each object present in the scene
        top = {}
        seen = set()
        for score, idx, oid, rot, trans in dets:
            if oid in seen:
                continue
            seen.add(oid)
            gt_t = gt_by_obj.get(oid)
            if gt_t is None:
                continue
            views = [v for v, vis in enumerate(gt_t["visib"]) if vis >= VISIB_MIN]
            if not views:
                continue
            pts, diam = objects[oid]
            e_mssd = _mssd(rot, trans, gt_t["R"], gt_t["t"], pts) / diam
            e_mspd = {v: _mspd_view(rot, trans, gt_t["R"], gt_t["t"], pts, scene["cameras"][v]) for v in views}
            top[oid] = (idx, e_mssd, e_mspd)
        for v in range(len(scene["cameras"])):
            for score, idx, oid, rot, trans in dets:
                gt_t = gt_by_obj.get(oid)
                if gt_t is not None and gt_t["visib"][v] < VISIB_MIN:
                    continue
                if oid in top and top[oid][0] == idx:
                    ap_entries[oid].append((score, s_order, v, idx, "cand", top[oid][1], top[oid][2][v]))
                else:
                    ap_entries[oid].append((score, s_order, v, idx, "fp", None, None))
            for oid, gt_t in gt_by_obj.items():
                if gt_t["visib"][v] < VISIB_MIN:
                    continue
                n_gt[oid] += 1
                if oid in top:
                    e_mssd, e_mspd = top[oid][1], top[oid][2][v]
                    rec_mssd = float(np.mean(e_mssd < MSSD_FRACTIONS))
                    rec_mspd = float(np.mean(e_mspd < mspd_thr))
                else:
                    e_mssd = e_mspd = None
                    rec_mssd = rec_mspd = 0.0
                per_target.append({"scene_id": sid, "view": v, "obj_id": oid, "recall_mssd": rec_mssd,
                                   "recall_mspd": rec_mspd, "mssd_rel": e_mssd, "mspd_px": e_mspd})
    if per_target:
        ar_mssd = float(np.mean([p["recall_mssd"] for p in per_target]))
        ar_mspd = float(np.mean([p["recall_mspd"] for p in per_target]))
    else:
        ar_mssd = ar_mspd = 0.0
    width = prepared["scenes"][info["scenes"][0]]["width"] if info["scenes"] else 640
    mspd_thr = MSPD_PIXELS_AT_640 * width / 640.0
    ap_m, ap_p = [], []
    for oid in library:
        if n_gt[oid] == 0:
            continue
        entries = sorted(ap_entries[oid], key=lambda e: (-e[0], e[1], e[2], e[3]))
        vals_m, vals_p = [], []
        for thr in MSSD_FRACTIONS:
            vals_m.append(ap101([e[4] == "cand" and e[5] < thr for e in entries], n_gt[oid]))
        for thr in mspd_thr:
            vals_p.append(ap101([e[4] == "cand" and e[6] < thr for e in entries], n_gt[oid]))
        ap_m.append(float(np.mean(vals_m)))
        ap_p.append(float(np.mean(vals_p)))
    ap_mssd = float(np.mean(ap_m)) if ap_m else 0.0
    ap_mspd = float(np.mean(ap_p)) if ap_p else 0.0
    metrics = {"AR": 100.0 * 0.5 * (ar_mssd + ar_mspd), "AP": 100.0 * 0.5 * (ap_mssd + ap_mspd)}
    details = {"AR_MSSD": 100.0 * ar_mssd, "AR_MSPD": 100.0 * ar_mspd, "AP_MSSD": 100.0 * ap_mssd,
               "AP_MSPD": 100.0 * ap_mspd, "n_targets": len(per_target),
               "n_detections": int(sum(len(dets_by_scene.get(s, [])) for s in info["scenes"])),
               "per_target": per_target,
               "ap_entries": {oid: [[e[0], e[1], e[4], e[5], e[6]]
                                    for e in sorted(ap_entries[oid], key=lambda e: (-e[0], e[1], e[2], e[3]))]
                              for oid in library},
               "n_gt": n_gt, "scenes": list(info["scenes"]), "mspd_thresholds": [float(x) for x in mspd_thr]}
    return metrics, details
