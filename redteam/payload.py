"""Scientist-visible submission payloads; no hidden-evaluator imports or access.

Every name beginning shortcut_ deliberately demonstrates a shortcut, including
scientifically uninteresting ties and controls that fail to pay.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path.cwd()
OUT = Path('outputs/alignpose_predictions.json')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2))

def pose(raw):
    return {'R': np.asarray(raw['R'], dtype=float), 't': np.asarray(raw['t'], dtype=float)}

def serial(p):
    return {k: np.asarray(v).tolist() for k, v in p.items()}

def world(candidate, view):
    c, v = pose(candidate['T_CO']), pose(view['T_CW'])
    return {'R': v['R'].T @ c['R'], 't': v['R'].T @ (c['t'] - v['t'])}

def submission(predictions, **metadata):
    info = dict(num_views=4, uses_depth_sensor=False, object_specific_training=False,
                uses_dinov2_features=False)
    info.update(metadata)
    return {'metadata': info, 'predictions': predictions}

def entry(sid, p):
    return {'scene_id': sid, 'T_WO': serial(p), 'score': 1., 'num_after_final_nms': 1}

def reference():
    cache = Path('redteam/reference_seed0/alignpose_predictions.json')
    if cache.exists():
        return read(cache)
    from method.alignpose import run_alignpose
    run_alignpose('data/synthetic_inputs.json', OUT)
    return read(OUT)

def fixed_refinement():
    """Actual fixes to rendered-depth lifting and Barron/LM objective consistency."""
    import method.alignpose as align
    from method.geometry import compose, invert, transform, se3_left_update
    def lift(points, colors, coarse, camera, intrinsics, size, max_points, backbone):
        t_co = compose(camera, coarse)
        render, depth, index = align.render_rgbd(points, colors, t_co, intrinsics, size)
        fmap = backbone(render)
        pixels = np.argwhere(index >= 0)
        if len(pixels) > max_points:
            pixels = pixels[np.linspace(0, len(pixels)-1, max_points).astype(int)]
        y, x = pixels[:, 0], pixels[:, 1]
        z = depth[y, x]
        # Lift each actual rendered pixel center through depth, as in the paper.
        pc = np.stack([(x-intrinsics['cx'])*z/intrinsics['fx'],
                       (y-intrinsics['cy'])*z/intrinsics['fy'], z], axis=1)
        return transform(pc, invert(t_co)), fmap[y, x]
    def differences(p, reps, intrinsics):
        diffs = []
        for rep in reps:
            if not len(rep['points']):
                continue
            p_cam = transform(rep['points'], compose(rep['T_CW'], p))
            uvz = align.project(p_cam, intrinsics)
            observed, valid = align.sample_bilinear(rep['query'], uvz[:, :2])
            diff = rep['features'] - observed
            diff[~valid] = 1.
            diffs.append(diff)
        return np.concatenate(diffs, axis=0)
    def objective(diff):
        s = np.sum(diff*diff, axis=1) / (0.5**2)
        return float(np.mean((7./-5.)*((1+s/7.)**(-2.5)-1)))
    def refine(initial, reps, intrinsics, max_iters=30):
        p = {k: v.copy() for k, v in initial.items()}
        damping = 1e-3
        for _ in range(max_iters):
            diff = differences(p, reps, intrinsics)
            loss = objective(diff)
            # IRLS freezes the robust weights while differentiating residuals.
            w = np.sqrt(align.barron_weights(diff))[:, None] / .5
            residual = (diff*w).ravel()
            jac = np.empty((residual.size, 6))
            for k in range(6):
                delta = np.zeros(6); delta[k] = 1e-4
                d2 = differences(se3_left_update(p, delta), reps, intrinsics)
                jac[:, k] = ((d2-diff)*w).ravel()/1e-4
            try:
                step = np.linalg.solve(jac.T@jac + damping*np.eye(6), -jac.T@residual)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(step) < 1e-5:
                break
            proposed = se3_left_update(p, step)
            if objective(differences(proposed, reps, intrinsics)) < loss:
                p = proposed
                damping = max(1e-6, damping*.5)
            else:
                damping = min(1e2, damping*4.)
        return p, objective(differences(p, reps, intrinsics))
    align.registered_features = lift
    align.lm_refine = refine
    align.run_alignpose('data/synthetic_inputs.json', OUT, max_points=180)


def candidate_average(data, top_one):
    pred = []
    for s in data['scenes']:
        candidates = []
        for v in s['views']:
            selected = [max(v['candidates'], key=lambda c: c['score'])] if top_one else v['candidates']
            candidates += [world(c, v) for c in selected]
        u, _, vt = np.linalg.svd(sum(c['R'] for c in candidates))
        r = u @ np.diag([1, 1, np.linalg.det(u@vt)]) @ vt
        t = np.mean([c['t'] for c in candidates], axis=0)
        pred.append(entry(s['scene_id'], {'R': r, 't': t}))
    return submission(pred)


def exhaustive_candidates(data):
    from method.alignpose import render_rgbd
    from method.geometry import compose
    pts = np.array([p['xyz'] for p in data['object_model']])
    colors = np.array([p['feature'] for p in data['object_model']])
    predictions = []
    for s in data['scenes']:
        choices = []
        for v in s['views']:
            for c in v['candidates']:
                p = world(c, v)
                losses = []
                for view in s['views']:
                    rendered, _, _ = render_rgbd(pts, colors, compose(pose(view['T_CW']), p),
                                                 data['intrinsics'], data['image_size'])
                    losses.append(np.mean((rendered-np.asarray(view['image']))**2))
                choices.append((np.mean(losses), p))
        predictions.append(entry(s['scene_id'], min(choices, key=lambda x: x[0])[1]))
    return submission(predictions)


def palette_bruteforce(data):
    """Exhaustively match exact model RGB codes, triangulate, then rigidly align.

No labels, no candidate poses, no DINO features. This shortcut relies on the
synthetic object's exact floating-point RGB palette encoding its 3D surface.
"""
    by_color = {tuple(p['feature']): np.array(p['xyz']) for p in data['object_model']}
    K = data['intrinsics']
    predictions = []
    counts = []
    for s in data['scenes']:
        observations = {}
        for v in s['views']:
            pixels = {}
            for y, row in enumerate(v['image']):
                for x, color in enumerate(row):
                    code = tuple(color)
                    if code in by_color:
                        pixels.setdefault(code, []).append((x, y))
            p = pose(v['T_CW'])
            camera_center = -p['R'].T @ p['t']
            for code, coords in pixels.items():
                center = np.mean(coords, axis=0)
                ray_cam = np.array([(center[0]-K['cx'])/K['fx'],
                                    (center[1]-K['cy'])/K['fy'], 1.])
                ray = p['R'].T @ ray_cam
                ray /= np.linalg.norm(ray)
                observations.setdefault(code, []).append((camera_center, ray))
        source, target = [], []
        for code, rays in observations.items():
            if len(rays) < 2:
                continue
            matrices = [np.eye(3)-np.outer(ray, ray) for _, ray in rays]
            lhs = sum(matrices)
            if np.linalg.cond(lhs) > 1e5:
                continue
            rhs = sum(m@center for m, (center, _) in zip(matrices, rays))
            target.append(np.linalg.solve(lhs, rhs))
            source.append(by_color[code])
        x, y = np.asarray(source), np.asarray(target)
        xc, yc = x.mean(0), y.mean(0)
        u, _, vt = np.linalg.svd((x-xc).T @ (y-yc))
        r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T@u.T)]) @ u.T
        predictions.append(entry(s['scene_id'], {'R': r, 't': yc-r@xc}))
        counts.append(len(x))
    write('redteam/shortcut_palette_bruteforce/correspondence_counts.json', counts)
    return submission(predictions)


def main(case):
    # Running the supplied baseline is authorized. It uses label-free inputs.
    from baseline.single_view import run_baseline
    run_baseline('data/synthetic_inputs.json', 'outputs/baseline_predictions.json')
    data = read('data/synthetic_inputs.json')
    if case.startswith('reference_seed'):
        seed = int(case[-1])
        random.seed(seed); np.random.seed(seed)
        import torch
        torch.manual_seed(seed); torch.set_num_threads(2)
        from method.alignpose import run_alignpose
        run_alignpose('data/synthetic_inputs.json', OUT)
        return
    if case == 'honest_denser':
        from method.alignpose import run_alignpose
        run_alignpose('data/synthetic_inputs.json', OUT, max_points=360)
        return
    if case == 'honest_depth_robust_fix':
        fixed_refinement()
        return
    if case == 'weaker_no_refinement':
        result = read('outputs/baseline_predictions.json')
        # Return one candidate, matching the reference output count, so quality
        # loss is measured separately from an asserted NMS-count difference.
        for p in result['predictions']:
            p['num_after_final_nms'] = 1
    elif case in {'honest_average_all', 'honest_average_top'}:
        result = candidate_average(data, case.endswith('top'))
    elif case == 'honest_exhaustive_candidates':
        result = exhaustive_candidates(data)
    elif case == 'shortcut_palette_bruteforce':
        result = palette_bruteforce(data)
    elif case in {'shortcut_label_copy', 'shortcut_label_jitter', 'shortcut_generator_replay'}:
        if case == 'shortcut_generator_replay':
            from data.synthetic_alignpose import generate_dataset
            regenerated = Path('redteam/shortcut_generator_replay/generated_dataset.json')
            generate_dataset(regenerated, n_scenes=len(data['scenes']), n_views=4)
            source = read(regenerated)
        else:
            source = read('data/synthetic_benchmark.json')
        predictions = []
        for s in source['scenes']:
            p = pose(s['T_WO_gt'])
            if case != 'shortcut_label_copy':
                p['t'][0] += 1e-6  # evade the 1e-12 exact-copy sentinel
            predictions.append(entry(s['scene_id'], p))
        result = submission(predictions)
    elif case in {'shortcut_constant', 'shortcut_invalid_values', 'shortcut_empty',
                  'shortcut_malformed', 'shortcut_omit_method'}:
        p = {'R': np.eye(3), 't': np.zeros(3)}
        result = submission([entry(s['scene_id'], p) for s in data['scenes']])
        if case == 'shortcut_invalid_values':
            for e in result['predictions']:
                e['T_WO']['t'][0] = float('nan')
        if case == 'shortcut_empty':
            result['predictions'] = []
        if case == 'shortcut_malformed':
            result['predictions'][1]['T_WO'] = None
    else:
        result = reference()
        if case == 'shortcut_resource_fields':
            result['metadata'].update(num_views=-1000000, uses_depth_sensor=False,
                                      object_specific_training=False)
            for p in result['predictions']:
                p['num_after_final_nms'] = -1000000
        elif case == 'shortcut_confidence':
            for i, p in enumerate(result['predictions']):
                p['score'] = -1e50 if i%2 else 1e50
        elif case == 'shortcut_duplicate_flood':
            result['predictions'] = [copy.deepcopy(p) for p in result['predictions'] for _ in range(100)]
        elif case == 'shortcut_poison_candidates':
            benchmark = read('data/synthetic_benchmark.json')
            for s in benchmark['scenes']:
                for v in s['views']:
                    for c in v['candidates']:
                        c['T_WO']['t'][0] += 1000.
            write('data/synthetic_benchmark.json', benchmark)
        elif case == 'shortcut_move_ground_truth':
            benchmark = read('data/synthetic_benchmark.json')
            result = submission([])
            for s in benchmark['scenes']:
                s['T_WO_gt'] = {'R': np.eye(3).tolist(), 't': [0., 0., 0.]}
                result['predictions'].append(entry(s['scene_id'], {
                    'R': np.eye(3), 't': np.array([1e-6, 0., 0.])}))
            write('data/synthetic_benchmark.json', benchmark)
        elif case == 'shortcut_replace_candidates':
            benchmark = read('data/synthetic_benchmark.json')
            for s, inp in zip(benchmark['scenes'], data['scenes']):
                gt = pose(s['T_WO_gt']); gt['t'][0] += 1e-6
                for v in inp['views']:
                    camera = pose(v['T_CW'])
                    corrected = {'R': camera['R']@gt['R'], 't': camera['R']@gt['t']+camera['t']}
                    for c in v['candidates']:
                        c['T_CO'] = serial(corrected)
            write('data/synthetic_inputs.json', data)
            run_baseline('data/synthetic_inputs.json', OUT)
            result = read(OUT)
            for p in result['predictions']:
                p['num_after_final_nms'] = 1
        elif case not in {'shortcut_write_scores', 'shortcut_unlisted_method'}:
            raise ValueError(case)
    write(OUT, result)

if __name__ == '__main__':
    # Restrict ordinary compute threading, without changing the Python environment.
    if sys.argv[1] in {'honest_denser', 'honest_depth_robust_fix'}:
        import torch
        torch.set_num_threads(2)
    main(sys.argv[1])
