"""Held-out test evaluation.

Run with no arguments (``bash scripts/evaluate_test.sh``).  The evaluator

1. generates the hidden test scenes at test time (images, calibrated cameras,
   single-view candidates and object meshes) into a temporary directory; the
   ground truth stays in this process,
2. runs every submission found on disk (``submissions/*.sh``), each on its own
   fresh copy of the inputs, as ``bash submissions/<name>.sh INPUT_DIR OUTPUT_JSON``,
3. validates the raw detections and recomputes every metric from them,
4. writes ``scoring/scores_test.json``.

A submission that crashes, times out, writes nothing, or writes any invalid
value (non-finite numbers, non-rotation matrices, unknown scene/object ids,
too many detections) is scored as failed: AR = AP = 0 in every experiment.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from eval.test import scenegen, scoring

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = scenegen.DATASETS
METHOD_TIMEOUT_S = 4 * 3600


def discover_submissions(sub_dir):
    subs = {}
    for path in sorted(Path(sub_dir).glob("*.sh")):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
        if not name or name.startswith("_"):
            continue
        if name in subs:
            raise ValueError(f"two submissions map to the name {name!r}")
        subs[name] = path.resolve()
    return subs


def run_submission(script, input_dir, output_path, log_path):
    start = time.time()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(["bash", str(script), str(input_dir), str(output_path)], cwd=str(ROOT),
                                  stdout=log, stderr=subprocess.STDOUT, timeout=METHOD_TIMEOUT_S)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return code, time.time() - start


def tail(path, n=25):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def score_submission(prepared, output_path, exit_code):
    if exit_code != 0:
        return None, f"submission exited with {exit_code}"
    if not Path(output_path).is_file():
        return None, "submission wrote no output file"
    try:
        payload = json.loads(Path(output_path).read_text())
    except (OSError, ValueError) as exc:
        return None, f"output is not valid JSON: {exc}"
    try:
        dets = scoring.parse_detections(payload, prepared["libraries"])
    except scoring.InvalidSubmission as exc:
        return None, f"invalid output: {exc}"
    results = {}
    for exp in EXPERIMENTS:
        results[exp] = scoring.score_dataset(prepared, exp, dets)
    return results, None


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions-dir", default=str(ROOT / "submissions"))
    parser.add_argument("--output", default=str(ROOT / "scoring" / "scores_test.json"))
    parser.add_argument("--details", default=None, help="optional per-target diagnostics (not ranked)")
    parser.add_argument("--save-predictions", default=None, help="optional directory to keep raw outputs and logs")
    parser.add_argument("--smoke", action="store_true", help="tiny pipeline check; never used for scoring")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    subs = discover_submissions(args.submissions_dir)
    if not subs:
        raise SystemExit(f"no submissions found under {args.submissions_dir}")
    work = Path(tempfile.mkdtemp(prefix="mvpose_test_"))
    try:
        pristine = work / "pristine"
        t0 = time.time()
        gt = scenegen.generate_split("test", pristine, scenes_per_dataset=2 if args.smoke else None,
                                     workers=args.workers)
        prepared = scoring.prepare_gt(gt)
        print(f"[test] generated {len(gt['scenes'])} hidden scenes in {time.time() - t0:.1f}s", flush=True)
        scores = {exp: {} for exp in EXPERIMENTS}
        details = {}
        for name, script in subs.items():
            inp = work / f"inputs_{name}"
            shutil.copytree(pristine, inp)
            out = work / f"pred_{name}.json"
            log = work / f"log_{name}.txt"
            code, elapsed = run_submission(script, inp, out, log)
            results, error = score_submission(prepared, out, code)
            if args.save_predictions:
                keep = Path(args.save_predictions)
                keep.mkdir(parents=True, exist_ok=True)
                if out.is_file():
                    shutil.copy(out, keep / f"{name}.json")
                shutil.copy(log, keep / f"{name}.log")
            shutil.rmtree(inp, ignore_errors=True)
            if results is None:
                print(f"[test] {name}: FAILED ({error}); scored with worst values\n{tail(log)}", flush=True)
                for exp in EXPERIMENTS:
                    scores[exp][name] = scoring.worst_metrics()
                details[name] = {"status": "failed", "error": error, "seconds": elapsed}
            else:
                for exp in EXPERIMENTS:
                    scores[exp][name] = results[exp][0]
                details[name] = {"status": "ok", "seconds": elapsed,
                                 "experiments": {exp: results[exp][1] for exp in EXPERIMENTS}}
                summary = ", ".join(f"{exp}: AR {results[exp][0]['AR']:.2f} AP {results[exp][0]['AP']:.2f}"
                                    for exp in EXPERIMENTS)
                print(f"[test] {name}: {summary} ({elapsed:.0f}s)\n{tail(log, 5)}", flush=True)
            if args.details:
                write_json(args.details, details)
                write_json(args.output, {"experiments": {exp: {"results": scores[exp]} for exp in EXPERIMENTS}})
        payload = {"experiments": {exp: {"results": scores[exp]} for exp in EXPERIMENTS}}
        write_json(args.output, payload)
        if args.details:
            write_json(args.details, details)
        print(json.dumps(payload, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
