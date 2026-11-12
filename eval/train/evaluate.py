"""Visible train-split evaluation.

Run with no arguments (``bash scripts/evaluate_train.sh``).  Every submission
found on disk (``submissions/*.sh``) is run on its own fresh copy of the visible
train inputs (``data/train``) as ``bash submissions/<name>.sh INPUT_DIR OUTPUT_JSON``.
The raw detections are validated and scored against the train labels in
``eval/train/train_gt.json``; results go to ``scoring/scores_train.json``.

The protocol, validity rules and metrics are identical to the hidden test; the
train split uses different scenes and a disjoint object library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from eval.train import scoring

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ("textureless_clutter", "textured_household", "reflective_industrial")
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


def run_cached(script, name, input_dir, output_path, log_path, resume_dir):
    """Run a submission; with ``resume_dir`` reuse a finished run of the same script."""
    if resume_dir is None:
        return run_submission(script, input_dir, output_path, log_path)
    key = hashlib.sha1(Path(script).read_bytes()).hexdigest()[:16]
    store = Path(resume_dir) / "outputs"
    store.mkdir(parents=True, exist_ok=True)
    code_file = store / f"{name}.{key}.code"
    if code_file.is_file():
        code = code_file.read_text().strip()
        code = int(code) if code.lstrip("-").isdigit() else code
        if (store / f"{name}.{key}.json").is_file():
            shutil.copy(store / f"{name}.{key}.json", output_path)
        if (store / f"{name}.{key}.log").is_file():
            shutil.copy(store / f"{name}.{key}.log", log_path)
        return code, 0.0
    code, elapsed = run_submission(script, input_dir, output_path, log_path)
    if Path(output_path).is_file():
        shutil.copy(output_path, store / f"{name}.{key}.json")
    shutil.copy(log_path, store / f"{name}.{key}.log")
    code_file.write_text(str(code))
    return code, elapsed


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
    return {exp: scoring.score_dataset(prepared, exp, dets) for exp in EXPERIMENTS}, None


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(ROOT / "data" / "train"))
    parser.add_argument("--labels", default=str(ROOT / "eval" / "train" / "train_gt.json"))
    parser.add_argument("--submissions-dir", default=str(ROOT / "submissions"))
    parser.add_argument("--output", default=str(ROOT / "scoring" / "scores_train.json"))
    parser.add_argument("--details", default=None, help="optional per-target diagnostics")
    parser.add_argument("--save-predictions", default=None, help="optional directory to keep raw outputs and logs")
    parser.add_argument("--resume-dir", default=None,
                        help="optional cache so an interrupted evaluation resumes (never used for official scoring)")
    args = parser.parse_args()

    subs = discover_submissions(args.submissions_dir)
    if not subs:
        raise SystemExit(f"no submissions found under {args.submissions_dir}")
    prepared = scoring.prepare_gt(json.loads(Path(args.labels).read_text()))
    work = Path(tempfile.mkdtemp(prefix="mvpose_train_"))
    try:
        scores = {exp: {} for exp in EXPERIMENTS}
        details = {}
        for name, script in subs.items():
            inp = work / f"inputs_{name}"
            shutil.copytree(args.inputs, inp)
            out = work / f"pred_{name}.json"
            log = work / f"log_{name}.txt"
            code, elapsed = run_cached(script, name, inp, out, log, args.resume_dir)
            results, error = score_submission(prepared, out, code)
            if args.save_predictions:
                keep = Path(args.save_predictions)
                keep.mkdir(parents=True, exist_ok=True)
                if out.is_file():
                    shutil.copy(out, keep / f"{name}.json")
                shutil.copy(log, keep / f"{name}.log")
            shutil.rmtree(inp, ignore_errors=True)
            if results is None:
                print(f"[train] {name}: FAILED ({error}); scored with worst values\n{tail(log)}", flush=True)
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
                print(f"[train] {name}: {summary} ({elapsed:.0f}s)\n{tail(log, 5)}", flush=True)
        payload = {"experiments": {exp: {"results": scores[exp]} for exp in EXPERIMENTS}}
        write_json(args.output, payload)
        if args.details:
            write_json(args.details, details)
        print(json.dumps(payload, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
