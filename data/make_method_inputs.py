"""Create method-visible synthetic benchmark inputs without evaluation labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def strip_labels(source_path: Path, output_path: Path) -> None:
    payload = json.loads(source_path.read_text())
    for scene in payload.get("scenes", []):
        scene.pop("T_WO_gt", None)
        for view in scene.get("views", []):
            for candidate in view.get("candidates", []):
                candidate.pop("T_WO", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/synthetic_benchmark.json")
    parser.add_argument("--output", default="data/synthetic_inputs.json")
    args = parser.parse_args()
    strip_labels(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()
