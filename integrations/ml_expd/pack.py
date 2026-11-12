#!/usr/bin/env python3
"""Stage a ComfyResearch graph as an ML-Expd submission directory.

    python integrations/ml_expd/pack.py GRAPH.json STAGE_DIR [--trainer ID] [--device cpu]
    ml-exp submit STAGE_DIR --executor pai-native-cpu --gpus 0 --time 30m -- python3 run_graph.py

GRAPH.json is a saved graph: a library entry with a ``document``, or the
document itself with ``nodes`` and ``edges``. The stage holds only the Python
package, the environment files and the request, not the docs, frontend or data,
so ML-Expd reuses the environment when only the graph or the code changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", "tests", ".pytest_cache"}


def request_from(graph: dict, trainer: str | None, device: str | None) -> dict:
    document = graph.get("document", graph)
    nodes, edges = document["nodes"], document["edges"]
    trainers = [n["id"] for n in nodes if n.get("type") == "trainer"]
    if trainer is None:
        if len(trainers) != 1:
            sys.exit(f"pass --trainer: the graph has trainers {trainers}")
        trainer = trainers[0]
    elif trainer not in trainers:
        sys.exit(f"{trainer} is not a trainer in this graph: {trainers}")
    if device is not None:
        for node in nodes:
            if node["id"] == trainer:
                node.setdefault("data", {})["computeDevice"] = device
    return {"trainer_node_id": trainer, "nodes": nodes, "edges": edges}


def copy_package(target: Path) -> int:
    count = 0
    for path in sorted((REPO / "comfy_research").rglob("*")):
        rel = path.relative_to(REPO)
        if any(part in SKIP_DIRS for part in rel.parts) or path.suffix == ".pyc":
            continue
        if path.is_file():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / rel)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--trainer", help="Trainer node ID when the graph has several")
    parser.add_argument("--device", help="Override the Trainer computeDevice, e.g. cpu or cuda")
    args = parser.parse_args()

    if args.stage.exists() and any(args.stage.iterdir()):
        sys.exit(f"{args.stage} is not empty")
    args.stage.mkdir(parents=True, exist_ok=True)
    request = request_from(json.loads(args.graph.read_text()), args.trainer, args.device)
    files = copy_package(args.stage)
    shutil.copy2(REPO / "requirements.txt", args.stage / "requirements.txt")
    shutil.copy2(HERE / "Dockerfile.cpu", args.stage / "Dockerfile")
    shutil.copy2(HERE / "run_graph.py", args.stage / "run_graph.py")
    (args.stage / "request.json").write_text(json.dumps(request, indent=1))
    print(f"staged {files} package files, trainer {request['trainer_node_id']}, in {args.stage}")


if __name__ == "__main__":
    main()
