#!/usr/bin/env python3
"""Run one ComfyResearch TrainRequest as an ML-Expd job.

ML-Expd mounts the code read-only and gives the job OUTPUT_DIR (archived as the
Run's results) and METRICS_PATH (records published to W&B and MLflow). This
wrapper runs ``comfy_research.remote.train_cli`` unchanged and translates its
NDJSON stream:

* each scalar point in the cumulative histories of a ``metrics`` event becomes
  one ML-Expd metric record, written once;
* every non-``metrics`` event is kept in ``OUTPUT_DIR/events.ndjson``, and the
  last ``metrics`` event (the full histories, including Observable
  visualisations) in ``OUTPUT_DIR/final_metrics.json``.

Usage inside the job: ``python3 run_graph.py [request.json]``.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_./-]+")


def metric_name(*parts: str) -> str:
    name = "/".join(NAME_UNSAFE.sub("_", p).strip("_") or "_" for p in parts)
    return name[:128]


def scalars(name: str, value):
    """Yield (name, number) for a history element: a number, a list or a dict of numbers."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            yield name, float(value)
    elif isinstance(value, list):
        for j, item in enumerate(value):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                yield from scalars(metric_name(name, str(j)), item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                yield from scalars(metric_name(name, str(key)), item)


class Translator:
    def __init__(self, metrics, units: dict[str, str] | None = None) -> None:
        self.metrics = metrics
        self.units = units or {}
        self.written: dict[str, int] = {}
        self.records = 0

    def series(self, name: str, values, steps) -> None:
        if not isinstance(values, list):
            return
        aligned = isinstance(steps, list) and len(steps) == len(values)
        for i in range(self.written.get(name, 0), len(values)):
            step = steps[i] if aligned else i
            for key, number in scalars(name, values[i]):
                # "value" is an explicit generic numeric coordinate, not a claim of
                # physical dimensionlessness. Scientific units can be supplied by name.
                unit = self.units.get(
                    key, "loss" if key.startswith("loss/") else "value"
                )
                record = {"name": key, "value": number, "unit": unit, "step": int(step)}
                self.metrics.write(json.dumps(record) + "\n")
                self.records += 1
        self.written[name] = len(values)
        self.metrics.flush()

    def event(self, ev: dict) -> None:
        steps = ev.get("step_ticks")
        self.series("loss/train", ev.get("loss_history"), steps)
        self.series("loss/test", ev.get("test_loss_history"), steps)
        self.series("loss/reg", ev.get("reg_loss_history"), steps)
        for key, history in (ev.get("observable_metric_histories") or {}).items():
            self.series(metric_name("obs", *str(key).split("::")), history, steps)


def main() -> int:
    request = CODE / (sys.argv[1] if len(sys.argv) > 1 else "request.json")
    request_bytes = request.read_bytes()
    request_hash = hashlib.sha256(request_bytes).hexdigest()
    request_body = json.loads(request_bytes)
    output = Path(os.environ["OUTPUT_DIR"])
    # train_cli caches checkpoints under ./.comfyresearch; the code mount is read-only.
    work = output / "comfyresearch-work"
    work.mkdir(parents=True, exist_ok=True)
    for relative, source in ((".comfyresearch/artifacts", CODE / "input-artifacts"),):
        if source.is_dir():
            shutil.copytree(source, work / relative, dirs_exist_ok=True)
    effective_request = work / "request.json"
    effective_request.write_text(
        json.dumps(resolve_dataset_paths(request_body)), encoding="utf-8"
    )
    env = dict(os.environ, PYTHONPATH=str(CODE), MPLBACKEND="Agg")

    last_metrics = None
    complete = None
    terminal = None
    with (
        open(os.environ["METRICS_PATH"], "a") as metrics,
        open(output / "events.ndjson", "w") as events,
        open(effective_request, "rb") as stdin,
    ):
        translator = Translator(metrics, request_body.get("ml_exp_metric_units"))
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "comfy_research.remote.train_cli"],
            cwd=work,
            env=env,
            stdin=stdin,
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                ev = json.loads(raw)
            except ValueError:
                print(f"non-JSON line from train_cli: {raw[:200]!r}", file=sys.stderr)
                continue
            kind = ev.get("type")
            if kind == "metrics":
                translator.event(ev)
                last_metrics = ev
                print(
                    f"step {ev.get('step')}: {translator.records} metric records",
                    flush=True,
                )
            else:
                if kind == "complete":
                    translator.event(ev)
                    complete = ev
                events.write(json.dumps(ev) + "\n")
                events.flush()
                print(f"event {kind}", flush=True)
                if kind in {"complete", "aborted", "paused", "error"}:
                    terminal = kind
        rc = proc.wait()

    if last_metrics is not None:
        (output / "final_metrics.json").write_text(json.dumps(last_metrics))
    if complete is not None and rc == 0 and terminal == "complete":
        (output / "comfyresearch-result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "request_sha256": request_hash,
                    "trainer_node_id": request_body["trainer_node_id"],
                    "complete": complete,
                }
            ),
            encoding="utf-8",
        )
    print(
        f"train_cli exit {rc}, terminal event {terminal}, {translator.records} records",
        flush=True,
    )
    if rc != 0:
        return rc
    return 0 if terminal == "complete" else 1


def resolve_dataset_paths(value):
    """Explicit ml-exp-data://NAME/path references use verified mounted assets."""
    if isinstance(value, str) and value.startswith("ml-exp-data://"):
        name, _, relative = value[len("ml-exp-data://") :].partition("/")
        datasets = json.loads(os.environ.get("DATASETS_JSON", "{}"))
        if name not in datasets:
            raise ValueError(f"Dataset binding {name} is not available")
        root = Path(datasets[name]).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Dataset path escapes its mounted asset")
        return str(path)
    if isinstance(value, dict):
        return {key: resolve_dataset_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_dataset_paths(item) for item in value]
    return value


if __name__ == "__main__":
    sys.exit(main())
