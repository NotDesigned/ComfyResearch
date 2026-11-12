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
import math
import os
import re
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
    def __init__(self, metrics) -> None:
        self.metrics = metrics
        self.written: dict[str, int] = {}
        self.records = 0

    def series(self, name: str, values, steps) -> None:
        if not isinstance(values, list):
            return
        aligned = isinstance(steps, list) and len(steps) == len(values)
        for i in range(self.written.get(name, 0), len(values)):
            step = steps[i] if aligned else i
            for key, number in scalars(name, values[i]):
                record = {"name": key, "value": number, "step": int(step)}
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
    output = Path(os.environ["OUTPUT_DIR"])
    # train_cli caches checkpoints under ./.comfyresearch; the code mount is read-only.
    work = output / "comfyresearch-work"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(CODE), MPLBACKEND="Agg")

    last_metrics = None
    terminal = None
    with open(os.environ["METRICS_PATH"], "a") as metrics, \
            open(output / "events.ndjson", "w") as events, \
            open(request, "rb") as stdin:
        translator = Translator(metrics)
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "comfy_research.remote.train_cli"],
            cwd=work, env=env, stdin=stdin, stdout=subprocess.PIPE,
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
                print(f"step {ev.get('step')}: {translator.records} metric records", flush=True)
            else:
                if kind == "complete":
                    translator.event(ev)
                events.write(json.dumps(ev) + "\n")
                events.flush()
                print(f"event {kind}", flush=True)
                if kind in {"complete", "aborted", "paused", "error"}:
                    terminal = kind
        rc = proc.wait()

    if last_metrics is not None:
        (output / "final_metrics.json").write_text(json.dumps(last_metrics))
    print(f"train_cli exit {rc}, terminal event {terminal}, {translator.records} records", flush=True)
    if rc != 0:
        return rc
    return 0 if terminal == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
