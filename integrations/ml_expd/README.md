# Run ComfyResearch graphs on ML-Expd

[ML-Expd](https://github.com/NotDesigned/ml-experiment-control) schedules jobs on
PAI DLC and Slurm, keeps each Run's code, environment and data identities, and
publishes metrics to W&B and MLflow. This directory runs a saved ComfyResearch
graph there as a batch job instead of over a live SSH session.

## Use

```bash
python integrations/ml_expd/pack.py data/graph_library/templates/edge-of-stability-cpu.json ./stage --device cpu
ml-exp submit ./stage --executor pai-native-cpu --gpus 0 --time 30m -- python3 -u run_graph.py
ml-exp status
ml-exp results        # events.ndjson, final_metrics.json, metrics.jsonl
```

`pack.py` stages the `comfy_research` package, `requirements.txt`, the
environment (`Dockerfile.cpu`), `run_graph.py` and the graph as `request.json`.
Only the Dockerfile and `requirements.txt` define the environment, so a changed
graph or code change reuses the built image.

## What the job does

`run_graph.py` runs `python -m comfy_research.remote.train_cli` unchanged, with
`request.json` on stdin, in a writable directory under `OUTPUT_DIR` (the code
mount is read-only). From its NDJSON stream it writes:

| Output | Contents |
| --- | --- |
| `METRICS_PATH` | One record per new scalar point: `loss/train`, `loss/test`, `loss/reg`, and `obs/<observable id>/<index or key>` from `observable_metric_histories` |
| `OUTPUT_DIR/events.ndjson` | Every event other than `metrics` (session, phases, `complete`, errors) |
| `OUTPUT_DIR/final_metrics.json` | The last `metrics` event: full histories and Observable visualisation updates |

The job fails unless `train_cli` exits 0 with a `complete` event.

## Limits

- Batch, not interactive: pause and resume are not wired; `ml-exp cancel` stops the job.
- The workbench does not read the results back yet; open them from `ml-exp results`, W&B or MLflow.
- `Dockerfile.cpu` installs CPU PyTorch. A GPU executor needs a CUDA image and `--device cuda`.
