# Run ComfyResearch graphs on ML-Expd

[ML-Expd](https://github.com/NotDesigned/ml-experiment-control) schedules jobs on
PAI DLC and Slurm, keeps each Run's code, environment and data identities, and
publishes metrics to W&B and MLflow. This directory runs a saved ComfyResearch
graph there as a batch job instead of over a live SSH session.

## Use from the workbench

Install the native `ml-exp` client (0.13.4 or newer) on the machine running the
ComfyResearch API and configure it as that service user:

```bash
ml-exp --url https://your-api.example/ml-expd config set --token-file /private/ml-exp.token
ml-exp check
```

Set `COMFYRESEARCH_ML_EXP_BINARY` if the executable is not on PATH, and
`COMFYRESEARCH_ML_EXP_CONFIG` if its private client configuration is not at the
native client's default `~/.config/ml-exp/client.json`. The configuration must
contain a saved URL and an absolute token-file reference. Browser forms and
graphs never contain tokens or choose executable/configuration paths.

In an ordinary Trainer, choose **execution → ML-Exp**, enter an approved Project,
executor, CPU/GPU count, memory and time limit, then **Submit to ML-Exp**. Zero
GPUs selects the CPU image; a positive GPU count selects the pinned CUDA image.
The graph's local/AutoDL device selection does not choose the remote device.

The server freezes the execution graph, code, dependencies, user definitions and
initial checkpoint files before starting submission. It returns a durable task
ID immediately after staging; image preparation proceeds in the background.
Closing the page stops observing, not the remote job. Reopening its workspace
recovers the task. **New experiment** retains the old binding under **Previous
tasks**. Small/Medium graph exports remove those machine-local bindings.

**Continue submission** advances the same frozen submission after a lost reply or
server restart. It never chooses a new Run or changes its source. Read-only GETs
never submit; status refresh, continuation and cancellation are explicit POSTs.
Each job snapshots its server URL/configuration and always addresses its original
Submission/Attempt, independent of the CLI's latest-submission shortcut.

**Load results** downloads through `ml-exp run download --keep`, which verifies
the archive and retains the remote copy. ComfyResearch reads only its bounded
result document, checks its submitted-request SHA256 and Trainer identity, then
uses the existing checkpoint/curve rendering path. A changed graph must be
restored before loading. Result checkpoints are retained as data; this endpoint
does not unpickle them. Metrics use explicit `loss`/`value` coordinate labels;
scientific units can be provided by metric name in `ml_exp_metric_units` in a
batch request.

Optional **Dataset bindings** map names to uploaded `asset.SHA256` identities.
Use `ml-exp-data://NAME/relative/path` in graph path fields to resolve them from
the worker's `DATASETS_JSON`. Ordinary workstation paths are not uploaded.

API endpoints are under `/api/train/ml-exp`: `GET /connection`, `POST /jobs`,
`GET /jobs/{id}`, and `POST /jobs/{id}/{continue,refresh,cancel,results}`.
Job sources and private recovery records live under `.comfyresearch/ml-exp/`
(override with `COMFYRESEARCH_ML_EXP_STATE_DIR`); retain this directory with the
workspace to recover after a restart. Like the existing notebook and SSH
execution APIs, these endpoints assume a trusted workbench deployment.

## Use from the command line

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
- The workbench submits one ordinary Trainer and one parameter combination at a
  time. CRL Trainers, automatic sweeps, graph-assist execution and checkpoint
  continuation are not connected to this backend. A Trainer marked ML-Exp is
  refused by the local `/api/train` path instead of falling back to local compute.
- Status and training step are refreshed while the page is open. Curves,
  high-dimensional visualizations and model checkpoints load on completion;
  continuous Observable streaming is not yet connected to the workbench.
- `pack.py` uses `Dockerfile.cpu`. For a command-line GPU submission replace the
  staged Dockerfile with `Dockerfile.cuda` and pass `--device cuda` while packing.
- The CUDA recipe pins PyTorch 2.5.1 / CUDA 12.4. Actual accelerator and backend
  compatibility must be checked on the selected executor; unit tests do not
  establish provider acceptance.
