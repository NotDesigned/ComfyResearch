import { useReactFlow } from "@xyflow/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { defaultMlExpSettings, graphSignature, MlExpApiError, mlExpApi, mlExpRequest, mlExpRequestId, type MlExpJob, type MlExpSettings } from "../../graph/mlExp";
import type { TrainStreamComplete } from "../../graph/readNdjsonTrainStream";
import { flushCheckpointApplyTrainerVizAndHydrateTv0d } from "../../graph/trainerTrainCompleteCommit";
import type { TrainerNodeData } from "./trainerDefaults";

export function MlExpTrainerPanel({ id, data, update }: {
  id: string;
  data: TrainerNodeData;
  update: (patch: Partial<TrainerNodeData>) => void;
}) {
  const { getNodes, getEdges, setNodes } = useReactFlow();
  const settings = { ...defaultMlExpSettings, ...data.mlExpSettings };
  const [job, setJob] = useState<MlExpJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [connection, setConnection] = useState<string | null>(null);
  const [executors, setExecutors] = useState<{ executor_id: string; limits?: { fixed_resources?: boolean; gpus?: number; cpus?: number; memory_gb?: number } }[]>([]);
  const [busy, setBusy] = useState(false);
  const working = useRef(false);
  const [datasetText, setDatasetText] = useState(JSON.stringify(settings.datasets));
  const jobId = data.mlExpJobId;

  const action = useCallback(async (operation: () => Promise<void>) => {
    if (working.current) return;
    working.current = true;
    setBusy(true);
    setError(null);
    try { await operation(); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { working.current = false; setBusy(false); }
  }, []);

  useEffect(() => {
    if (!jobId) { setJob(null); return; }
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        if (!working.current) {
          const cached = await mlExpApi<MlExpJob>(`/jobs/${jobId}`);
          const current = cached.submission_id
            ? await mlExpApi<MlExpJob>(`/jobs/${jobId}/refresh`, {}) : cached;
          if (!disposed) { setJob(current); setPollError(null); }
          if (current.observation?.publication?.download_available) return;
        }
      } catch (err) {
        if (!disposed) setPollError(err instanceof Error ? err.message : String(err));
      }
      if (!disposed) timer = setTimeout(() => void poll(), 8000);
    };
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [jobId]);

  const patchSettings = (patch: Partial<MlExpSettings>) => update({ mlExpSettings: { ...settings, ...patch } });
  const submit = () => action(async () => {
    if (jobId) throw new Error("This experiment already has a task. Continue it or select New experiment.");
    const request = mlExpRequest(getNodes(), getEdges(), id);
    const signature = await graphSignature(request);
    const newId = mlExpRequestId();
    // Persist before sending. A lost reply leaves the exact recovery identity in the workspace.
    update({ mlExpJobId: newId, mlExpGraphSignature: signature, mlExpAppliedJobId: undefined });
    setJob(await mlExpApi<MlExpJob>("/jobs", { request_id: newId, train: request, options: settings }));
  });

  const loadResults = () => action(async () => {
    if (!jobId) return;
    if (data.mlExpAppliedJobId === jobId) return;
    if (await graphSignature(mlExpRequest(getNodes(), getEdges(), id)) !== data.mlExpGraphSignature) {
      throw new Error("The graph changed after submission. Restore its submitted configuration before loading results.");
    }
    const result = await mlExpApi<{ complete: TrainStreamComplete }>(`/jobs/${jobId}/results`, {});
    if (await graphSignature(mlExpRequest(getNodes(), getEdges(), id)) !== data.mlExpGraphSignature) {
      throw new Error("The graph changed while downloading. Restore its submitted configuration before loading results.");
    }
    await flushCheckpointApplyTrainerVizAndHydrateTv0d({
      isCrl: false, setNodes, getNodes, getEdges, trainerNodeId: id,
      wires: getEdges(), payload: result.complete,
    });
    update({ mlExpAppliedJobId: jobId });
  });

  const numeric = (key: "gpus" | "cpus" | "memory_gb" | "walltime_seconds", label: string, min: number) => (
    <label className="cr-comfy-widget">
      <span className="cr-comfy-widget__label">{label}</span>
      <input type="number" min={min} step={1} className="cr-input cr-comfy-widget__control nodrag nopan"
        aria-label={`ML-Exp ${label}`} value={settings[key]} disabled={Boolean(jobId)}
        onChange={(event) => patchSettings({ [key]: Number(event.target.value) })} />
    </label>
  );

  return <div className="cr-trainer-remote nodrag nopan" aria-label="ML-Exp execution">
    <p className="cr-trainer-remote__hint">Run independently of this page. GPU count selects CPU or CUDA. Credentials use the server's ml-exp connection.</p>
    {(["project", "executor"] as const).map((key) => <label className="cr-comfy-widget" key={key}>
      <span className="cr-comfy-widget__label">{key}</span>
      <input className="cr-input cr-comfy-widget__control nodrag nopan" aria-label={`ML-Exp ${key}`} list={key === "executor" ? `ml-exp-executors-${id}` : undefined}
        value={settings[key]} disabled={Boolean(jobId)} onChange={(event) => {
          const value = event.target.value;
          const row = key === "executor" ? executors.find((item) => item.executor_id === value) : undefined;
          patchSettings({ [key]: value, ...(row?.limits?.fixed_resources ? { gpus: row.limits.gpus, cpus: row.limits.cpus, memory_gb: row.limits.memory_gb } : {}) });
        }} />
    </label>)}
    <datalist id={`ml-exp-executors-${id}`}>{executors.map((row) => <option key={row.executor_id} value={row.executor_id} />)}</datalist>
    {numeric("gpus", "GPUs", 0)}
    {numeric("cpus", "CPUs", 1)}
    {numeric("memory_gb", "memory (GB)", 1)}
    {numeric("walltime_seconds", "time limit (seconds)", 60)}
    <details><summary>Dataset bindings</summary>
      <p className="cr-trainer-remote__hint">JSON: name to asset.SHA256. Use ml-exp-data://name/path in dataset path fields.</p>
      <textarea className="cr-input nodrag nopan" aria-label="ML-Exp dataset bindings" value={datasetText} disabled={Boolean(jobId)}
        onChange={(event) => setDatasetText(event.target.value)} onBlur={() => {
          try {
            const parsed = JSON.parse(datasetText);
            if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error();
            patchSettings({ datasets: parsed });
            setError(null);
          } catch { setError("Dataset bindings must be a JSON object."); }
        }} />
    </details>
    <div className="cr-trainer-train-btn-row">
      <button type="button" className="cr-trainer-train-btn" disabled={busy} onClick={() => void action(async () => {
        const response = await mlExpApi<{ connected: boolean; server_version?: string; executors?: typeof executors }>("/connection");
        setExecutors(response.executors ?? []);
        setConnection(`Connected${response.server_version ? ` (${response.server_version})` : ""}`);
      })}>Check connection</button>
      {!jobId ? <button type="button" className="cr-trainer-train-btn" disabled={busy || !settings.project || !settings.executor} onClick={() => void submit()}>Submit to ML-Exp</button> : null}
    </div>
    {connection ? <p role="status">{connection}</p> : null}
    {jobId ? <>
      <p role="status">{job?.phase ?? "Recovering task…"} {job?.observation?.scheduler?.state ?? ""}</p>
      {job ? <p className="cr-trainer-remote__hint">{job.project}/{job.run_id}{job.attempt_id ? `/${job.attempt_id}` : ""}<br />Training: {job.observation?.training?.status ?? "unknown"}{job.observation?.model?.step != null ? ` · step ${job.observation.model.step}` : ""}</p> : null}
      <div className="cr-trainer-train-btn-row">
        <button type="button" disabled={busy} onClick={() => void action(async () => {
          try {
            setJob(await mlExpApi<MlExpJob>(`/jobs/${jobId}/continue`, {}));
          } catch (err) {
            if (!(err instanceof MlExpApiError) || err.status !== 404) throw err;
            const request = mlExpRequest(getNodes(), getEdges(), id);
            if (await graphSignature(request) !== data.mlExpGraphSignature) throw new Error("Restore the submitted graph before recovering this request.");
            setJob(await mlExpApi<MlExpJob>("/jobs", { request_id: jobId, train: request, options: settings }));
          }
        })}>Continue submission</button>
        <button type="button" disabled={busy || !job?.submission_id} onClick={() => void action(async () => {
          setJob(await mlExpApi<MlExpJob>(`/jobs/${jobId}/cancel`, {}));
        })}>Cancel job</button>
        <button type="button" disabled={busy || !job?.submission_id || data.mlExpAppliedJobId === jobId} onClick={() => void loadResults()}>
          {data.mlExpAppliedJobId === jobId ? "Results loaded" : "Load results"}
        </button>
      </div>
      <button type="button" disabled={busy} onClick={() => {
        update({ mlExpHistory: [...(data.mlExpHistory ?? []).filter((item) => item.job_id !== jobId), { job_id: jobId, signature: data.mlExpGraphSignature ?? "", applied: data.mlExpAppliedJobId === jobId }], mlExpJobId: undefined, mlExpGraphSignature: undefined, mlExpAppliedJobId: undefined });
        setJob(null);
      }}>New experiment (existing task keeps running)</button>
    </> : null}
    {!jobId && data.mlExpHistory?.length ? <label className="cr-comfy-widget">
      <span className="cr-comfy-widget__label">Previous tasks</span>
      <select className="cr-input nodrag nopan" aria-label="Previous ML-Exp tasks" value="" onChange={(event) => {
        const binding = data.mlExpHistory?.find((item) => item.job_id === event.target.value);
        if (binding) update({ mlExpJobId: binding.job_id, mlExpGraphSignature: binding.signature, mlExpAppliedJobId: binding.applied ? binding.job_id : undefined });
      }}><option value="">Select a saved task</option>{data.mlExpHistory.map((item) => <option key={item.job_id} value={item.job_id}>{item.job_id}</option>)}</select>
    </label> : null}
    {job?.error ? <p className="cr-trainer-train-err">{job.error}</p> : null}
    {pollError ? <p className="cr-trainer-train-err">{pollError}</p> : null}
    {error ? <p role="alert" className="cr-trainer-train-err">{error}</p> : null}
  </div>;
}
