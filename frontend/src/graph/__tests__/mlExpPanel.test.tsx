// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MlExpTrainerPanel } from "../../components/nodes/MlExpTrainerPanel";
import { defaultTrainerData } from "../../components/nodes/trainerDefaults";
import { graphSignature, mlExpApi, mlExpRequest } from "../mlExp";
import { flushCheckpointApplyTrainerVizAndHydrateTv0d } from "../trainerTrainCompleteCommit";

const flow = vi.hoisted(() => ({
  nodes: [{ id: "trainer", type: "trainer", position: { x: 0, y: 0 }, data: { trainingSteps: 3 } }],
  getEdges: vi.fn(() => []), setNodes: vi.fn(),
}));
vi.mock("@xyflow/react", () => ({ useReactFlow: () => ({ getNodes: () => flow.nodes, getEdges: flow.getEdges, setNodes: flow.setNodes }) }));
vi.mock("../mlExp", async (original) => ({ ...await original<typeof import("../mlExp")>(), mlExpApi: vi.fn() }));
vi.mock("../trainerTrainCompleteCommit", () => ({ flushCheckpointApplyTrainerVizAndHydrateTv0d: vi.fn(async () => {}) }));

const jobId = "a".repeat(32);
const job = { job_id: jobId, trainer_node_id: "trainer", phase: "SUBMITTED", project: "study", run_id: "cr-a", submission_id: "action-b", observation: { publication: { download_available: true } } };
let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.clearAllMocks();
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

function button(label: string): HTMLButtonElement {
  const found = Array.from(host.querySelectorAll("button")).find((item) => item.textContent === label);
  if (!found) throw new Error(`Missing button ${label}`);
  return found;
}

describe("ML-Exp Trainer panel", () => {
  it("reconnects a persisted task without resubmitting and loads through the existing result path", async () => {
    const request = mlExpRequest(flow.nodes, [], "trainer");
    const signature = await graphSignature(request);
    const update = vi.fn();
    const complete = { type: "complete", loss_history: [1], step_ticks: [0] };
    vi.mocked(mlExpApi).mockImplementation(async (path) => path.endsWith("/results") ? { complete } : job);
    await act(async () => root.render(<MlExpTrainerPanel id="trainer" data={{ ...defaultTrainerData(), mlExpJobId: jobId, mlExpGraphSignature: signature }} update={update} />));
    expect(vi.mocked(mlExpApi).mock.calls.map(([path]) => path)).toEqual([`/jobs/${jobId}`, `/jobs/${jobId}/refresh`]);
    await act(async () => button("Load results").click());
    expect(flushCheckpointApplyTrainerVizAndHydrateTv0d).toHaveBeenCalledWith(expect.objectContaining({ trainerNodeId: "trainer", payload: complete }));
    expect(update).toHaveBeenCalledWith({ mlExpAppliedJobId: jobId });
  });

  it("refuses to apply a result to a changed graph", async () => {
    vi.mocked(mlExpApi).mockResolvedValue(job);
    await act(async () => root.render(<MlExpTrainerPanel id="trainer" data={{ ...defaultTrainerData(), mlExpJobId: jobId, mlExpGraphSignature: "different" }} update={vi.fn()} />));
    await act(async () => button("Load results").click());
    expect(host.textContent).toContain("graph changed");
    expect(flushCheckpointApplyTrainerVizAndHydrateTv0d).not.toHaveBeenCalled();
    expect(vi.mocked(mlExpApi).mock.calls.some(([path]) => path.endsWith("/results"))).toBe(false);
  });
});
