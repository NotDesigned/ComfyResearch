import type { Edge, Node } from "@xyflow/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { graphSignature, MlExpApiError, mlExpApi, mlExpRequest, mlExpRequestId } from "../mlExp";
import { applyGraphFileExportTier } from "../graphFileExportTier";
import type { GraphDocument } from "../../types/graph";

const nodes: Node[] = [{ id: "trainer", type: "trainer", position: { x: 0, y: 0 }, data: {
  trainingSteps: 3, batchSize: -1, executionBackend: "ml_exp",
  mlExpJobId: "a".repeat(32), mlExpSettings: { project: "private" },
  mlExpHistory: [{ job_id: "old" }], lossHistory: [1, .5],
} }];
const edges: Edge[] = [];

afterEach(() => vi.unstubAllGlobals());

describe("ML-Exp execution contract", () => {
  it("exports experiment inputs without previous task identities or curves", () => {
    const request = mlExpRequest(nodes, edges, "trainer");
    expect(request.nodes[0].data.mlExpJobId).toBeUndefined();
    expect(request.nodes[0].data.mlExpHistory).toBeUndefined();
    expect(request.nodes[0].data.mlExpSettings).toBeUndefined();
    expect(request.nodes[0].data.lossHistory).toBeUndefined();
    expect(request.nodes[0].data.trainingSteps).toBe(3);
  });

  it("requires a single parameter combination instead of silently choosing one", () => {
    const varying = nodes.map((node) => ({ ...node, data: { ...node.data, trainingSteps: [3, 5] } }));
    expect(() => mlExpRequest(varying, edges, "trainer")).toThrow("one parameter combination");
  });

  it("detects graph changes without depending on secure-context WebCrypto", async () => {
    const first = mlExpRequest(nodes, edges, "trainer");
    const signature = await graphSignature(first);
    expect(await graphSignature(JSON.parse(JSON.stringify(first)))).toBe(signature);
    first.nodes[0].data.trainingSteps = 4;
    expect(await graphSignature(first)).not.toBe(signature);
    expect(mlExpRequestId()).toMatch(/^[a-f0-9]{32}$/);
  });

  it("uses explicit POST for effects and preserves recoverable HTTP status", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({ detail: "not found" }) });
    vi.stubGlobal("fetch", fetcher);
    await expect(mlExpApi("/jobs/missing")).rejects.toMatchObject({ status: 404, message: "not found" });
    expect(fetcher.mock.calls[0][1].method).toBe("GET");
    await expect(mlExpApi("/jobs/missing/continue", {})).rejects.toBeInstanceOf(MlExpApiError);
    expect(fetcher.mock.calls[1][1].method).toBe("POST");
  });

  it("removes live task bindings from shared Small graph files", () => {
    const document: GraphDocument = { version: 1, nodes: nodes.map((node) => ({ ...node, type: "trainer" })), edges, viewport: { x: 0, y: 0, zoom: 1 } };
    const exported = applyGraphFileExportTier(document, "small");
    expect(exported.nodes[0].data.mlExpJobId).toBeUndefined();
    expect(exported.nodes[0].data.mlExpHistory).toBeUndefined();
    expect(nodes[0].data.mlExpJobId).toBeDefined();
  });
});
