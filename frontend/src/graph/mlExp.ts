import type { Edge, Node } from "@xyflow/react";
import { resolveTrainingLengthOnNodes } from "./trainingLengthResolve";
import { planTrainSeriesAssignments, serializeExecutionGraphForTarget } from "./trainSeriesPlan";

export type MlExpSettings = {
  project: string;
  executor: string;
  gpus: number;
  cpus: number;
  memory_gb: number;
  walltime_seconds: number;
  datasets: Record<string, string>;
};

export type MlExpJob = {
  job_id: string;
  trainer_node_id: string;
  phase: string;
  project: string;
  run_id: string;
  submission_id: string | null;
  attempt_id: string | null;
  error: string | null;
  observation?: {
    scheduler?: { state?: string; stale?: boolean };
    training?: { status?: string };
    publication?: { download_available?: boolean };
    model?: { step?: number };
  };
};

export const defaultMlExpSettings: MlExpSettings = {
  project: "", executor: "", gpus: 0, cpus: 2, memory_gb: 4,
  walltime_seconds: 1800, datasets: {},
};

export function mlExpRequest(nodes: Node[], edges: Edge[], trainerId: string) {
  const graph = serializeExecutionGraphForTarget(nodes, edges, trainerId);
  if (planTrainSeriesAssignments(nodes, edges, trainerId).length > 1) {
    throw new Error("ML-Exp currently submits one parameter combination. Choose one value for each sweep field.");
  }
  return {
    trainer_node_id: trainerId,
    nodes: resolveTrainingLengthOnNodes(graph.nodes, graph.edges),
    edges: graph.edges,
  };
}

export async function graphSignature(request: unknown): Promise<string> {
  // UI edit detection only. The server independently binds results with SHA256.
  // Keep it usable on an HTTP LAN workbench, where crypto.subtle is unavailable.
  const stable = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(stable);
    if (value && typeof value === "object") return Object.fromEntries(
      Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, stable(item)]),
    );
    return value;
  };
  let experiment = request;
  if (request && typeof request === "object" && "nodes" in request && Array.isArray(request.nodes)) {
    experiment = { ...request, nodes: request.nodes.map((node: Record<string, unknown>) => {
      const { position: _position, style: _style, hidden: _hidden, ...definition } = node;
      return definition;
    }) };
  }
  const bytes = new TextEncoder().encode(JSON.stringify(stable(experiment)));
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
  return hash.toString(16).padStart(16, "0");
}

export function mlExpRequestId(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export class MlExpApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export async function mlExpApi<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/train/ml-exp${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const value = await response.json();
  if (!response.ok) throw new MlExpApiError(typeof value.detail === "string" ? value.detail : `ML-Exp request failed (${response.status})`, response.status);
  return value as T;
}
