/**
 * Runtime state for a running tube, built for event rates of thousands per
 * second: a mutable store consumes event batches once, and components
 * subscribe to narrow slices (one node, one signal handle, the event
 * buffer) via useSyncExternalStore. Only the subtrees whose slice changed
 * re-render, and notifications are coalesced to animation frames.
 *
 * The context carries the store itself — a stable object — so providing it
 * never invalidates the tree.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  useSyncExternalStore,
} from "react";
import type { Edge } from "@xyflow/react";
import type { RunnerError, RunnerStatus, TubeRunnerHandle } from "./base.ts";
import { type Epoch, type Event } from "../types/events.ts";
import type { RuntimeStore } from "./events.ts";

export const EVENT_BUFFER_SIZE = 500;

export interface NodeRuntime {
  epoch: Epoch;
  /** monotonic batch counter at this node's last emission; key glow animations on it */
  seq: number;
}

export interface SignalRuntime {
  value: unknown;
  seq: number;
}

type Listener = () => void;

export interface RuntimeContextValue {
  store: RuntimeStore;
  /** slot handle id → the source (signal) handle id feeding it */
  slotSources: Record<string, string>;
}

/** Null outside a runnable view — graph components render statically then. */
export const RuntimeContext = createContext<RuntimeContextValue | null>(null);

const noopSubscribe = () => () => {
  /* no store in context: nothing to unsubscribe */
};

export function useNodeRuntime(id: string): NodeRuntime | undefined {
  const context = useContext(RuntimeContext);
  const store = context?.store;
  const subscribe = useCallback(
    (cb: Listener) =>
      store ? store.subscribeKey(`node:${id}`, cb) : noopSubscribe(),
    [store, id],
  );
  return useSyncExternalStore(subscribe, () => store?.getNode(id));
}

/**
 * Runtime of the signal feeding a handle: pass a source handle id directly,
 * or a slot handle id, which is resolved through the edge map.
 */
export function useHandleRuntime(handleId: string): SignalRuntime | undefined {
  const context = useContext(RuntimeContext);
  const store = context?.store;
  const signalId = context?.slotSources[handleId] ?? handleId;
  const subscribe = useCallback(
    (cb: Listener) =>
      store ? store.subscribeKey(`signal:${signalId}`, cb) : noopSubscribe(),
    [store, signalId],
  );
  return useSyncExternalStore(subscribe, () => store?.getSignal(signalId));
}

export function useRuntimeEvents(): Event[] {
  const context = useContext(RuntimeContext);
  const store = context?.store;
  const subscribe = useCallback(
    (cb: Listener) =>
      store ? store.subscribeKey("events", cb) : noopSubscribe(),
    [store],
  );
  const getEvents = useCallback(
    () => store?.getEvents() ?? EMPTY_EVENTS,
    [store],
  );
  return useSyncExternalStore(subscribe, getEvents);
}

const EMPTY_EVENTS: Event[] = [];

/** Subscribe react state to a runner's low-rate status/error streams. */
export function useRunnerState(handle: TubeRunnerHandle): {
  status: RunnerStatus;
  statusDetail?: string;
  error: RunnerError | null;
  clearError: () => void;
} {
  const [status, setStatus] = useState<RunnerStatus>(handle.status);
  const [statusDetail, setStatusDetail] = useState<string | undefined>(
    undefined,
  );
  const [error, setError] = useState<RunnerError | null>(null);

  useEffect(() => {
    const unsubscribes = [
      handle.statusChanges.subscribe(({ status, detail }) => {
        setStatus(status);
        setStatusDetail(detail);
      }),
      handle.errors.subscribe(setError),
    ];
    return () => unsubscribes.forEach((unsubscribe) => unsubscribe());
  }, [handle]);

  return { status, statusDetail, error, clearError: () => setError(null) };
}

/**
 * Map each slot handle id to the source (signal) handle id feeding it, so
 * slot value display can look up the signal runtime. Built once per graph.
 */
export function buildSlotSources(edges: Edge[]): Record<string, string> {
  const sources: Record<string, string> = {};
  for (const edge of edges) {
    if (edge.sourceHandle && edge.targetHandle) {
      sources[edge.targetHandle] = edge.sourceHandle;
    }
  }
  return sources;
}

/** Compact one-line rendering of a wire value for badges and table cells. */
export function formatValue(value: unknown, maxLength = 24): string {
  if (typeof value === "string" && value.startsWith("pck__"))
    return "⟨pickled⟩";
  const text = value === undefined ? "undefined" : JSON.stringify(value);
  return text.length > maxLength ? text.slice(0, maxLength - 1) + "…" : text;
}

export function formatEpoch(epoch: Epoch): string {
  if (typeof epoch === "number") return String(epoch);
  return epoch
    .map((part) => (typeof part === "number" ? String(part) : part.join(":")))
    .join("/");
}
