/**
 * TypeScript mirror of noob's runner message protocol
 * (packages/noob/src/noob/network/message.py) — keep the two in sync.
 *
 * These are the same envelopes the zmq runner's command node and node
 * runners exchange; a browser UI is just another peer speaking them over a
 * different pipe (websocket, or in-page pyodide FFI). `node_id` identifies
 * the *sender* of a message.
 */
import type { TubeSpecification } from "./types.ts";

/**
 * JSON form of a noob `Epoch` (via its pydantic serializer): a bare int for
 * root epochs, otherwise `[root, [node, epoch], ...]`.
 */
export type EpochSegment = [node: string, epoch: number];
export type Epoch = number | (number | EpochSegment)[];

/**
 * JSON form of a noob `Event`/`MetaEvent` (noob/event.py); MetaEvents travel
 * in the same stream, distinguished by `node_id === "meta"` with signal
 * `"NodeReady" | "EpochEnded"`.
 *
 * Caveats inherited from the python side:
 * - `id` is a 64-bit int; above 2**53 `JSON.parse` rounds it. Treat ids as
 *   display-only on this side of the wire.
 * - `value` is plain JSON when the value was JSON-serializable, otherwise a
 *   `"pck__"`-prefixed opaque pickle string (see `isPickled`).
 */
export interface Event {
  id: number;
  timestamp: string;
  node_id: string;
  signal: string;
  epoch: Epoch;
  value: unknown;
}

/** noob.network.message.NodeStatus — also used for whole-runner status. */
export type NodeStatus = "stopped" | "waiting" | "ready" | "running" | "closed";

interface BaseMessage {
  node_id: string;
  timestamp: string;
}

export interface InitMsg extends BaseMessage {
  type: "init";
  value?: null;
}
export interface DeinitMsg extends BaseMessage {
  type: "deinit";
  value?: null;
}
export interface ProcessMsg extends BaseMessage {
  type: "process";
  /** epoch null = let the receiving runner decide */
  value: { epoch: Epoch | null; input: Record<string, unknown> | null };
}
/** value = number of iterations to free-run, or null for unbounded. */
export interface StartMsg extends BaseMessage {
  type: "start";
  value: number | null;
}
export interface StopMsg extends BaseMessage {
  type: "stop";
  value?: null;
}
export interface StatusMsg extends BaseMessage {
  type: "status";
  value: NodeStatus;
}
export interface EventMsg extends BaseMessage {
  type: "event";
  value: Event[];
}
/**
 * value is `Picklable[ErrorValue]`: `err_type` is a python type, so the
 * whole payload usually arrives as an opaque `"pck__"` string rather than
 * the structured form. Transports with in-process access to the exception
 * (pyodide) surface errors directly instead.
 */
export interface ErrorMsg extends BaseMessage {
  type: "error";
  value: string | { err_type: unknown; err_args: unknown[]; traceback: string };
}
export interface EpochEndedMsg extends BaseMessage {
  type: "epoch_ended";
  value: Epoch;
}
/** Streamed tube definition (the `/spec/{tube_id}` websocket payload). */
export interface SpecMsg extends BaseMessage {
  type: "spec";
  value: TubeSpecification;
}

export type RunnerMessage =
  | InitMsg
  | DeinitMsg
  | ProcessMsg
  | StartMsg
  | StopMsg
  | StatusMsg
  | EventMsg
  | ErrorMsg
  | EpochEndedMsg
  | SpecMsg;

/** Sender node_id for GUI-originated messages (noob/const.py GUI_NODE_ID). */
export const GUI_NODE_ID = "__gui";

function stamp<T extends RunnerMessage>(
  msg: Omit<T, "node_id" | "timestamp">,
): T {
  return {
    ...msg,
    node_id: GUI_NODE_ID,
    timestamp: new Date().toISOString(),
  } as T;
}

export const commands = {
  init: () => stamp<InitMsg>({ type: "init", value: null }),
  deinit: () => stamp<DeinitMsg>({ type: "deinit", value: null }),
  process: (input: Record<string, unknown> | null = null) =>
    stamp<ProcessMsg>({ type: "process", value: { epoch: null, input } }),
  start: (n: number | null = null) =>
    stamp<StartMsg>({ type: "start", value: n }),
  stop: () => stamp<StopMsg>({ type: "stop", value: null }),
};

export function isMetaEvent(event: Event): boolean {
  return event.node_id === "meta";
}

/**
 * The NoEvent sentinel (noob.event.MetaSignal): "nothing was emitted" —
 * e.g. Return nodes always signal it, since the runner collects their
 * value out-of-band with `get()`.
 */
export function isNoEvent(value: unknown): boolean {
  return value === "__META__NoEvent";
}
