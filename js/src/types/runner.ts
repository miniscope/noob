import type { Epoch } from "./events.ts";
import type { RuntimeStore } from "../runner/events.ts";

export interface NodeRuntime {
  epoch: Epoch;
  /** monotonic batch counter at this node's last emission; key glow animations on it */
  seq: number;
}

export interface SignalRuntime {
  value: unknown;
  seq: number;
}

export type Listener = () => void;

export interface RuntimeContextValue {
  store: RuntimeStore;
  /** slot handle id → the source (signal) handle id feeding it */
  slotSources: Record<string, string>;
}

export const EVENT_BUFFER_SIZE = 500;
