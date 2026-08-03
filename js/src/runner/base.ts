/**
 * The contract between tube runner backends and the UI: a small lifecycle
 * surface plus push streams — most importantly `events`, batches of
 * wire-form noob events. The UI never knows what's behind it: a websocket
 * to a python process in fullstack mode, or a pyodide interpreter running
 * the tube in this page.
 */
import type { NodeStatus, Event } from "../types/events.ts";
import type { Stream } from "./events.ts";

/**
 * Runner state as the UI sees it: the NodeStatus vocabulary plus two
 * client-local transport states — `unloaded` (transport needs enable(),
 * e.g. pyodide not downloaded) and `loading`.
 */
export type RunnerStatus = NodeStatus | "unloaded" | "loading";

export interface StatusChange {
  status: RunnerStatus;
  detail?: string;
}

export interface RunnerError {
  message: string;
  traceback?: string;
}

export interface TubeRunnerHandle {
  /** Short transport label for UI copy, e.g. "pyodide", "websocket". */
  readonly transport: string;
  readonly status: RunnerStatus;

  /** Batches of events as the tube runs; one batch per process step. */
  readonly events: Stream<Event[]>;
  /** What each process step's collected Return node value was, if any. */
  readonly returns: Stream<unknown>;
  readonly statusChanges: Stream<StatusChange>;
  readonly errors: Stream<RunnerError>;

  /**
   * Bootstrap the transport itself: load pyodide + wheels, or open the
   * socket. Idempotent; nothing is downloaded before this. Runners start
   * `unloaded`.
   */
  enable(): Promise<void>;

  init(): Promise<void>;
  deinit(): Promise<void>;
  /** Process one step of data from each source. */
  process(): Promise<void>;
  /** Process repeatedly until stop(); pacing is client-side where the loop is client-driven. */
  start(intervalMs?: number): void;
  stop(): void;

  /** Tear down the transport and close all streams. */
  dispose(): void;
}
