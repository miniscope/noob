/**
 * The chrome around a running tube: lifecycle buttons, the collected
 * return value, and the event table. All of it binds to TubeRunnerHandle
 * or the runtime store, so the same components serve every transport.
 */
import { useEffect, useState } from "react";
import { Panel } from "@xyflow/react";
import type { TubeRunnerHandle } from "./base.ts";
import { useRunnerState } from "./runtime.tsx";

function run(action: () => void | Promise<void>): void {
  void Promise.resolve()
    .then(action)
    .catch(() => {
      /* failures surface through the runner's error stream */
    });
}

/**
 * Which buttons are enabled derives purely from the runner status: against
 * a backend that never reports one (e.g. today's view-only websocket
 * server) everything but "load" stays disabled.
 */
export function RunnerControls({ handle }: { handle: TubeRunnerHandle }) {
  const { status, statusDetail, error, clearError } = useRunnerState(handle);
  const loaded =
    status !== "unloaded" && status !== "loading" && status !== "closed";

  return (
    <div className="noob-runner-controls">
      {!loaded ? (
        <button
          className="runner-enable"
          disabled={status === "loading"}
          onClick={() => run(() => handle.enable())}
        >
          {status === "loading"
            ? (statusDetail ?? "loading…")
            : `load ${handle.transport}`}
        </button>
      ) : (
        <>
          <button
            disabled={status !== "stopped"}
            onClick={() => run(() => handle.init())}
          >
            init
          </button>
          <button
            disabled={status !== "ready" && status !== "running"}
            onClick={() => run(() => handle.deinit())}
          >
            deinit
          </button>
          <button
            disabled={status !== "ready"}
            onClick={() => run(() => handle.process())}
          >
            process
          </button>
          <button
            disabled={status !== "ready"}
            onClick={() => run(() => handle.start())}
          >
            start
          </button>
          <button
            disabled={status !== "running"}
            onClick={() => run(() => handle.stop())}
          >
            stop
          </button>
        </>
      )}
      <span className="runner-status" data-status={status}>
        {status}
      </span>
      {error && (
        <div className="runner-error" title={error.traceback}>
          <span>{error.message}</span>
          <button onClick={clearError}>×</button>
        </div>
      )}
    </div>
  );
}

/**
 * The tube's latest collected return value, parked in the bottom-right
 * corner of the flow. Return nodes emit NoEvent into the graph — the
 * runner collects their actual value out-of-band — so this panel is where
 * that value surfaces: a scalar for single-dependency returns, a dict
 * keyed by slot for multi-dependency ones.
 */
export function ReturnPanel({ handle }: { handle: TubeRunnerHandle }) {
  const [value, setValue] = useState<unknown>(undefined);
  useEffect(() => handle.returns.subscribe(setValue), [handle]);

  if (value === undefined || value === null) return null;
  return (
    <Panel position="bottom-right" className="noob-return-panel">
      <span className="return-label">return</span>
      <pre>{JSON.stringify(value, null, 1)}</pre>
    </Panel>
  );
}
