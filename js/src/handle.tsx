import type { HandleProps } from "@xyflow/system";
import { Handle, Position } from "@xyflow/react";
import { isNoEvent } from "./types/events.ts";
import { formatValue, useHandleRuntime } from "./runner/runtime.tsx";

export type LabeledHandleProps = HandleProps & {
  label: string;
};

/**
 * A handle with its slot/signal name, and — when a runner is live — the
 * last value that crossed it. Each handle subscribes to only its own
 * signal's runtime, so a value update re-renders just this handle.
 */
export function LabeledHandle(props: LabeledHandleProps) {
  const signal = useHandleRuntime(props.id ?? "");
  const posClass =
    props.position === Position.Left ? "label-left" : "label-right";
  return (
    <Handle {...props}>
      <div className={"handle-label " + posClass}>
        {props.label}
        {signal !== undefined && !isNoEvent(signal.value) && (
          <span className="handle-value" key={signal.seq}>
            {formatValue(signal.value)}
          </span>
        )}
      </div>
    </Handle>
  );
}
