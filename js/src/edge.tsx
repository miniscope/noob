import {
  BaseEdge,
  getBezierPath,
  Position,
  type EdgeProps,
} from "@xyflow/react";
import { useHandleRuntime } from "./runner/runtime.tsx";

/**
 * Custom edge that can be used between an outer grouping node and its interior nodes.
 * Used to connect inputs/returns to interior nodes in recursive tubes.
 * See: https://github.com/xyflow/xyflow/issues/5775
 */
export function InputEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  style = {},
  markerEnd,
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition: Position.Right,
    targetX,
    targetY,
    targetPosition: Position.Left,
  });

  return <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />;
}

export function ReturnEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  style = {},
  markerEnd,
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition: Position.Right,
    targetX,
    targetY,
    targetPosition: Position.Left,
  });
  return <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />;
}

/**
 * The default edge: pulses when its source signal fires. Subscribes to only
 * its own signal's runtime, and renders identically to the built-in default
 * edge when no runtime context is present (static views).
 */
export function RuntimeEdge({
  sourceX,
  sourceY,
  sourcePosition,
  targetX,
  targetY,
  targetPosition,
  sourceHandleId,
  style = {},
  markerEnd,
}: EdgeProps) {
  const signal = useHandleRuntime(sourceHandleId ?? "");
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <BaseEdge
      // keyed on seq so the pulse animation replays per emission
      key={signal?.seq ?? "static"}
      className={signal !== undefined ? "runtime-edge-active" : undefined}
      path={edgePath}
      markerEnd={markerEnd}
      style={style}
    />
  );
}
