import {
  BaseEdge,
  getBezierPath,
  Position,
  type EdgeProps,
} from "@xyflow/react";

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
