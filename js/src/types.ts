import { type Node } from "@xyflow/react";

export interface TubeSpecification {
  noob_id: string;
  noob_mode: string;
  noob_version: string;
  nodes: Record<string, NoobNode>;
}

export interface NoobNode {
  type: string;
  depends?: Record<string, string>[];
}

// Node class expects a type with string keys, not interface
// eslint-disable-next-line @typescript-eslint/consistent-type-definitions
export type ElkNodeData = {
  label: string;
  sourceHandles: { id: string; label: string; key: string }[];
  targetHandles: { id: string; label: string; key: string }[];
};

export type ElkNode = Node<ElkNodeData, "elk">;
