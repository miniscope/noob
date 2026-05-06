import { type Node } from "@xyflow/react";

export interface TubeSpecification {
  noob_id: string;
  noob_mode: string;
  noob_version: string;
  nodes: Record<string, NoobNode>;
  input?: Record<string, InputSpecification>;
}

export interface InputSpecification {
  id: string;
  type: string;
  scope: InputScope;
  default?: string;
  description?: string;
}

export enum InputScope {
  tube = "tube",
  process = "process",
}

export interface NoobNode {
  id: string;
  type: string;
  depends?: Record<string, string>[];
  params?: Record<string, string | object>;
}

export interface TubeNode extends NoobNode {
  type: "tube";
  params: { tube: TubeSpecification };
}

export interface Handle {
  id: string;
  label: string;
  key: string;
}

export interface Handles {
  sourceHandles: Handle[];
  targetHandles: Handle[];
}

// Node class expects a type with string keys, not interface
export type ElkNodeData = {
  label: string;
} & Handles;

export type ElkNode = Node<ElkNodeData, "elk">;
export type GroupNode = Node<ElkNodeData, "group">;
export type NodeUnion = ElkNode | GroupNode;
