import { type Node } from "@xyflow/react";

export interface TubeSpecification {
  noob_id: string;
  noob_model: string;
  noob_version: string;
  description: string;
  nodes: Record<string, NoobNode>;
  input?: Record<string, InputSpecification>;
}

export interface NodeInfo {
  node_id: string;
  type: string;
  signals: Record<string, Signal>;
  slots: Record<string, Slot>;
}

export interface InputSpecification {
  id: string;
  type: string;
  scope: InputScope;
  default?: string;
  description?: string;
}

const InputScope = {
  tube: "tube",
  process: "process",
} as const;
export type InputScope = (typeof InputScope)[keyof typeof InputScope];

export interface NoobNode {
  id: string;
  type: string;
  depends?: Record<string, string>[] | string;
  params?: Record<string, string | object>;
  nodeinfo: NodeInfo;
}

export interface Signal {
  name: string;
  annotation: string;
}

export interface Slot {
  name: string;
  annotation: string;
  required: boolean;
}

export interface TubeNode extends NoobNode {
  type: "tube";
  params: { tube: TubeSpecification };
}

export interface Handle {
  id: string;
  label: string;
  key: string;
  required: boolean;
}

// Same as below - node class expects type with string keys
// eslint-disable-next-line @typescript-eslint/consistent-type-definitions
export type Handles = {
  sourceHandles: Handle[];
  targetHandles: Handle[];
};

// Node class expects a type with string keys, not interface
export type ElkNodeData = {
  label: string;
  nodeType: string;
  description?: string;
} & Handles;

export type TitleNodeData = {
  description: string;
} & Omit<ElkNodeData, "nodeType">;

export type ElkNode = Node<ElkNodeData, "elk">;
export type GroupNode = Node<ElkNodeData, "group">;
export type TitleNode = Node<TitleNodeData, "title">;
export type NodeUnion = ElkNode | GroupNode | TitleNode;
