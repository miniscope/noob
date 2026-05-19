// Custom node class with labeled handles

import { type NodeProps, useNodesData } from "@xyflow/react";

import { type TitleNode as TitleNodeType, type NodeUnion } from "../types.ts";

export default function TitleNode({ id, data }: NodeProps<TitleNodeType>) {
  const nodeData = useNodesData<NodeUnion>(id);
  if (nodeData === null) {
    throw new Error("Node with no data! " + id);
  }

  return (
    <div className={"title-node"}>
      <h2 className="nodeID">{data.label}</h2>
      <p className="description">{data.description}</p>
    </div>
  );
}
