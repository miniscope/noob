// Custom node class with labeled handles

import {
  type NodeProps,
  Position,
  useNodesData,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { useEffect, useState } from "react";

import { type ElkNode as ElkNodeType, type NodeUnion } from "../types.ts";
import { LabeledHandle } from "../handle.tsx";

export default function ElkNode({ id, data }: NodeProps<ElkNodeType>) {
  const nodeData = useNodesData<NodeUnion>(id);
  if (nodeData === null) {
    throw new Error("Node with no data! " + id);
  }
  const [sourceHandles, setSourceHandles] = useState(
    nodeData.data.sourceHandles,
  );
  const [targetHandles, setTargetHandles] = useState(
    nodeData.data.sourceHandles,
  );
  const updateNodeInternals = useUpdateNodeInternals();

  /* Need to reactively update handle order because reactflow memoizes node internals
     See: https://reactflow.dev/api-reference/hooks/use-update-node-internals
   */
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSourceHandles([...nodeData.data.sourceHandles]);
    setTargetHandles([...nodeData.data.targetHandles]);
    updateNodeInternals(id);
  }, [id, nodeData, updateNodeInternals]);

  return (
    <>
      <div className="handles targets">
        {targetHandles.map((handle) => (
          <LabeledHandle
            key={handle.key}
            id={handle.id}
            type="source" // Allow nested node inputs/returns to connect - hack until we replace the edge type
            position={Position.Left}
            label={handle.label}
          />
        ))}
      </div>
      <h2 className="nodeID">{data.label}</h2>
      <span className="nodeType">{data.nodeType}</span>
      <div className="handles sources">
        {sourceHandles.map((handle) => (
          <LabeledHandle
            key={handle.key}
            id={handle.id}
            type="source"
            position={Position.Right}
            label={handle.label}
          />
        ))}
      </div>
    </>
  );
}
