// Custom node class with labeled handles

import {
  Handle,
  type NodeProps,
  Position,
  useNodesData,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { type HandleProps } from "@xyflow/system";
import { useState, useEffect } from "react";

import { type ElkNode as ElkNodeType, type NodeUnion } from "./types";

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
      <div className="label">{data.label}</div>
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

export type LabeledHandleProps = HandleProps & {
  label: string;
};

function LabeledHandle(props: LabeledHandleProps) {
  const posClass =
    props.position === Position.Left ? "label-left" : "label-right";
  return (
    <Handle {...props}>
      <div className={"handle-label " + posClass}>{props.label}</div>
    </Handle>
  );
}
