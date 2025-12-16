// Wholesale copied from https://reactflow.dev/examples/layout/elkjs-multiple-handles
// at the moment, haven't refined anything yet

import { Handle, type NodeProps, Position } from "@xyflow/react";
import { type HandleProps } from "@xyflow/system";

import { type ElkNode as ElkNodeType } from "./types";

export default function ElkNode({ data }: NodeProps<ElkNodeType>) {
  return (
    <>
      <div className="handles targets">
        {data.targetHandles.map((handle) => (
          <LabeledHandle
            key={handle.key}
            id={handle.id}
            type="target"
            position={Position.Left}
            label={handle.label}
          />
        ))}
      </div>
      <div className="label">{data.label}</div>
      <div className="handles sources">
        {data.sourceHandles.map((handle) => (
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
