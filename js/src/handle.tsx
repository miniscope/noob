import type { HandleProps } from "@xyflow/system";
import { Handle, Position } from "@xyflow/react";

export type LabeledHandleProps = HandleProps & {
  label: string;
};

export function LabeledHandle(props: LabeledHandleProps) {
  const posClass =
    props.position === Position.Left ? "label-left" : "label-right";
  return (
    <Handle {...props}>
      <div className={"handle-label " + posClass}>{props.label}</div>
    </Handle>
  );
}
