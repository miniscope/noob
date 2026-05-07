// based on
// https://reactflow.dev/examples/layout/elkjs-multiple-handles

import type {TubeSpecification,} from "./types.ts";
import {Background, ConnectionMode, Controls, ReactFlow, useEdgesState, useNodesState,} from "@xyflow/react";
import ElkNode from "./node.tsx";
import useLayoutNodes from "./useLayoutNodes.tsx";
import "@xyflow/react/dist/style.css";
import {tubeToFlow} from "./tube.tsx";

interface NoobFlowProps {
  tube: TubeSpecification;
  color: "dark" | "light";
}

const nodeTypes = {
  elk: ElkNode,
  group: ElkNode,
};

export function NoobFlow(props: NoobFlowProps) {
  const [edgesInit, nodesInit] = tubeToFlow(props.tube);

  const [nodes, , onNodesChange] = useNodesState(nodesInit);
  const [edges, , onEdgesChange] = useEdgesState(edgesInit);

  useLayoutNodes();
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      colorMode={props.color}
      connectionMode={ConnectionMode.Loose} // allow the inputs/returns of nested tubes to connect both ways
      fitView
    >
      <Background />
      <Controls />
    </ReactFlow>
  );
}

