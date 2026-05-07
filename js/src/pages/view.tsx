import type { TubeSpecification } from "../types.ts";
import {
  Background,
  ConnectionMode,
  Controls,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import { useEffect } from "react";
import { tubeToFlow } from "../tube.tsx";
import useLayoutNodes from "../useLayoutNodes.tsx";
import ElkNode from "../node.tsx";

interface ViewProps {
  tube_id: string;
  color: "dark" | "light";
}

const nodeTypes = {
  elk: ElkNode,
  group: ElkNode,
};

/**
 * Live viewer that refreshes a tube definition from a websocket
 */
export default function View(props: ViewProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    const socket = new WebSocket(`/spec/${props.tube_id}`);
    socket.addEventListener("message", (event) => {
      if (typeof event.data !== 'string') return;
      const spec = JSON.parse(event.data) as TubeSpecification;
      const [edges, nodes] = tubeToFlow(spec);
      setNodes(nodes);
      setEdges(edges);
    });
  }, [props.tube_id, setNodes, setEdges]);

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
