import type { NodeUnion, TubeSpecification } from "../types.ts";
import {
  Background,
  ConnectionMode,
  Controls,
  type Edge,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import { useEffect } from "react";
import { tubeToFlow } from "../tube.tsx";
import useLayoutNodes from "../useLayoutNodes.tsx";
import ElkNode from "../node.tsx";
import { InputEdge, ReturnEdge } from "../edge.tsx";

interface ViewProps {
  tube_id: string;
  color: "dark" | "light";
}

const nodeTypes = {
  elk: ElkNode,
  group: ElkNode,
};

const edgeTypes = {
  inputEdge: InputEdge,
  returnEdge: ReturnEdge,
};

/**
 * Live viewer that refreshes a tube definition from a websocket
 */
export default function View(props: ViewProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<NodeUnion>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    const socket = new WebSocket(`/spec/${props.tube_id}`);
    socket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") return;
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
      edgeTypes={edgeTypes}
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
