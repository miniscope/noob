// based on
// https://reactflow.dev/examples/layout/elkjs-multiple-handles

import type {
  TubeSpecification,
  ElkNode as ElkNodeType,
  NoobNode,
} from "./types.ts";
import {
  type Edge,
  ReactFlow,
  Background,
  Controls,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import ElkNode from "./node.tsx";
import useLayoutNodes from "./useLayoutNodes.tsx";
import "@xyflow/react/dist/style.css";

interface NoobFlowProps {
  tube: TubeSpecification;
  color: "dark" | "light";
}

const nodeTypes = {
  elk: ElkNode,
};

export function NoobFlow(props: NoobFlowProps) {
  const [edgesInit, nodesInit] = tubeToElk(props.tube);

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
      fitView
    >
      <Background />
      <Controls />
    </ReactFlow>
  );
}

function tubeToElk(tube: TubeSpecification): [Edge[], ElkNodeType[]] {
  const edges = getEdges(tube.nodes);
  const nodes = getNodes(tube.nodes, edges);
  return [edges, nodes];
}

function getEdges(nodes: Record<string, NoobNode>): Edge[] {
  return Object.entries(nodes).flatMap<Edge>(([node_id, node]): Edge[] => {
    if (node.depends === undefined || node.depends === null) {
      return [];
    } else {
      return node.depends.map<Edge>((slotsig) => {
        const slot = Object.keys(slotsig)[0];
        const signal = slotsig[slot];
        const sourceHandle = signal;
        const sourceNode = signal.split(".")[0];
        const targetHandle = `${node_id}.${slot}`;
        return {
          id: `${sourceHandle}-${targetHandle}`,
          source: sourceNode,
          sourceHandle,
          target: node_id,
          targetHandle,
        };
      });
    }
  });
}

function getNodes(
  nodes: Record<string, NoobNode>,
  edges: Edge[],
): ElkNodeType[] {
  return Object.keys(nodes).map((node_id) => {
    // Create handle description for node and then filter to unique entries
    const sourceHandles = edges
      .filter((e) => e.source === node_id && e.sourceHandle !== undefined)
      .map((e) => {
        const label = (e.sourceHandle as string).split(".")[1];
        return {
          id: e.sourceHandle as string,
          label: label,
          key: e.id,
        };
      })
      .filter(
        (e, index, self) => self.map((x) => x.id).indexOf(e.id) === index,
      );
    const targetHandles = edges
      .filter((e) => e.target === node_id && e.targetHandle !== undefined)
      .map((e) => {
        const label = (e.targetHandle as string).split(".")[1];
        return { id: e.targetHandle as string, label: label, key: e.id };
      })
      .filter(
        (e, index, self) => self.map((x) => x.id).indexOf(e.id) === index,
      );

    return {
      id: node_id,
      data: {
        label: node_id,
        sourceHandles,
        targetHandles,
      },
      position: { x: 0, y: 0 },
      type: "elk",
    };
  });
}
