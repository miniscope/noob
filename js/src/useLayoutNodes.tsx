// Initialized from https://reactflow.dev/examples/layout/elkjs-multiple-handles

import { useEffect } from "react";
import ELK from "elkjs/lib/elk.bundled.js";
import { type Edge, useNodesInitialized, useReactFlow } from "@xyflow/react";

import { type NodeUnion } from "./types";

import type {
  ElkNode as OElkNode,
  LayoutOptions,
  ElkPort as OElkPort,
} from "elkjs/lib/elk-api";

// https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html
// https://eclipse.dev/elk/reference/options/org-eclipse-elk-nodeSize-options.html
// https://eclipse.dev/elk/blog/posts/2025/25-08-22-node-labels.html
const layoutOptions = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.layered.spacing.edgeNodeBetweenLayers": "20",
  "elk.spacing.nodeNode": "50",
  "elk.edgeRouting": "SPLINES",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.nodePlacement.bk.edgeStraightening": "NONE",
  "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
  "elk.layered.crossingMinimization.strategy": "MEDIAN_LAYER_SWEEP",
  "elk.nodeSize.constraints": "NODE_LABELS PORT_LABELS PORTS",
  "elk.nodeLabels.placement": "INSIDE H_CENTER V_CENTER",
  "elk.nodeSize.options": "COMPUTE_PADDING",
  "elk.portConstraints": "FIXED_SIDE",
  // "elk.nodeSize.minimum"
};

const elk = new ELK();

/**
 * uses elkjs to give each node a layouted position
 * Have to re-nest the graph here - elk uses nested graph, reactflow uses flat node structure
 * https://github.com/xyflow/xyflow/discussions/3495
 */
export const getLayoutedNodes = async (
  nodes: NodeUnion[],
  edges: Edge[],
): Promise<NodeUnion[]> => {
  const graph = {
    id: "root",
    layoutOptions,
    children: nodes
      .filter((n) => n.parentId === undefined)
      .map((n) => nodeToElk(n, nodes)),
    edges: edges.map((e) => ({
      id: e.id,
      sources: [e.sourceHandle || e.source],
      targets: [e.targetHandle || e.target],
    })),
  };
  const layoutedGraph = await elk.layout(graph, {
    layoutOptions: layoutOptions,
  });
  const flatChildren = flattenChildren(layoutedGraph);
  return nodes.map<NodeUnion>((node) => {
    const layoutedNode = flatChildren?.find((lgNode) => lgNode.id === node.id);

    return {
      ...node,
      position: {
        x: layoutedNode?.x ?? 0,
        y: layoutedNode?.y ?? 0,
      },
      // the reactflow-generated widths/heights are better for display,
      // but the elk widths/heights are better for nested nodes for some reason.
      ...(node.type === "group" && {
        width: layoutedNode?.width,
        height: layoutedNode?.height,
      }),
    };
  });
};

export default function useLayoutNodes() {
  const nodesInitialized = useNodesInitialized();
  const { getNodes, getEdges, setNodes, fitView } = useReactFlow<NodeUnion>();

  useEffect(() => {
    if (nodesInitialized) {
      const layoutNodes = async () => {
        const layoutedNodes = await getLayoutedNodes(getNodes(), getEdges());
        setNodes(layoutedNodes);
        await fitView();
      };

      void layoutNodes();
    }
  }, [nodesInitialized, getNodes, getEdges, setNodes, fitView]);

  return null;
}

function nodeToElk(n: NodeUnion, nodes: NodeUnion[]): PropertiedElkNode {
  const targetPorts = n.data.targetHandles.map((t) => ({
    id: t.id,
    labels: [{ text: t.label }],
    width: 7 * t.label.length,
    properties: {
      side: "WEST",
    },
  }));

  const sourcePorts = n.data.sourceHandles.map((s) => ({
    id: s.id,
    labels: [{ text: s.label }],
    width: 7 * s.label.length,
    properties: {
      side: "EAST",
    },
  }));

  const childNodes = nodes
    .filter((child) => child.parentId === n.id)
    .map((child) => nodeToElk(child, nodes));

  return {
    id: n.id,
    ...(n.width && { width: n.width }),
    ...(n.height && { height: n.height }),
    labels: [
      {
        text: n.data.label,
        ...(n.width && { width: n.width }),
      },
    ],
    // we are also passing the id, so we can also handle edges without a sourceHandle or targetHandle option
    ports: [...targetPorts, ...sourcePorts],
    children: childNodes,
  };
}

function flattenChildren(
  layoutedGraph: PropertiedElkNode,
): PropertiedElkNode[] {
  return layoutedGraph.children
    ? layoutedGraph.children.flatMap((c) => [c, ...flattenChildren(c)])
    : [];
}

interface PropertiedElkNode extends OElkNode {
  properties?: LayoutOptions;
  ports?: PropertiedElkPort[];
}

interface PropertiedElkPort extends OElkPort {
  properties?: LayoutOptions;
}
