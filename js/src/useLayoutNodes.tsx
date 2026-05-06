// Wholesale copied from https://reactflow.dev/examples/layout/elkjs-multiple-handles
// at the moment, haven't refined anything yet

// elk layouting options can be found here:
// https://www.eclipse.org/elk/reference/algorithms/org-eclipse-elk-layered.html
import { useEffect } from "react";
import ELK from "elkjs/lib/elk.bundled.js";
import { type Edge, useNodesInitialized, useReactFlow } from "@xyflow/react";

import { type ElkNode } from "./types";

import type {
  ElkNode as OElkNode,
  LayoutOptions,
  ElkPort as OElkPort,
} from "elkjs/lib/elk-api";

// https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html
// https://eclipse.dev/elk/reference/options/org-eclipse-elk-nodeSize-options.html
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
};

const elk = new ELK();

/**
 * uses elkjs to give each node a layouted position
 * Have to re-nest the graph here - elk uses nested graph, reactflow uses flat node structure
 * https://github.com/xyflow/xyflow/discussions/3495
 */
export const getLayoutedNodes = async (nodes: ElkNode[], edges: Edge[]) => {
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
  return nodes.map((node) => {
    const layoutedNode = flatChildren?.find((lgNode) => lgNode.id === node.id);

    return {
      ...node,
      position: {
        x: layoutedNode?.x ?? 0,
        y: layoutedNode?.y ?? 0,
      },
      width: layoutedNode?.width ?? node.width,
      height: layoutedNode?.height ?? node.height,
    };
  });
};

function flattenChildren(
  layoutedGraph: PropertiedElkNode,
): PropertiedElkNode[] {
  return layoutedGraph.children
    ? layoutedGraph.children.flatMap((c) => [c, ...flattenChildren(c)])
    : [];
}

function nodeToElk(n: ElkNode, nodes: ElkNode[]): PropertiedElkNode {
  const targetPorts = n.data.targetHandles.map((t) => ({
    id: t.id,
    labels: [{ text: t.label }],
    width: 5 * t.label.length,
    properties: {
      side: "WEST",
    },
  }));

  const sourcePorts = n.data.sourceHandles.map((s) => ({
    id: s.id,
    labels: [{ text: s.label }],
    width: 5 * s.label.length,
    properties: {
      side: "EAST",
    },
  }));

  const childNodes = nodes
    .filter((child) => child.parentId === n.id)
    .map((child) => nodeToElk(child, nodes));

  return {
    id: n.id,
    width: n.width ?? 50,
    height: n.height ?? 50,
    properties: {
      "org.eclipse.elk.portConstraints": "FIXED_SIDE",
    },
    // we are also passing the id, so we can also handle edges without a sourceHandle or targetHandle option
    ports: [...targetPorts, ...sourcePorts],
    children: childNodes,
  };
}

export default function useLayoutNodes() {
  const nodesInitialized = useNodesInitialized();
  const { getNodes, getEdges, setNodes, fitView } = useReactFlow<ElkNode>();

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

interface PropertiedElkNode extends OElkNode {
  properties?: LayoutOptions;
  ports?: PropertiedElkPort[];
}

interface PropertiedElkPort extends OElkPort {
  properties?: LayoutOptions;
}
