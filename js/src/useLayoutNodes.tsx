// Initialized from https://reactflow.dev/examples/layout/elkjs-multiple-handles

import { useEffect } from "react";
import ELK from "elkjs/lib/elk.bundled.js";
import { type Edge, useNodesInitialized, useReactFlow } from "@xyflow/react";

import type { NodeUnion } from "./types";

import type {
  ElkNode as OElkNode,
  LayoutOptions,
  ElkPort as OElkPort,
} from "elkjs/lib/elk-api";

const TITLE_SPACING = 10;

// https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html
// https://eclipse.dev/elk/reference/options/org-eclipse-elk-nodeSize-options.html
// https://eclipse.dev/elk/blog/posts/2025/25-08-22-node-labels.html
const layoutOptions = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.layered.spacing.edgeNodeBetweenLayers": "10",
  "elk.spacing.nodeNode": "20",
  "elk.edgeRouting": "SPLINES",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.nodePlacement.bk.edgeStraightening": "NONE",
  "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
  "elk.layered.crossingMinimization.strategy": "MEDIAN_LAYER_SWEEP",
  "elk.nodeSize.constraints": "NODE_LABELS PORT_LABELS PORTS",
  "elk.nodeLabels.placement": "INSIDE H_CENTER V_CENTER",
  "elk.nodeSize.options": "COMPUTE_PADDING",
  "elk.portConstraints": "FIXED_SIDE",
  "elk.layered.considerModelOrder.crossingCounterPortInfluence": "0.001",
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

  const titleNode = nodes.filter((n) => n.type === "title")[0];
  const titleHeight = titleNode?.measured?.height ?? 0;

  return nodes.map<NodeUnion>((node) => {
    const layoutedNode = flatChildren.find((lgNode) => lgNode.id === node.id);

    // reorder ports by y position
    if (layoutedNode?.ports === undefined) {
      // eslint-disable-next-line no-console
      console.error("Node had no ports!", layoutedNode);
    } else {
      const portY = Object.fromEntries(
        layoutedNode.ports.map((port) => [port.id, port.y || 0]),
      );
      node.data.sourceHandles.sort(
        (a, b) => (portY[a.id] || 0) - (portY[b.id] || 0),
      );
      node.data.targetHandles.sort(
        (a, b) => (portY[a.id] || 0) - (portY[b.id] || 0),
      );
    }

    // shift everything by the title node
    const x = layoutedNode?.x ?? 0;
    const y =
      (layoutedNode?.y ?? 0) +
      (node.type === "title" || node.parentId !== undefined
        ? 0
        : titleHeight + TITLE_SPACING);

    return {
      ...node,
      data: { ...node.data },
      position: { x, y },
      // the reactflow-generated widths/heights are better for display,
      // but the elk widths/heights are better for nested nodes for some reason.
      ...(node.type === "group" && {
        width: (layoutedNode?.width ?? 0) + TITLE_SPACING,
        height: (layoutedNode?.height ?? 0) + TITLE_SPACING,
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
  const targetPorts = n.data.targetHandles.map<OElkPort>((t) => ({
    id: t.id,
    labels: [{ text: t.label }],
    width: 7 * t.label.length,
    layoutOptions: {
      side: "WEST",
    },
  }));

  const sourcePorts = n.data.sourceHandles.map((s) => ({
    id: s.id,
    labels: [{ text: s.label }],
    width: 7 * s.label.length,
    layoutOptions: {
      side: "EAST",
    },
  }));

  const childNodes = nodes
    .filter((child) => child.parentId === n.id)
    .map((child) => nodeToElk(child, nodes));

  return {
    id: n.id,
    width: n?.measured?.width || 0,
    height: n?.measured?.height || 0,
    labels: [
      {
        text: n.data.label,
        width: n?.measured?.width || 0,
        height: 60,
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
