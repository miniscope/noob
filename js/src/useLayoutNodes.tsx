// Initialized from https://reactflow.dev/examples/layout/elkjs-multiple-handles
// also with code from https://github.com/EmilStenstrom/elkjs-svg
// which has now been archived
// (both MIT licensed)

// elk layouting options can be found here:
// https://www.eclipse.org/elk/reference/algorithms/org-eclipse-elk-layered.html
import { useEffect } from "react";
import ELK from "elkjs/lib/elk.bundled.js";
import { type Edge, useNodesInitialized, useReactFlow } from "@xyflow/react";

import { type ElkNode } from "./types";

import type {
  ElkNode as OElkNode,
  ElkExtendedEdge as OElkEdge,
  LayoutOptions,
  ElkPort as OElkPort,
  ElkEdgeSection,
  ElkPoint,
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

/**
 * Create an SVG spline representation of an elk edge
 * This code and below from https://github.com/EmilStenstrom/elkjs-svg/blob/master/elkjs-svg.js
 * @param edge
 * @param routing_style
 */
function renderEdge(
  edge: OElkEdge,
  routing_style: "POLYLINE" | "SPLINES" = "SPLINES",
): string {
  if (edge.sections === undefined) {
    throw new Error("No got dang sections in these edges");
  }
  const bends = getBends(edge.sections);

  if (routing_style == "SPLINES") {
    return bendsToSpline(bends);
  }
  return bendsToPolyline(bends);
}

function getBends(sections: ElkEdgeSection[]): ElkPoint[] {
  let bends: ElkPoint[] = [];
  if (sections && sections.length > 0) {
    sections.forEach((section) => {
      if (section.startPoint) {
        bends.push(section.startPoint);
      }
      if (section.bendPoints) {
        bends = bends.concat(section.bendPoints);
      }
      if (section.endPoint) {
        bends.push(section.endPoint);
      }
    });
  }
  return bends;
}

function bendsToPolyline(bends: ElkPoint[]) {
  return bends.map((bend) => `${bend.x},${bend.y}`).join(" ");
}

function bendsToSpline(bends: ElkPoint[]) {
  if (!bends.length) {
    return "";
  }

  const { x, y } = bends[0];
  const points = [`M${x} ${y}`];

  for (let i = 1; i < bends.length; i = i + 3) {
    const left = bends.length - i;
    if (left == 1) {
      points.push(`L${bends[i].x + " " + bends[i].y}`);
    } else if (left == 2) {
      points.push(`Q${bends[i].x + " " + bends[i].y}`);
      points.push(bends[i + 1].x + " " + bends[i + 1].y);
    } else {
      points.push(`C${bends[i].x + " " + bends[i].y}`);
      points.push(bends[i + 1].x + " " + bends[i + 1].y);
      points.push(bends[i + 2].x + " " + bends[i + 2].y);
    }
  }
  return points.join(" ");
}

interface PropertiedElkNode extends OElkNode {
  properties?: LayoutOptions;
  ports?: PropertiedElkPort[];
}

interface PropertiedElkPort extends OElkPort {
  properties?: LayoutOptions;
}
