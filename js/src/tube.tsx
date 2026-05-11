/**
 * Handlers for tube specifications!
 * Converting specs to forms that can be consumed by react flow
 */

import type {
  ElkNode as ElkNodeType,
  GroupNode,
  Handle,
  Handles,
  NodeUnion,
  NoobNode,
  TubeNode,
  TubeSpecification,
} from "./types.ts";
import type { Edge } from "@xyflow/react";

/**
 * Create the reactflow form of a tube specification
 */
export function tubeToFlow(tube: TubeSpecification): [Edge[], NodeUnion[]] {
  const edges = getEdges(tube.nodes);
  const nodes = getNodes(tube.nodes, edges);
  return [edges, nodes];
}

function getEdges(nodes: Record<string, NoobNode>, prefix?: string): Edge[] {
  return Object.entries(nodes).flatMap<Edge>(([node_id, node]): Edge[] => {
    if (isTubeNode(node)) {
      return [
        ...getNodeEdges(node, prefix), // inputs
        ...getEdges(node.params.tube.nodes, node_id), // internal nodes
      ];
    } else {
      return getNodeEdges(node, prefix);
    }
  });
}

/**
 * A single node's edges.
 *
 * Prefixes identifiers if the edge is within a nested tube.
 *
 * Handles the level-shifting in nested tubes:
 * Inputs/returns are hoisted up to the edge of the containing node,
 * So they need to connect "one level up" rather than within the prefixed space.
 *
 * @param node
 * @param prefix Prefix to prepend to identifiers to deconflict nodes in nested tubes
 */
function getNodeEdges(node: NoobNode, prefix?: string): Edge[] {
  if (node.depends === undefined || node.depends === null) {
    return [];
  }
  // special case that should be moved to a normalization function -
  // only the return node can have a string as depends, used for returning scalars
  node.depends =
    typeof node.depends === "string" ? [{ value: node.depends }] : node.depends;

  return Array.from(node.depends).map<Edge>((slotsig) => {
    const slot = Object.keys(slotsig)[0];
    const signal = slotsig[slot];
    const signalParts = signal.split(".");
    let targetNode, targetHandle, sourceNode, sourceHandle;
    let edgeType = "default";
    if (typeof prefix !== "string") {
      // Not in a nested tube! handle normally at top level
      sourceNode = signalParts[0];
      sourceHandle = `${signalParts[0]}.signals.${signalParts[1]}`;
      targetNode = node.id;
      targetHandle = `${node.id}.slots.${slot}`;
    } else {
      // Handle nesting
      if (signalParts[0] === "input") {
        // inputs are on the container, rather than being a node themselves
        // e.g. if we are within child tube "b" and depend on input.c,
        // then we depend on b.c
        // (a node named "c" will be prefixed like b.c.value)
        // the input is also a slot, not a signal in this case
        sourceNode = prefix;
        sourceHandle = `${prefix}.slots.${signalParts[1]}`;
        edgeType = "inputEdge";
      } else {
        // Just normal nested depends
        sourceNode = `${prefix}.${signalParts[0]}`;
        sourceHandle = `${prefix}.${signalParts[0]}.signals.${signalParts[1]}`;
      }

      if (node.type === "return") {
        // similarly, return values are on the container, and they are signals
        targetNode = prefix;
        targetHandle = `${prefix}.signals.${slot}`;
        edgeType = "returnEdge";
      } else {
        targetNode = `${prefix}.${node.id}`;
        targetHandle = `${prefix}.${node.id}.slots.${slot}`;
      }
    }
    return {
      id: `${sourceHandle}-${targetHandle}`,
      source: sourceNode,
      sourceHandle,
      target: targetNode,
      targetHandle,
      type: edgeType,
    };
  });
}

// Get all nodes from a tube spec
function getNodes(nodes: Record<string, NoobNode>, edges: Edge[]): NodeUnion[] {
  const has_input = edges.some((e) => e.source === "input");
  if (has_input) {
    nodes = { ...nodes, input: { id: "input", type: "input" } };
  }
  return Object.values(nodes).flatMap((node) => getNode(node, edges));
}

/**
 * Create node data from a noob node spec,
 * expanding nested tube-nodes into group nodes and nested nodes
 *
 * See: https://reactflow.dev/examples/grouping/sub-flows
 */
function getNode(node: NoobNode, edges: Edge[], prefix?: string): NodeUnion[] {
  // Create handle description for node and then filter to unique entries
  if (isTubeNode(node)) {
    return getTubeNode(node, edges);
  } else {
    return getGenericNode(node, edges, prefix);
  }
}

function getGenericNode(
  node: NoobNode,
  edges: Edge[],
  prefix?: string,
): ElkNodeType[] {
  const id = prefix ? `${prefix}.${node.id}` : node.id;
  return [
    {
      id: id,
      data: {
        label: node.id,
        ...getNodeHandles(node.id, edges, prefix),
      },
      position: { x: 0, y: 0 },
      type: "elk",
    },
  ];
}

/**
 * Create a nested tube node,
 * whose source handles are its inputs and its target handles are its return signals.
 *
 * Renders inputs and the return node as handles on the border of an outer grouping node.
 */
function getTubeNode(node: TubeNode, edges: Edge[]): NodeUnion[] {
  // Make the outer grouping node
  const innerTube = node.params.tube;
  const targetHandles = innerTube.input
    ? Object.values(innerTube.input)
        .filter((input) => input.scope === "process")
        .map<Handle>((input) => {
          return {
            id: `${node.id}.slots.${input.id}`,
            label: input.id,
            key: `${node.id}.slots.${input.id}`,
          };
        })
    : [];

  const returnNode = Object.values(innerTube.nodes).filter(
    (node) => node.type === "return",
  )[0];
  returnNode.depends =
    typeof returnNode?.depends === "string"
      ? [{ value: returnNode.depends }]
      : returnNode?.depends;
  const sourceHandles = returnNode?.depends
    ? Array.from(returnNode.depends).map<Handle>((slotsig) => {
        const slot = Object.keys(slotsig)[0];
        return {
          id: `${node.id}.signals.${slot}`,
          label: slot,
          key: `${node.id}.signals.${slot}`,
        };
      })
    : [];

  const groupNode = {
    id: node.id,
    type: "group",
    data: {
      label: node.id,
      sourceHandles,
      targetHandles,
    },
    position: { x: 0, y: 0 },
    key: node.id,
  } as GroupNode;

  // Then the children that go within it
  let childNodes = Object.values(innerTube.nodes)
    .filter((child) => child.type !== "return")
    .flatMap<NodeUnion>((child) => getNode(child, edges, node.id));
  childNodes = childNodes.map((child) => {
    return {
      ...child,
      extent: "parent",
      parentId: node.id,
    } as NodeUnion;
  });
  return [groupNode, ...childNodes];
}

/**
 * Infer node handles from edges
 * FIXME - read the actual node's signals and slots rather than relying on the spec.
 */
function getNodeHandles(
  node_id: string,
  edges: Edge[],
  prefix?: string,
): Handles {
  // righthand signal handles
  node_id = prefix ? `${prefix}.${node_id}` : node_id;
  const sourceHandles = edges
    .filter((e) => e.source === node_id && e.sourceHandle !== undefined)
    .map((e) => {
      const label = (e.sourceHandle as string).split(".").at(-1) as string;
      const id = e.sourceHandle as string;
      return {
        id: id,
        label: label,
        key: id,
      };
    })
    .filter((e, index, self) => self.map((x) => x.id).indexOf(e.id) === index);
  // lefthand slot handles
  const targetHandles = edges
    .filter((e) => e.target === node_id && e.targetHandle !== undefined)
    .map((e) => {
      const label = (e.targetHandle as string).split(".").at(-1) as string;
      const id = e.targetHandle as string;
      return {
        id: id,
        label: label,
        key: id,
      };
    })
    .filter((e, index, self) => self.map((x) => x.id).indexOf(e.id) === index);
  return { sourceHandles, targetHandles };
}

function isTubeNode(node: NoobNode): node is TubeNode {
  return node.type === "tube";
}

export const testExports = {
  getNodeEdges,
  getTubeNode,
  getEdges,
};
