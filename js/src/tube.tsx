/**
 * Handlers for tube specifications!
 * Converting specs to forms that can be consumed by react flow
 */

import type {
  AssetSpecification,
  ElkNode as ElkNodeType,
  GroupNode,
  Handle,
  Handles, InputSpecification,
  NodeUnion,
  NoobNode,
  TitleNode,
  TubeNode,
  TubeSpecification,
} from "./types.ts";
import type { Edge } from "@xyflow/react";

/**
 * Create the reactflow form of a tube specification
 */
export function tubeToFlow(tube: TubeSpecification): [Edge[], NodeUnion[]] {
  const edges = getEdges(tube.nodes);
  let nodes = getNodes(tube.nodes);
  nodes = [_titleNode(tube.noob_id, tube.description ?? ""), ...nodes];

  if (tube.input && Object.keys(tube.input).length !== 0) {
    nodes = [...nodes, _inputNode(tube.input)]
  }
  if (tube.assets && Object.keys(tube.assets).length !== 0) {
    nodes = [...nodes, _assetNode(tube.assets)]
  }
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
function getNodes(nodes: Record<string, NoobNode>): NodeUnion[] {
  return Object.values(nodes).flatMap((node) => getNode(node));
}

/**
 * Create node data from a noob node spec,
 * expanding nested tube-nodes into group nodes and nested nodes
 *
 * See: https://reactflow.dev/examples/grouping/sub-flows
 */
function getNode(node: NoobNode, prefix?: string): NodeUnion[] {
  // Create handle description for node and then filter to unique entries
  if (isTubeNode(node)) {
    return getTubeNode(node, prefix);
  } else {
    return getGenericNode(node, prefix);
  }
}

function getGenericNode(node: NoobNode, prefix?: string): ElkNodeType[] {
  const id = prefix ? `${prefix}.${node.id}` : node.id;
  return [
    {
      id: id,
      data: {
        label: node.id,
        nodeType: node.type,
        ...getNodeHandles(node, prefix),
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
function getTubeNode(node: TubeNode, prefix?: string): NodeUnion[] {
  // Make the outer grouping node
  const newPrefix = prefix ? `${prefix}.${node.id}` : node.id

  const innerTube = node.params.tube;
  const targetHandles = innerTube.input
    ? Object.values(innerTube.input).map<Handle>((input) => {
        return {
          id: `${newPrefix}.slots.${input.id}`,
          label: input.id,
          key: `${newPrefix}.slots.${input.id}`,
          required: true,
        };
      })
    : [];

  const returnNode = Object.values(innerTube.nodes).filter(
    (node) => node.type === "return",
  )[0];
  let sourceHandles: Handle[] = [];
  if (returnNode !== undefined) {
    returnNode.depends =
      typeof returnNode?.depends === "string"
        ? [{ value: returnNode.depends }]
        : returnNode?.depends;
    sourceHandles = returnNode?.depends
      ? Array.from(returnNode.depends).map<Handle>((slotsig) => {
          const slot = Object.keys(slotsig)[0];
          return {
            id: `${newPrefix}.signals.${slot}`,
            label: slot,
            key: `${newPrefix}.signals.${slot}`,
            required: false, // return node slots are never really required, special case.
          };
        })
      : [];
  }

  const groupNode = {
    id: newPrefix,
    type: "group",
    data: {
      label: node.id,
      nodeType: node.type,
      sourceHandles,
      targetHandles,
    },
    position: { x: 0, y: 0 },
    key: newPrefix,
  } as GroupNode;



  // Then the children that go within it
  let childNodes = Object.values(innerTube.nodes)
    .filter((child) => child.type !== "return")
    .flatMap<NodeUnion>((child) => getNode(child, newPrefix));
  if (innerTube.assets && Object.keys(innerTube.assets).length !== 0) {
    childNodes = [...childNodes, _assetNode(innerTube.assets, newPrefix)]
    console.log(childNodes);
  }
  childNodes = childNodes.map((child) => {
    return {
      ...child,
      extent: "parent",
      parentId: newPrefix,
    };
  });
  return [groupNode, ...childNodes];
}

/**
 * Infer node handles from edges
 */
function getNodeHandles(node: NoobNode, prefix?: string): Handles {
  // righthand signal handles
  const node_id = prefix ? `${prefix}.${node.id}` : node.id;
  const sourceHandles = Object.values(node.nodeinfo.signals).map((sig) => {
    const id = `${node_id}.signals.${sig.name}`;
    return {
      id: id,
      label: sig.name,
      key: id,
      required: true, // signals don't really have a requiredness...
    };
  });
  // lefthand slot handles
  const targetHandles = Object.values(node.nodeinfo.slots).map((slot) => {
    const id = `${node_id}.slots.${slot.name}`;
    return {
      id: id,
      label: slot.name,
      key: id,
      required: slot.required,
    };
  });
  return { sourceHandles, targetHandles };
}

function isTubeNode(node: NoobNode): node is TubeNode {
  return node.type === "tube";
}

function _titleNode(title: string, description: string): TitleNode {
  return {
    id: "__title__",
    position: { x: 0, y: 0 },
    data: {
      label: title,
      description: description,
      sourceHandles: [],
      targetHandles: [],
    },
    type: "title",
  };
}

// TODO: Dedicated representation of inputs
function _inputNode(input: Record<string, InputSpecification>): ElkNodeType {
  return {
    id: "input",
    position: { x: 0, y: 0 },
    type: "elk",
    data: {
      label: "input",
      nodeType: "input",
      targetHandles: [],
      sourceHandles: Object.values(input).map((i) => {
        return {
          id: `input.signals.${i.id}`,
          label: i.id,
          key: `input.signals.${i.id}`,
          required: true,
        };
      }),
    },
  }
}

// TODO: dedicated representation of assets
function _assetNode(assets: Record<string, AssetSpecification>, prefix?: string): ElkNodeType {
  console.log(assets)
  return {
    id: prefix ? `${prefix}.assets` : "assets",
    position: { x: 0, y: 0 },
    type: "elk",
    data: {
      label: "assets",
      nodeType: "assets",
      targetHandles: [],
      sourceHandles: Object.values(assets).map((i) => {
        return {
          id: prefix ? `${prefix}.assets.signals.${i.id}` : `assets.signals.${i.id}`,
          label: i.id,
          key: prefix ? `${prefix}.assets.signals.${i.id}` : `assets.signals.${i.id}`,
          required: true,
        };
      }),
    },
  }
}

export const testExports = {
  getNodeEdges,
  getTubeNode,
  getEdges,
};
