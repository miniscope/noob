import { recursiveTube } from "./data/tubes";
import { test, beforeAll, describe, expect } from "vitest";
import { tubeToFlow, testExports } from "../src/tube";
import type { TubeNode, NodeUnion } from "../src/types.ts";
import type { Edge } from "@xyflow/react";

const { getNodeEdges, getTubeNode, getEdges } = testExports;

let recursiveNodes: NodeUnion[], recursiveEdges: Edge[], tubeNode: TubeNode;

beforeAll(() => {
  [recursiveEdges, recursiveNodes] = tubeToFlow(recursiveTube);
  tubeNode = Object.values(recursiveTube.nodes).find(
    (n) => n.type === "tube",
  ) as TubeNode;
});

const baseNode = { id: "test", type: "module.function" };

describe("getNodeEdges", () => {
  test("handles string dependencies as value signals", () => {
    const edges = getNodeEdges({ ...baseNode, depends: "b.something" });

    expect(edges).length(1);
    expect(edges[0]).toHaveProperty("targetHandle", "test.slots.value");
  });

  describe("with prefix", () => {
    test("prefixes sources and targets", () => {
      const edges = getNodeEdges(
        { ...baseNode, depends: [{ slot: "b.something" }] },
        "prefix",
      );

      expect(edges).length(1);
      expect(edges[0]).toMatchObject({
        id: "prefix.b.signals.something-prefix.test.slots.slot",
        source: "prefix.b",
        sourceHandle: "prefix.b.signals.something",
        target: "prefix.test",
        targetHandle: "prefix.test.slots.slot",
      });
    });

    test("unnests input dependencies with prefix", () => {
      const edges = getNodeEdges(
        { ...baseNode, depends: [{ slot: "input.something" }] },
        "prefix",
      );

      expect(edges[0]).toHaveProperty("source", "prefix");
      expect(edges[0]).toHaveProperty("sourceHandle", "prefix.slots.something");
    });

    test("unnests return targets with prefix", () => {
      const edges = getNodeEdges(
        { id: "test", type: "return", depends: [{ slot: "b.something" }] },
        "prefix",
      );

      expect(edges[0]).toHaveProperty("target", "prefix");
      expect(edges[0]).toHaveProperty("targetHandle", "prefix.signals.slot");
    });
  });
});

describe("getTubeNode", () => {
  let nodes: NodeUnion[];

  beforeAll(() => {
    nodes = getTubeNode(tubeNode, recursiveEdges);
  });

  test("removes a return node", () => {
    const return_node = Object.values(tubeNode.params.tube.nodes).find(
      (n) => n.type === "return",
    ) as NodeUnion;

    expect(return_node).toBeDefined();
    expect(nodes).length(4);
    expect(nodes.map((n) => n.id)).not.toContain(`b.${return_node.id}`);
  });

  test("creates an outer group node", () => {
    expect(nodes[0].type).toBe("group");
    expect(nodes[0].id).toStrictEqual(tubeNode.id);
  });

  test("creates inner elk nodes with parentId set", () => {
    expect(nodes.slice(1).every((n) => n.parentId === tubeNode.id)).toBe(true);
  });
});

describe("getEdges", () => {
  test("dispatches on tube nodes", () => {
    const edges = getEdges({ tube: tubeNode });

    expect(edges).length(7);
    // outer edges
    expect(edges[0]).toMatchObject({ source: "a", target: "b" });
    expect(edges[1]).toMatchObject({ source: "input", target: "b" });
    // inner edges
    expect(edges.slice(2).every((e) => e.id.startsWith("tube"))).toBe(true);
  });

  test("differentiates same-named slots and signals", () => {
    const edges = getEdges({
      a: { id: "a", type: "test.test", depends: [{ value: "z.value" }] },
      b: { id: "b", type: "test.test", depends: [{ value: "a.value" }] },
    });

    expect(edges[0]).toHaveProperty("targetHandle", "a.slots.value");
    expect(edges[1]).toHaveProperty("sourceHandle", "a.signals.value");
  });
});

describe(tubeToFlow, () => {
  test("connects nested inputs/returns with outer dependencies", () => {
    const groupNode = recursiveNodes.find(
      (n) => n.type == "group",
    ) as NodeUnion;

    expect(groupNode.data.sourceHandles).toMatchObject([
      { id: "b.signals.value", label: "value", key: "b.signals.value" },
    ]);
    expect(groupNode.data.targetHandles).toMatchObject([
      {
        id: "b.slots.child_multiply_inner",
        label: "child_multiply_inner",
        key: "b.slots.child_multiply_inner",
      },
      {
        id: "b.slots.child_multiply_input",
        label: "child_multiply_input",
        key: "b.slots.child_multiply_input",
      },
    ]);

    // depends on other top-level nodes -> child input
    // just checking that there exists edges between handles (asserted above) are created.
    const abEdge = recursiveEdges.find(
      (e) => e.id === "a.signals.index-b.slots.child_multiply_inner",
    );

    expect(abEdge).toMatchObject({
      source: "a",
      target: "b",
      sourceHandle: "a.signals.index",
      targetHandle: "b.slots.child_multiply_inner",
    });

    // other nodes depend on nested node's returns
    expect(
      recursiveEdges.some((e) => e.sourceHandle === "b.signals.value"),
    ).toBe(true);
  });
});
