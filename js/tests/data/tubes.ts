import type { TubeSpecification } from "../../src/types";

// placeholder value, not used in tests yet
const nodeinfo = { node_id: "", type: "", signals: {}, slots: {} };

export const recursiveTube: TubeSpecification = {
  noob_id: "testing-recursive-parent",
  noob_model: "noob.tube.TubeSpecification",
  noob_version: "0.0.1.dev155+g2f9639f",
  input: {
    parent_start: { id: "parent_start", type: "int", scope: "tube" },
    parent_multiply: { id: "parent_multiply", type: "int", scope: "process" },
    child_start: { id: "child_start", type: "int", scope: "tube" },
    child_multiply: { id: "child_multiply", type: "int", scope: "process" },
  },
  assets: {},
  nodes: {
    a: {
      type: "noob.testing.count_source",
      id: "a",
      params: { start: "input.parent_start" },
      nodeinfo,
    },
    b: {
      type: "tube",
      id: "b",
      depends: [
        { child_multiply_inner: "a.index" },
        { child_multiply_input: "input.child_multiply" },
      ],
      nodeinfo,
      params: {
        tube: {
          noob_id: "testing-recursive-child",
          noob_model: "noob.tube.TubeSpecification",
          noob_version: "0.0.1.dev155+g2f9639f",
          input: {
            child_start: { id: "child_start", type_: "int", scope: "tube" },
            child_multiply_inner: {
              id: "child_multiply_inner",
              type_: "int",
              scope: "process",
            },
            child_multiply_input: {
              id: "child_multiply_input",
              type_: "int",
              scope: "process",
            },
          },
          nodes: {
            a: {
              type: "noob.testing.count_source",
              id: "a",
              params: { start: "input.child_start" },
              nodeinfo,
            },
            b: {
              type: "noob.testing.multiply",
              id: "b",
              depends: [
                { left: "a.index" },
                { right: "input.child_multiply_inner" },
              ],
              nodeinfo,
            },
            c: {
              type: "noob.testing.multiply",
              id: "c",
              depends: [
                { left: "b.value" },
                { right: "input.child_multiply_input" },
              ],
              nodeinfo,
            },
            d: { type: "return", id: "d", depends: "c.value", nodeinfo },
          },
        },
      },
    },
    c: {
      type: "noob.testing.multiply",
      id: "c",
      depends: [{ left: "b.value" }, { right: "input.parent_multiply" }],
      nodeinfo,
    },
    d: {
      type: "return",
      id: "d",
      depends: [
        { index: "a.index" },
        { child: "b.value" },
        { parent: "c.value" },
      ],
      nodeinfo,
    },
  },
};
