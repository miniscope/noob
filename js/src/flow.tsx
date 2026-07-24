// based on
// https://reactflow.dev/examples/layout/elkjs-multiple-handles

import type { NodeUnion, TubeSpecification } from "./types/types.ts";
import {
  Background,
  ConnectionMode,
  Controls,
  type Edge,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import { useEffect, useMemo } from "react";
import ElkNode from "./nodes/elk.tsx";
import TitleNode from "./nodes/title.tsx";
import useLayoutNodes from "./useLayoutNodes.tsx";
import { tubeToFlow } from "./tube.tsx";
import { InputEdge, ReturnEdge, RuntimeEdge } from "./edge.tsx";
import type { TubeRunnerHandle } from "./runner/base.ts";
import { ReturnPanel, RunnerControls } from "./runner/controls.tsx";
import { buildSlotSources, RuntimeContext } from "./runner/runtime.tsx";
import { RuntimeStore } from "./runner/events.ts";
import { EventTable } from "./runner/table.tsx";

interface NoobFlowProps {
  tube: TubeSpecification;
  color: "dark" | "light";
  /** Attach a runner: adds lifecycle controls, live runtime display, and the event table */
  runner?: TubeRunnerHandle;
  // Whether the title should be displayed
  // (false for compact tubes, e.g. in the docs)
  title: boolean;
}

const nodeTypes = {
  elk: ElkNode,
  title: TitleNode,
  group: ElkNode,
};

const edgeTypes = {
  default: RuntimeEdge,
  inputEdge: InputEdge,
  returnEdge: ReturnEdge,
};

/**
 * The tube view: a static graph of the specification, or — when a runner
 * is attached — a runnable one. The pipeline is runner → event stream →
 * store → per-slice subscribers; this component wires those together.
 */
export function NoobFlow({ tube, color, title, runner }: NoobFlowProps) {
  const [flowEdges, flowNodes] = useMemo(
    () => tubeToFlow(tube, title),
    [tube, title],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState<NodeUnion>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  // stable while the tube is: high-rate updates flow through the store's
  // per-slice subscriptions, never through provider invalidation
  const context = useMemo(
    () =>
      runner
        ? {
            store: new RuntimeStore(),
            slotSources: buildSlotSources(flowEdges),
          }
        : null,
    [runner, flowEdges],
  );
  useEffect(() => {
    if (runner && context) return runner.events.subscribe(context.store.push);
  }, [runner, context]);

  useLayoutNodes();
  const graph = (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      colorMode={color}
      connectionMode={ConnectionMode.Loose} // allow the inputs/returns of nested tubes to connect both ways
      minZoom={0.1}
      maxZoom={10}
      fitView
    >
      <Background />
      <Controls />
      {runner && <ReturnPanel handle={runner} />}
    </ReactFlow>
  );

  return (
    <RuntimeContext.Provider value={context}>
      {runner ? (
        <div className="noob-runnable">
          <RunnerControls handle={runner} />
          <div className="noob-runnable-flow">{graph}</div>
          <EventTable />
        </div>
      ) : (
        graph
      )}
    </RuntimeContext.Provider>
  );
}
