import { createRoot } from "react-dom/client";
import { NoobFlow } from "./flow.tsx";
import { ReactFlowProvider } from "@xyflow/react";

import type { TubeSpecification } from "./types.ts";

import "./index.css";

export const renderPipeline = (selector: string, tube: TubeSpecification, color: "dark" | "light" = "dark") => {
  const node = document.querySelector(selector);
  if (node === null) {
    throw Error("selector not found");
  }
  const root = createRoot(node);
  root.render(
    <ReactFlowProvider>
      <NoobFlow tube={tube} color={color}></NoobFlow>
    </ReactFlowProvider>,
  );
};