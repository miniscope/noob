import { createRoot } from "react-dom/client";
import { NoobFlow } from "./flow.tsx";
import View from "./pages/view.tsx";
import { PyodideRunner, type PyodideRunnerOptions } from "./runner/pyodide.ts";
import { ReactFlowProvider } from "@xyflow/react";

import type { TubeSpecification } from "./types/types.ts";

import "@xyflow/react/dist/style.css";
import "./css/index.css";

export type RunnableOptions = Omit<PyodideRunnerOptions, "spec">;

function mount(selector: string, element: React.ReactNode) {
  const node = document.querySelector(selector);
  if (node === null) {
    throw Error("selector not found");
  }
  createRoot(node).render(<ReactFlowProvider>{element}</ReactFlowProvider>);
}

export const renderPipeline = (
  selector: string,
  tube: TubeSpecification,
  title = false,
  color: "dark" | "light" = "dark",
) => {
  mount(selector, <NoobFlow tube={tube} color={color} title={title} />);
};

/**
 * Render a tube that can be run in-page with a lazily loaded pyodide
 * runtime. Nothing is downloaded until the user clicks "load pyodide".
 */
export function renderRunnableTube(
  selector: string,
  tube: TubeSpecification,
  options: RunnableOptions,
  title = false,
  color: "dark" | "light" = "dark",
) {
  const runner = new PyodideRunner({ spec: tube, ...options });
  mount(
    selector,
    <NoobFlow tube={tube} color={color} title={title} runner={runner} />,
  );
}

export function initView(
  selector: string,
  tube_id: string,
  color: "dark" | "light" = "dark",
) {
  mount(selector, <View tube_id={tube_id} color={color} />);
}
