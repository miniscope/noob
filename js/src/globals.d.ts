import type { TubeSpecification } from "./types/types.ts";
import type { RunnableOptions } from "./main.tsx";

declare global {
  interface Window {
    renderPipeline: (selector: string, tube: TubeSpecification) => void;
    renderRunnableTube: (
      selector: string,
      tube: TubeSpecification,
      options: RunnableOptions,
      title: boolean,
      color?: "dark" | "light",
    ) => void;
    initView: (selector: string, tube_id: string) => void;
  }
}
