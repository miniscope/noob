import type { TubeSpecification } from "./types";

declare global {
  interface Window {
    renderPipeline: (selector: string, tube: TubeSpecification) => void;
  }
}
