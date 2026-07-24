import { useEffect, useState } from "react";
import type { TubeSpecification } from "../types/types.ts";
import { NoobFlow } from "../flow.tsx";
import type { SpecMsg } from "../types/events.ts";

interface ViewProps {
  tube_id: string;
  color: "dark" | "light";
}

/**
 * Live viewer that refreshes a tube definition from a websocket
 */
export default function View(props: ViewProps) {
  const [spec, setSpec] = useState<TubeSpecification | null>(null);

  useEffect(() => {
    const socket = new WebSocket(`/spec/${props.tube_id}`);
    socket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") return;
      const spec = JSON.parse(event.data) as SpecMsg;
      setSpec(spec.value);
    });
  }, [props.tube_id]);

  return spec && <NoobFlow tube={spec} color={props.color} title={true} />;
}
