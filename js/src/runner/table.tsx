import { formatEpoch, formatValue, useRuntimeEvents } from "./runtime.tsx";
import { Fragment, useState } from "react";
import type { Event } from "../types/events.ts";

/**
 * Collapsible table of runner events, newest first. Subscribes to the
 * store's event buffer (frame-coalesced); each row expands to the full
 * event object as received on the wire.
 */
export function EventTable() {
  const events = useRuntimeEvents();
  const [expanded, setExpanded] = useState<string | null>(null);
  return (
    <details className="noob-event-table">
      <summary>events ({events.length})</summary>
      <table>
        <thead>
          <tr>
            <th>time</th>
            <th>node</th>
            <th>signal</th>
            <th>epoch</th>
            <th>value</th>
          </tr>
        </thead>
        <tbody>
          {events
            .map((event, index) => (
              <EventRow
                event={event}
                index={index}
                expanded={expanded}
                setExpanded={setExpanded}
              />
            ))
            .reverse()}
        </tbody>
      </table>
    </details>
  );
}

function EventRow({ event, index, expanded, setExpanded }: EventRowProps) {
  const key = `${event.id}-${index}`;
  return (
    <Fragment key={key}>
      <tr
        className={event.node_id === "meta" ? "meta-event" : undefined}
        onClick={() => setExpanded(expanded === key ? null : key)}
      >
        <td>{event.timestamp.slice(11, 23)}</td>
        <td>{event.node_id}</td>
        <td>{event.signal}</td>
        <td>{formatEpoch(event.epoch)}</td>
        <td>{formatValue(event.value, 48)}</td>
      </tr>
      {expanded === key && (
        <tr className="event-detail">
          <td colSpan={5}>
            <pre>{JSON.stringify(event, null, 2)}</pre>
          </td>
        </tr>
      )}
    </Fragment>
  );
}

interface EventRowProps {
  event: Event;
  index: number;
  expanded: string | null;
  setExpanded: (expanded: string | null) => void;
}
