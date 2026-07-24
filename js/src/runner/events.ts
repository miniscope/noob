import { isMetaEvent, type Event } from "../types/events.ts";
import {
  EVENT_BUFFER_SIZE,
  type NodeRuntime,
  type SignalRuntime,
} from "./runtime.tsx";

/* No-arg callback change notifier */
export type Listener = () => void;

export type Unsubscribe = () => void;

export class Stream<T> {
  #subscribers = new Set<(value: T) => void>();

  push(value: T): void {
    this.#subscribers.forEach((cb) => cb(value));
  }

  subscribe(cb: (value: T) => void): Unsubscribe {
    this.#subscribers.add(cb);
    return () => this.#subscribers.delete(cb);
  }

  close(): void {
    this.#subscribers.clear();
  }
}

/*
Event broker that receives events and routes them to the individual nodes that subscribe to them.

TODO: Sort of shitty, the MVP for getting tubes to run in the docs,
but will need to be reworked for smarter event handling when we implement the GUI runner on the server -
i.e. don't waste time pickling values, etc.
 */
export class RuntimeStore {
  #nodes = new Map<string, NodeRuntime>();
  /** keyed by source handle id: `${node_id}.signals.${signal}` */
  #signals = new Map<string, SignalRuntime>();
  #events: Event[] = [];
  #seq = 0;
  #listeners = new Map<string, Set<Listener>>();
  #dirty = new Set<string>();
  #flushScheduled = false;

  /** Fold one batch of events in; notify affected subscribers next frame. */
  push = (batch: Event[]): void => {
    this.#seq += 1;
    for (const event of batch) {
      // meta events only show in the event buffer, not on the graph
      if (isMetaEvent(event)) continue;
      this.#nodes.set(event.node_id, { epoch: event.epoch, seq: this.#seq });
      const handleId = `${event.node_id}.signals.${event.signal}`;
      this.#signals.set(handleId, { value: event.value, seq: this.#seq });
      this.#dirty.add(`node:${event.node_id}`);
      this.#dirty.add(`signal:${handleId}`);
    }
    if (batch.length > 0) {
      this.#events = [...this.#events, ...batch].slice(-EVENT_BUFFER_SIZE);
      this.#dirty.add("events");
    }
    this.#scheduleFlush();
  };

  subscribeKey(key: string, cb: Listener): Unsubscribe {
    let listeners = this.#listeners.get(key);
    if (!listeners) {
      listeners = new Set();
      this.#listeners.set(key, listeners);
    }
    listeners.add(cb);
    return () => listeners.delete(cb);
  }

  getNode(id: string): NodeRuntime | undefined {
    return this.#nodes.get(id);
  }

  getSignal(handleId: string): SignalRuntime | undefined {
    return this.#signals.get(handleId);
  }

  getEvents(): Event[] {
    return this.#events;
  }

  /** Deliver pending notifications immediately (normally rAF-coalesced). */
  flush = (): void => {
    this.#flushScheduled = false;
    const dirty = this.#dirty;
    this.#dirty = new Set();
    dirty.forEach((key) => this.#listeners.get(key)?.forEach((cb) => cb()));
  };

  #scheduleFlush(): void {
    if (this.#flushScheduled) return;
    this.#flushScheduled = true;
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(this.flush);
    } else {
      setTimeout(this.flush, 16);
    }
  }
}
