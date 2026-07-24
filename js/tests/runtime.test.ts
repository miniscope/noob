import { describe, expect, it } from "vitest";

import {
  buildSlotSources,
  EVENT_BUFFER_SIZE,
  formatEpoch,
  formatValue,
} from "../src/runner/runtime.tsx";
import type { Event } from "../src/types/events.ts";
import { Stream } from "../src/runner/events.ts";
import { RuntimeStore } from "../src/runner/events.ts";

function event(overrides: Partial<Event>): Event {
  return {
    id: 1,
    timestamp: "2026-07-15T00:00:00.000000",
    node_id: "a",
    signal: "index",
    epoch: 0,
    value: 42,
    ...overrides,
  };
}

describe(RuntimeStore, () => {
  it("records node epoch and signal value per emission", () => {
    const store = new RuntimeStore();
    store.push([event({ node_id: "a", signal: "index" })]);

    expect(store.getNode("a")).toStrictEqual({ epoch: 0, seq: 1 });
    expect(store.getSignal("a.signals.index")).toStrictEqual({
      value: 42,
      seq: 1,
    });
    expect(store.getEvents()).toHaveLength(1);
  });

  it("notifies only subscribers of touched slices", () => {
    const store = new RuntimeStore();
    const calls: string[] = [];
    store.subscribeKey("node:a", () => calls.push("a"));
    store.subscribeKey("node:b", () => calls.push("b"));
    store.subscribeKey("signal:a.signals.index", () => calls.push("a.index"));

    store.push([event({ node_id: "a", signal: "index" })]);
    store.flush();

    expect(calls.sort()).toStrictEqual(["a", "a.index"]);
  });

  it("coalesces multiple batches into one notification flush", () => {
    const store = new RuntimeStore();
    let calls = 0;
    store.subscribeKey("node:a", () => (calls += 1));

    store.push([event({})]);
    store.push([event({ value: 43 })]);
    store.flush();

    expect(calls).toBe(1);
    expect(store.getSignal("a.signals.index")?.value).toBe(43);
  });

  it("stops notifying after unsubscribe", () => {
    const store = new RuntimeStore();
    let calls = 0;
    const unsubscribe = store.subscribeKey("node:a", () => (calls += 1));
    unsubscribe();

    store.push([event({})]);
    store.flush();

    expect(calls).toBe(0);
  });

  it("keeps meta events in the buffer but off the graph state", () => {
    const store = new RuntimeStore();
    store.push([
      event({ node_id: "meta", signal: "NodeReady", value: "b", epoch: 3 }),
    ]);

    expect(store.getNode("meta")).toBeUndefined();
    expect(store.getEvents()).toHaveLength(1);
  });

  it("caps the event buffer", () => {
    const store = new RuntimeStore();
    store.push(
      Array.from({ length: EVENT_BUFFER_SIZE + 20 }, (_, i) =>
        event({ id: i }),
      ),
    );

    expect(store.getEvents()).toHaveLength(EVENT_BUFFER_SIZE);
  });
});

describe(Stream, () => {
  it("delivers pushed values to subscribers until unsubscribed", () => {
    const stream = new Stream<number>();
    const seen: number[] = [];
    const unsubscribe = stream.subscribe((v) => seen.push(v));

    stream.push(1);
    unsubscribe();
    stream.push(2);

    expect(seen).toStrictEqual([1]);
  });
});

describe(buildSlotSources, () => {
  it("maps slot handles to the signal handles feeding them", () => {
    const sources = buildSlotSources([
      {
        id: "a.signals.index-b.slots.left",
        source: "a",
        sourceHandle: "a.signals.index",
        target: "b",
        targetHandle: "b.slots.left",
      },
    ]);

    expect(sources["b.slots.left"]).toBe("a.signals.index");
  });
});

describe("formatting", () => {
  it("formats wire values compactly", () => {
    expect(formatValue(42)).toBe("42");
    expect(formatValue("pck__abc123")).toBe("⟨pickled⟩");
    expect(formatValue("x".repeat(100))).toHaveLength(24);
  });

  it("formats root and nested epochs", () => {
    expect(formatEpoch(0)).toBe("0");
    expect(formatEpoch([0, ["mapped", 2]])).toBe("0/mapped:2");
  });
});
