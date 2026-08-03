/**
 * In-page tube runner: a pyodide interpreter with noob installed, driven
 * directly.
 *
 * `enable()` is lazy: nothing downloads before it (pyodide from a pinned
 * CDN, noob + noob-core wheels via micropip). Free-running is a client-side
 * `setInterval` loop — pyodide has no threads. `process()` runs
 * synchronously on the main thread: fine for docs-scale tubes, a web worker
 * is the future home for heavier ones.
 */
import type { TubeRunnerProxy, TubeSpecification } from "../types/types.ts";
import type {
  RunnerError,
  RunnerStatus,
  StatusChange,
  TubeRunnerHandle,
} from "./base.ts";
import type { Event } from "../types/events.ts";
import type { PyodideInterface } from "pyodide";
import type { PyDict } from "pyodide/ffi";
import { Stream } from "./events.ts";

/**
 * Pin of the pyodide runtime the GUI loads. Keep in sync with the
 * pyodide-build xbuildenv producing the noob-core wasm wheel — both must
 * target the same pyemscripten ABI.
 */
export const DEFAULT_PYODIDE_URL =
  "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/pyodide.mjs";

export interface PythonError extends Error {
  type?: string;
}

/**
 * noob dependencies that ship in the pyodide distribution, loaded through
 * pyodide (which brings their transitive deps) before micropip resolves the
 * rest — micropip satisfies these from the distribution without pulling
 * what they themselves import.
 */
const PYODIDE_STOCK_DEPS = ["micropip", "pydantic", "rich", "pygments"];

let pyodidePromise: Promise<PyodideInterface> | null = null;

/** Load the pyodide runtime once per page, shared by all runners. */
function loadPyodideOnce(url: string): Promise<PyodideInterface> {
  if (!pyodidePromise) {
    pyodidePromise = import(/* @vite-ignore */ url).then(
      (mod: {
        loadPyodide: (opts: { indexURL: string }) => Promise<PyodideInterface>;
      }) =>
        mod.loadPyodide({ indexURL: url.slice(0, url.lastIndexOf("/") + 1) }),
    );
  }
  return pyodidePromise;
}

export interface PyodideRunnerOptions {
  spec: TubeSpecification;
  /** Wheel URLs for micropip: noob + noob-core wasm; other deps resolve from PyPI/pyodide. */
  wheels: string[];
  pyodideUrl?: string;
  /** Pacing of the client-side free-run loop. */
  intervalMs?: number;
  /** Any additional dependencies to install into the environment from pypi */
  extra_deps?: string[];
}

export class PyodideRunner implements TubeRunnerHandle {
  readonly transport = "pyodide";
  readonly events = new Stream<Event[]>();
  readonly returns = new Stream<unknown>();
  readonly statusChanges = new Stream<StatusChange>();
  readonly errors = new Stream<RunnerError>();

  #status: RunnerStatus = "unloaded";
  #opts: PyodideRunnerOptions;
  #session: TubeRunnerProxy | null = null;
  /** This runner's private python globals, isolating it from co-hosted runners in the shared interpreter. */
  #namespace: PyDict | null = null;
  #loop: ReturnType<typeof setInterval> | null = null;
  #enabling: Promise<void> | null = null;

  constructor(options: PyodideRunnerOptions) {
    this.#opts = options;
  }

  get status(): RunnerStatus {
    return this.#status;
  }

  enable(): Promise<void> {
    this.#enabling ??= this.#enable().catch((e: PythonError) => {
      this.#enabling = null;
      this.#setStatus("unloaded");
      this.#notifyError(e);
      throw e;
    });
    return this.#enabling;
  }

  async #enable(): Promise<void> {
    this.#setStatus("loading", "loading pyodide runtime");
    const pyodide: PyodideInterface = await loadPyodideOnce(
      this.#opts.pyodideUrl ?? DEFAULT_PYODIDE_URL,
    );

    for (const dep of PYODIDE_STOCK_DEPS) {
      this.#setStatus("loading", "installing " + dep);
      await pyodide.loadPackage(dep);
    }

    if (this.#opts.extra_deps !== undefined) {
      await pyodide.loadPackage("micropip");
      const micropip = pyodide.pyimport("micropip") as Micropip;

      for (const dep of this.#opts.extra_deps) {
        this.#setStatus("loading", "installing " + dep);
        await micropip.install(dep);
      }
    }

    if (this.#opts.wheels.length === 0) {
      this.#setStatus("loading", "installing noob from pypi");
      await pyodide.loadPackage(["noob", "noob-core"]);
    } else {
      this.#setStatus("loading", "installing noob from local wheels");
      // eslint-disable-next-line @typescript-eslint/no-unsafe-call
      pyodide.globals.set("_noob_wheels", pyodide.toPy(this.#opts.wheels)); // eslint
      await pyodide.runPythonAsync(
        "import micropip\nawait micropip.install(_noob_wheels)",
      );
    }

    this.#setStatus("loading", "creating session");

    // Do a combination of python and js adapting:
    // from the python side, we convert to JSON using the pydantic adapter,
    // which handles pickling any non-jsonable events,
    // and then here we parse the json from a string so that
    // the event store can operate only on events -
    // all the special behavior for pyodide is contained within the pyodide adapter
    // so that we can have event streams from within the browser and from the python server
    const event_cb = (event: string) => {
      const data: Event[] = JSON.parse(event) as Event[];
      this.events.push(data);
    };

    // Each runner shares the one interpreter but runs in its own globals dict:
    this.#namespace = pyodide.toPy({
      event_cb,
      tube_spec: JSON.stringify(this.#opts.spec),
    }) as PyDict;
    this.#session = (await pyodide.runPythonAsync(
      `
    import json
    from pydantic import TypeAdapter

    from noob.event import EventUnion
    from noob.runner import SynchronousRunner
    from noob import Tube

    tube = Tube.from_specification(json.loads(tube_spec))
    runner = SynchronousRunner(tube)
    event_adapter = TypeAdapter[list[EventUnion]](list[EventUnion])

    def callback(event):
        event_cb(event_adapter.dump_json([event]).decode("utf-8"))

    runner.add_callback(callback)
    runner
    `,
      { globals: this.#namespace },
    )) as TubeRunnerProxy;
    this.#emitStatus();
  }

  #run(action: (session: TubeRunnerProxy) => void): void {
    if (!this.#session) {
      throw new Error("pyodide is not loaded; call enable() first");
    }
    try {
      action(this.#session);
    } catch (e) {
      this.#notifyError(e as PythonError);
      throw e;
    }
  }

  init(): Promise<void> {
    this.#run((session) => session.init());
    this.#emitStatus();
    return Promise.resolve();
  }

  deinit(): Promise<void> {
    this.#stopLoop();
    this.#run((session) => session.deinit());
    this.#emitStatus();
    return Promise.resolve();
  }

  process(): Promise<void> {
    this.#run((session) => {
      const result = session.process();
      this.returns.push(result);
    });
    return Promise.resolve();
  }

  start(intervalMs?: number): void {
    if (this.#loop !== null) return;
    this.#setStatus("running");
    this.#loop = setInterval(
      () => {
        void this.process().catch(() => this.stop());
      },
      intervalMs ?? this.#opts.intervalMs ?? 200,
    );
  }

  stop(): void {
    this.#stopLoop();
    this.#emitStatus();
  }

  #stopLoop(): void {
    if (this.#loop !== null) {
      clearInterval(this.#loop);
      this.#loop = null;
    }
  }

  #setStatus(status: RunnerStatus, detail?: string): void {
    this.#status = status;
    this.statusChanges.push({ status, detail });
  }

  #emitStatus(): void {
    if (this.#session)
      this.#setStatus(this.#session.running ? "ready" : "stopped");
  }

  #notifyError(e: PythonError): void {
    const err = {
      message: e.type ?? "Error",
      traceback: e.message ?? String(e),
    };
    this.errors.push(err);
  }

  dispose(): void {
    this.#stopLoop();
    this.#session?.destroy();
    this.#session = null;
    this.#namespace?.destroy();
    this.#namespace = null;
    this.events.close();
    this.statusChanges.close();
    this.errors.close();
    this.returns.close();
  }
}

interface Micropip {
  install: (module: string) => Promise<void>;
}
