# Contributing

```{warning}
Contributing docs are a work in progress and are not complete!
Maintainers reserve the right to create and enforce arbitrary rules for the moment.
```

## Developing

```{todo}
Document setting up dev environment, linting
```

## Raising a PR

```{todo}
Document issue/PR flow
```

## `noob-core`

```{todo}
Document basic noob-core dev practices
```

## Docs

### Pyodide examples

Runnable tubes can be embedded in the docs using the `noob-tube` directive
with the `:runnable:` flag

````
```{noob-tube} tube-id
:runnable:
```
````

The tube should be placed in `docs/assets/pipelines`,
and should only reference nodes that are present in the noob installation
(i.e., in `nobes`, `noob.testing`, or one of `noob`'s obligate dependencies).

The docs require built wheels for both `noob` and `noob-core` to be placed in
`docs/_static/wheels` - to do this use `pdm docs-build-wheels` which calls `scripts/build-docs-wheels.sh`

```shell
# build the wheels
pdm docs-build-wheels
# auto-rebuild and serve the docs and JS
pdm docs-js
```

If these wheels are absent, the docs will try and use the latest wheels from PyPI,
which may be enough if you have not made changes to the relevant code.

To build the wasm/emscripten wheels, 
you must have `emsdk` installed and the correct version of `emscripten` installed with it.

See the part of the `build-docs-wheels.sh` script that's gated to readthedocs builds for that,
that script doesn't modify the local environment on purpose to avoid surprises,
but the wasm build for pyodide is a bit finnicky and has to match the emscripten, rust, and pyodide versions.

Assuming `emsdk` is installed at `~/emsdk`, the basic pattern is something like this:

```shell
EMSCRIPTEN_VERSION="$(pyodide config get emscripten_version)"
RUST_TOOLCHAIN="$(pyodide config get rust_toolchain)"

~/emsdk/emsdk install "$EMSCRIPTEN_VERSION"
~/emsdk/emsdk activate "$EMSCRIPTEN_VERSION"
source ~/emsdk/emsdk_env.sh

rustup toolchain install "$RUST_TOOLCHAIN" --target wasm32-unknown-emscripten
rustup default "$RUST_TOOLCHAIN"

pdm docs-build-wheels
```

## Adding New Nobes Subpackages

- Create the package with the template: `pdm run new_nobes`
- Add dependencies in `nobes` package (see existing entries for examples)
  - Bare dependency to package in `[project.dependencies]`
  - Local dependency in `[tool.pdm.dev-dependencies]`
- Add overrides in root `pyproject.toml`
- Add to docs: `docs/nobes/{the new package shortname}`
- Write at least one test!

All dependencies for the nobes subpackage should be declared in the `pyproject.toml` for that subpackage,
and the top-level lockfile should be updated whenever they are!

```bash
pdm lock --with :all
```