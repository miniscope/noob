#!/usr/bin/env bash
# Build the wheels the in-browser pyodide runner installs — noob (pure python)
# and the noob-core wasm wheel — and collect them into docs/_static/wheels,
# where the plot directive globs them. Run locally with
# `pdm run docs-build-wheels`; Read the Docs runs the same entry point.
set -euo pipefail

# On Read the Docs, provision the pyodide-declared emscripten + rust toolchain.
if [ -n "${READTHEDOCS:-}" ]; then
  EMSCRIPTEN_VERSION="$(pyodide config get emscripten_version)"
  RUST_TOOLCHAIN="$(pyodide config get rust_toolchain)"

  rm -rf ~/emsdk
  git clone --depth 1 https://github.com/emscripten-core/emsdk.git ~/emsdk
  ~/emsdk/emsdk install "$EMSCRIPTEN_VERSION"
  ~/emsdk/emsdk activate "$EMSCRIPTEN_VERSION"
  source ~/emsdk/emsdk_env.sh

  # Add the wasm target to the pyodide toolchain AND make it the default: the
  # target must live on the toolchain that actually compiles, otherwise the
  # build hits "can't find crate for `core`" using RTD's default toolchain.
  rustup toolchain install "$RUST_TOOLCHAIN" --target wasm32-unknown-emscripten
  rustup default "$RUST_TOOLCHAIN"
fi

# pdm build and pyodide build each wipe their own output dir, so they can't
# share one — build into separate dist/ dirs, then collect into _static/wheels.
rm -rf docs/_static/wheels
mkdir -p docs/_static/wheels

pdm build --project packages/noob --dest packages/noob/dist --no-sdist

# A dev-like profile (fast, low-optimization); the docs wheel only runs example
# tubes, so it doesn't need the workspace's fat-LTO release optimization.
CARGO_PROFILE_RELEASE_OPT_LEVEL=0 \
CARGO_PROFILE_RELEASE_LTO=false \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 \
CARGO_BUILD_JOBS=1 \
  pyodide build packages/noob-core --outdir packages/noob-core/dist

cp packages/noob/dist/*.whl packages/noob-core/dist/*.whl docs/_static/wheels/
