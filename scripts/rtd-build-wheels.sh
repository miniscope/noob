#!/usr/bin/env bash
# Provision the pyodide-declared emscripten + rust toolchain, then build the
# docs wheels. Invoked by Read the Docs' pre_build job (see .readthedocs.yaml).

set -euo pipefail

EMSCRIPTEN_VERSION="$(pyodide config get emscripten_version)"
RUST_TOOLCHAIN="$(pyodide config get rust_toolchain)"

rm -rf ~/emsdk
git clone --depth 1 https://github.com/emscripten-core/emsdk.git ~/emsdk
~/emsdk/emsdk install "$EMSCRIPTEN_VERSION"
~/emsdk/emsdk activate "$EMSCRIPTEN_VERSION"
source ~/emsdk/emsdk_env.sh

rustup toolchain install "$RUST_TOOLCHAIN"
rustup target add wasm32-unknown-emscripten --toolchain "$RUST_TOOLCHAIN"

# RTD installs into the tool's own env; point pdm at it (as the other jobs do).
VIRTUAL_ENV="$(dirname "$(dirname "$(which python)")")"
export VIRTUAL_ENV
pdm run docs-build-wheels
