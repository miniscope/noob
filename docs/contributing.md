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

### Changelog

Each package has its own {doc}`changelog`, versioned and released
independently. Entries are [towncrier](https://towncrier.readthedocs.io)
fragments, and both the pending entries and the rendered `CHANGELOG.md` for a
package live in `changelog/<package>/`.

A PR that touches `packages/<package>/` needs at least one entry for that
package, and CI will fail without one. A PR touching two packages needs an
entry in each. Changes outside `packages/` - docs, CI, the JS frontend - belong
to no package and need no entry.

An entry is one file named `{pr number}.{type}.md`, where `type` is one of
`added`, `changed`, `deprecated`, `removed`, `fixed`, `perf`, `docs`, or `ci`:

```shell
# one entry
echo "Optional propagation now honors slots." > changelog/noob/263.fixed.md
# ... and a second one from the same PR
echo "Deduplicate expired nodes." > changelog/noob/263.fixed.2.md
```

The file contains only the text of the entry - towncrier adds the bullet and
the link to the PR.

```shell
pdm run changelog draft        # preview every package's pending entries
pdm run changelog draft noob   # just one
pdm run changelog check        # what CI runs
```

A PR that genuinely doesn't need an entry (a typo fix, a CI-only tweak) can be
labelled `no changelog` to skip the check.

For a change with no PR to point at, prefix the name with `+` to make it an
orphan entry: `+some-slug.changed.md` renders without a link.

### Releasing

Each package is released by pushing its own tag - `v*` for `noob`, `core-v*`
for `noob-core`, `nobes-video-v*` for `nobes-video`, and so on. The tag is the
only place the version is written; nothing in the tree is bumped.

From an up-to-date `main`, one command cuts a release:

```shell
pdm run release noob-core --minor   # 0.2.0 -> 0.3.0
pdm run release noob-core 0.3.0rc1  # or say it outright
pdm run release noob-core --minor --dry-run
```

It works out the next version from the package's newest tag, consumes its
pending entries into its `CHANGELOG.md`, commits, tags, and pushes the commit
and the tag. Pushing the tag is what starts the publish workflow, which stamps
the version from the tag and uploads to PyPI.
Once PyPI has it,
[`github-release.yml`](https://github.com/miniscope/noob/blob/main/.github/workflows/github-release.yml)
creates the GitHub Release: that version's changelog section as the body, the same
wheels and sdists that went to PyPI attached to it, and anything that isn't a
plain `x.y.z` marked as a prerelease.

The changelog goes in ahead of the tag rather than being amended into it
afterwards, because `main` takes no force pushes. Pushing to `main` at all
relies on the bypass repo admins have on the branch rules, so releasing is
maintainer-only; the commit it pushes touches nothing but `changelog/`.

`release` works for any package, including ones with no publish workflow yet.
To render a changelog and nothing else:

```shell
pdm run changelog build nobes-video 0.1.0
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
- Start its changelog: `pdm run changelog init nobes-{the new package shortname}`
  (then add it to the toctree in `docs/changelog.md`, as the command reminds you)
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