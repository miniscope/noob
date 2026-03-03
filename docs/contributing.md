# Contributing

```{warning}
Contributing docs are a work in progress and are not complete!
Maintainers reserve the right to create and enforce arbitrary rules for the moment.
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