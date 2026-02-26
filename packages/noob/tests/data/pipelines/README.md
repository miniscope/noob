# Test Pipelines

Tests pipelines are roughly grouped by subdirectory:

- `basic` - pipelines that test behavior that all runners should be able to do
- `asset` - tests specific to asset handling
- `input` - tests specific to input handling
- `special` - tests that might test errors, edge cases, or other behaviors that are likely to vary by runner.

Grouping is mostly for developer QoL, and are not strict categories.
E.g. there are "asset" tests in `special/` because they test invalid assets,
etc.