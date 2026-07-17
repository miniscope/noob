# noob-core (Python)

Python counterparts of the PyO3 structs,
due to limitations in the sphinx-rust compatibility and PyO3's docs generation, 
all the docstrings for `Epoch` are in {rust:module}`noob_core::epoch`

The Scheduler class is excluded to avoid multiple references,
three schedulers is confusing enough already.

See
- Python: {class}`noob.scheduler.Scheduler`
- Bridge (what would be here): {rust:struct}`noob_core::bridge::PyScheduler` 
- Rust: {rust:struct}`noob_core::scheduler::Scheduler`

```{eval-rst}
.. automodule:: noob_core
   :members:
   :undoc-members:
   :exclude-members: Scheduler
```