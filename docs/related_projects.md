# Related Projects

## Streaming processors

- [streamz](https://github.com/python-streamz/streamz)
  - Nice examples of what pipeline systems need, even if it's not exactly how we'd do them.
  - See, e.g. ["zip" vs "combine_latest"](https://streamz.readthedocs.io/en/latest/core.html#branching-and-joining)
- [RxPY](https://github.com/ReactiveX/RxPY)
- [Apache Kafka](https://kafka.apache.org/)
  - good discussion of [design](https://kafka.apache.org/documentation/#theproducer),
    even if a very different application - massive scale, simple one-step processing
    replicated over many machines.

## Batch-based

- [Dagster](https://docs.dagster.io/)
  - [code locations](https://docs.dagster.io/deployment/code-locations/) interesting
    handling of multiple conflicting dep trees/environments in same pipeline:
    "just don't specify the dependencies" (except in dagster+! where the cloud is the dependencies!)
    and specify the venv instead. each venv is independent and integrated over RPC
  - some analogies in division of labor: op -> node, graph -> tube, 
    job -> tube + runner (not sure), i/o manager -> store...
- [luigi](https://github.com/spotify/luigi)