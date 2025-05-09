# YAML Tube Config

See also: examples in tests.

## Initial Example

Example of the yaml syntax for tube configuration

```{note}
None of the examples in thie files are final!
Don't use this as a reference for how the package works! we are drafting!
```


```yaml
noob_id: example-pipeline
noob_model: example.MyPipeline
noob_version: v0.6.0

nodes:
  file:
    # the string name of the node type, in the `name` classvar
    type: "file-source"
    
    # each node can specify a TypedDict of config values,
    # this dict sets them statically
    config:
      # other models can be referred to by `noob_id` rather than inlining
      # so this would refer to some other config file, which would be merged on load
      layout: "sd-layout"
    
    # nodes can specify that they should receive some of their config values
    # when they are called, in this case "path" should be passed as "sd_path"
    passed:
      path: sd_path
      
    # connections between nodes are specified on the source node
    # each item in the array is a connection between one of the values in the node's output
    # and an item in the next node's processing signature.
    # e.g. in this case we return {"header": ..., "buffer": ...},
    # and header is passed to the `header` kwarg of the `merge` node
    # similarly, `buffer` is passed to the `buffer` kwargs
    outputs:
      - source: header
        target: merge.header
      - source: buffer
        target: merge.buffer
        
  merge:
    type: "merge-buffers"
    # config values can also be spec'd as coming from other nodes *after* model instantiation
    # e.g. when config values are computed dynamically
    fill:
      width: file.width
      height: file.height
    outputs:
      - source: frame
        target: return
  
  # the "return" node is a special kind of node that collects outputs from a pipeline run
  # and returns them from a `process()` call
  return:
    config:
      key: frame
    type: "return"

```