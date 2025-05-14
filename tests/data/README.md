# Test data

## Pipelines

- Basic: one input, one output, with return value from pipeline
- Branching: one input with an output mapped to two nodes
- Merging: two nodes converge onto one
- Cardinality: 
  - One output that takes multiple runs of an input
  - One input that emits multiple outputs that need to be mapped/flattened
- "config-like": node that runs once and emits config-like values
  - "pass the same value every time"
  - or, nodes can emit into a special "context" node that other nodes can request

## Stores

- Basic: sharing values between nodes
- Map output value to store, access from key
- Accessing store:
  - functional: passed in some context parameter
  - class: accessed from store attribute
