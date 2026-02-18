# Epochs: Map, Gather, and Cardinality Manipulation

(Basic issue is alignment, need to know which events go with which other events.
Say you have a freeform eventing system where one signal source yields 50 events per second,
and another thing that yields 1 event per second.
how do you align these things?)

## Map - Cardinality Expansion

(diagrams for each of these where y-axis is epoch and x axis is order, i think mermaid git diagrams can do this)

Mapping expands the number of events per unit execution time,
where one emitted event is split into several.

Subepoch rules:
- By default, events happen in an implicit "tube" epoch
- A node emits a finite set of events within a subepoch
- A node that induces subepochs becomes expired in the parent epoch
- Subepochs are created as the subgraph of nodes that are downstream from the creating node
- Subepoch graphs inherit the completion status of nodes that are implied in the parent graph:
  (diagram, if a node depends on pre-map node `a` and post map node `b`, and `a` is done, then `a` is marked done in subepoch)
- When a node is completed in the parent epoch, the corresponding nodes in the subepochs are marked as complete.

## Gather

(gather is inverse of map, describe how gathers can work across top-level epochs)

## Statefulness

(statefulness declares that a node is sensitive to the order of its inputs,
i.e., even if a node is called with the same arguments, if it is called in a different order,
the results will be different.
Describe the different assumptions made about statefulness for different node kinds,
and how that plays out when working with map and gather,
particularly map and the asyncio runner for now.
)