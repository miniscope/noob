from enum import Enum


class Component(Enum):
    """Enumeration of possible component types in the imaging data.

    Attributes:
        NEURON: Represents neuronal components.
        BACKGROUND: Represents background components (non-neuronal signals).
    """

    NEURON = "neuron"
    BACKGROUND = "background"
