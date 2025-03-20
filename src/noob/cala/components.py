from collections.abc import Iterable
from enum import Enum
from typing import Self


class Component(Enum):
    """Enumeration of possible component types in the imaging data.

    Attributes:
        NEURON: Represents neuronal components.
        BACKGROUND: Represents background components (non-neuronal signals).
    """

    NEURON = "neuron"
    BACKGROUND = "background"


class ComponentTypes(list):
    """A specialized list class for managing collections of Component types.

    This class ensures type safety by only allowing Component enum values
    to be added to the list.

    Args:
        iterable: Optional iterable of Component values to initialize the list.

    Raises:
        ValueError: If attempting to add non-Component items to the list.
    """

    def __init__(self, iterable: Iterable[Component] = None) -> None:
        """Initialize a new ComponentTypes list.

        Args:
            iterable: Optional iterable of Component values.
        """
        super().__init__()
        if iterable:
            self.extend(iterable)

    def _check_element(self, item: Component) -> Component:
        """Validate that an item is a Component enum value.

        Args:
            item: The item to validate.

        Returns:
            Component: The validated item.

        Raises:
            ValueError: If the item is not a Component enum value.
        """
        if not isinstance(item, Component):
            raise ValueError("Item must be an Component")
        return item

    def append(self, item: Component) -> None:
        """Append a Component to the list.

        Args:
            item: The Component to append.

        Raises:
            ValueError: If item is not a Component enum value.
        """
        super().append(self._check_element(item))

    def insert(self, index: int | slice, item: Component) -> None:
        """Insert a Component at a specific index.

        Args:
            index: The index at which to insert the item.
            item: The Component to insert.

        Raises:
            ValueError: If item is not a Component enum value.
        """
        super().insert(index, self._check_element(item))

    def extend(self, iterable: Iterable[Component]) -> None:
        """Extend the list with an iterable of Components.

        Args:
            iterable: An iterable of Component values.

        Raises:
            ValueError: If any item in the iterable is not a Component enum value.
        """
        for item in iterable:
            self.append(item)

    def __add__(self, other: Iterable[Component]) -> Self:
        """Create a new ComponentTypes list by concatenating with another iterable.

        Args:
            other: Another iterable of Components.

        Returns:
            ComponentTypes: A new ComponentTypes instance containing all elements.
        """
        result = ComponentTypes(self)
        result.extend(other)
        return result

    def __iadd__(self, other: Iterable[Component]) -> Self:
        """Implement in-place addition with another iterable.

        Args:
            other: Another iterable of Components.

        Returns:
            ComponentTypes: The modified ComponentTypes instance.
        """
        self.extend(other)
        return self

    def __setitem__(self, index: int | slice, item: Component | Iterable[Component]) -> None:
        """Set an item or slice of items in the list.

        Args:
            index: The index or slice to set.
            item: The Component or iterable of Components to set.

        Raises:
            ValueError: If any item is not a Component enum value.
        """
        if isinstance(index, slice):
            items = []
            for i in item:
                items.append(self._check_element(i))
            super().__setitem__(index, items)
        else:
            super().__setitem__(index, self._check_element(item))
