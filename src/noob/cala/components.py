from enum import Enum


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

    def __init__(self, iterable=None):
        """Initialize a new ComponentTypes list.

        Args:
            iterable: Optional iterable of Component values.
        """
        super(ComponentTypes, self).__init__()
        if iterable:
            self.extend(iterable)

    def _check_element(self, item):
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

    def append(self, item):
        """Append a Component to the list.

        Args:
            item: The Component to append.

        Raises:
            ValueError: If item is not a Component enum value.
        """
        super(ComponentTypes, self).append(self._check_element(item))

    def insert(self, index, item):
        """Insert a Component at a specific index.

        Args:
            index: The index at which to insert the item.
            item: The Component to insert.

        Raises:
            ValueError: If item is not a Component enum value.
        """
        super(ComponentTypes, self).insert(index, self._check_element(item))

    def extend(self, iterable):
        """Extend the list with an iterable of Components.

        Args:
            iterable: An iterable of Component values.

        Raises:
            ValueError: If any item in the iterable is not a Component enum value.
        """
        for item in iterable:
            self.append(item)

    def __add__(self, other):
        """Create a new ComponentTypes list by concatenating with another iterable.

        Args:
            other: Another iterable of Components.

        Returns:
            ComponentTypes: A new ComponentTypes instance containing all elements.
        """
        result = ComponentTypes(self)
        result.extend(other)
        return result

    def __iadd__(self, other):
        """Implement in-place addition with another iterable.

        Args:
            other: Another iterable of Components.

        Returns:
            ComponentTypes: The modified ComponentTypes instance.
        """
        self.extend(other)
        return self

    def __setitem__(self, index, item):
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
            super(ComponentTypes, self).__setitem__(index, items)
        else:
            super(ComponentTypes, self).__setitem__(index, self._check_element(item))
