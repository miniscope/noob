from dataclasses import dataclass
from typing import Type, Optional, get_origin, Annotated

import xarray as xr

from cala.streaming.core import ObservableStore


@dataclass
class Distributor:
    """Manages a collection of fluorescent components (neurons and background).

    This class serves as a central manager for different storages,
    including spatial footprints, temporal traces, and various statistics.
    """

    _: int = 0

    def get(self, type_: Type) -> Optional[ObservableStore]:
        """Retrieve a specific Observable instance based on its type.

        Args:
            type_ (Type): The type of Observable to retrieve (e.g., Footprints, Traces).

        Returns:
            Optional[ObservableStore]: The requested Observable instance if found, None otherwise.
        """
        store_type = self._get_store_type(type_)
        if store_type is None:
            return
        for attr_name, attr_type in self.__annotations__.items():
            if attr_type == store_type:
                return getattr(self, attr_name).warehouse

    def init(self, result: xr.DataArray, type_: Type) -> None:
        """Store a DataArray results in their appropriate Observable containers.

        This method automatically determines the correct storage location based on the
        type of the input DataArray.

        Args:
            result: A single xr.DataArray to be stored. Must correspond to a valid Observable type.
            type_: type of the result. If an observable, should be an Annotated type that links to Store class.
        """
        target_store_type = self._get_store_type(type_)
        if target_store_type is None:
            return

        store_name = target_store_type.__name__.lower()
        # Add to annotations
        self.__annotations__[store_name] = target_store_type
        # Create and set the store
        setattr(self, store_name, target_store_type(result))

    def update(self, result: xr.DataArray, type_: Type) -> None:
        """Update an appropriate Observable containers with a result DataArray.

        This method automatically determines the correct storage location based on the
        type of the input DataArray.

        Args:
            result: A single xr.DataArray to be stored. Must correspond to a valid Observable type.
            type_: type of the result. If an observable, should be an Annotated type that links to Store class.
        """
        target_store_type = self._get_store_type(type_)
        if target_store_type is None:
            return

        store_name = target_store_type.__name__.lower()

        # Update the store
        getattr(self, store_name).update(result)

    @staticmethod
    def _get_store_type(type_: Type) -> type | None:
        if get_origin(type_) is Annotated:
            if issubclass(type_.__metadata__[0], ObservableStore):
                return type_.__metadata__[0]
