from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import xarray as xr


@dataclass
class ObservableStore(ABC):
    """Base class for observable object storage."""

    _warehouse: xr.DataArray = field(init=False)

    persistent: ClassVar[bool] = False
    store_dir: str | Path | None = None
    peek_size: int | None = None

    @property
    def warehouse(self) -> xr.DataArray:
        return self._warehouse

    @warehouse.setter
    def warehouse(self, value: xr.DataArray) -> None:
        self._warehouse = value

    @abstractmethod
    def update(self, data: xr.DataArray) -> None:
        pass

    @property
    def store_path(self) -> Path:
        return self.store_dir / self.__class__.__name__.lower()
