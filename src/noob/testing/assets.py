from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import xarray as xr
from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient


def xarray_asset() -> xr.DataArray:
    return xr.DataArray(
        np.ones((3, 4, 5), dtype=float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(0, 3, 1), "y": np.arange(0, 4, 1), "z": np.arange(0, 5, 1)},
    )


def server_asset(host: str, port: int) -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
        yield

    router = APIRouter()

    @router.get("/")
    def read_root() -> dict:
        return {"Hello": "World"}

    @router.get("/items/{item_id}")
    def read_item(item_id: int, q: str | None = None) -> dict:
        return {"item_id": item_id, "q": q}

    app = FastAPI(lifespan=lifespan, debug=True)

    with TemporaryDirectory() as tmpdir:
        app.mount(path="/dist", app=StaticFiles(directory=tmpdir), name="dist")
    app.include_router(router)

    # figure out how to run this in the background
    # uvicorn.run(app, host=host, port=port)

    return TestClient(app)
