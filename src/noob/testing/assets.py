import numpy as np
import uvicorn
import xarray as xr
from fastapi import FastAPI


def xarray_asset() -> xr.DataArray:
    return xr.DataArray(
        np.ones((3, 4, 5), dtype=float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(0, 3, 1), "y": np.arange(0, 4, 1), "z": np.arange(0, 5, 1)},
    )


def server_asset(host, port) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: str | None = None):
        return {"item_id": item_id, "q": q}

    @app.get("/heartbeat")
    def read_heartbeat(msg: str):
        return {"message": msg}

    uvicorn.run(app, host=host, port=port)

    return app
