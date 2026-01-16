import sqlite3

# import xarray as xr


# def xarray_asset() -> xr.DataArray:
#     return xr.DataArray(
#         np.ones((3, 4, 5), dtype=float),
#         dims=["x", "y", "z"],
#         coords={"x": np.arange(0, 3, 1), "y": np.arange(0, 4, 1), "z": np.arange(0, 5, 1)},
#     )


def db_connection() -> sqlite3.Connection:
    """
    in-memory database connection
    """
    conn = sqlite3.connect(":memory:")

    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("INSERT INTO users(name)" "VALUES (?)", ["Hannah Montana"])
    conn.commit()
    return conn
