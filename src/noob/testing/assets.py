import sqlite3


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
