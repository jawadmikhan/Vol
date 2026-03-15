"""
TimescaleDB connection manager.

Provides a connection pool and helper methods for reading/writing
time-series data used by the volatility portfolio.
"""

import os
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import psycopg2
from psycopg2 import pool, extras

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool (singleton)
# ---------------------------------------------------------------------------

_pool: pool.ThreadedConnectionPool | None = None


def get_pool() -> pool.ThreadedConnectionPool:
    """Return the global connection pool, creating it on first call."""
    global _pool
    if _pool is None or _pool.closed:
        _pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=os.getenv("TSDB_HOST", "localhost"),
            port=int(os.getenv("TSDB_PORT", "5432")),
            dbname=os.getenv("TSDB_NAME", "vol_db"),
            user=os.getenv("TSDB_USER", "vol_user"),
            password=os.getenv("TSDB_PASSWORD", "vol_pass"),
        )
        logger.info("TimescaleDB connection pool created")
    return _pool


@contextmanager
def get_connection():
    """Yield a connection from the pool; return it on exit."""
    p = get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def insert_dataframe(table: str, df: pd.DataFrame) -> int:
    """
    Bulk-insert a DataFrame into *table* using COPY for speed.
    Column names in *df* must match the table columns exactly.
    Returns the number of rows inserted.
    """
    if df.empty:
        return 0

    cols = list(df.columns)
    col_list = ", ".join(cols)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Use execute_values for fast batch insert
            template = f"({', '.join(['%s'] * len(cols))})"
            query = f"INSERT INTO {table} ({col_list}) VALUES %s"
            extras.execute_values(
                cur, query, df.values.tolist(), template=template, page_size=1000
            )
            count = cur.rowcount
            logger.debug("Inserted %d rows into %s", count, table)
            return count


def upsert_row(table: str, data: dict, conflict_cols: list[str]) -> None:
    """Insert a single row, updating on conflict."""
    cols = list(data.keys())
    vals = list(data.values())
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    conflict = ", ".join(conflict_cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in conflict_cols)

    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, vals)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def query_df(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run a SELECT and return results as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def latest_regime() -> dict | None:
    """Return the most recent regime signal row as a dict."""
    df = query_df(
        "SELECT * FROM vol_regime_signals ORDER BY ts DESC LIMIT 1"
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def latest_iv_surface() -> pd.DataFrame:
    """Return the most recent implied vol surface snapshot."""
    return query_df("""
        SELECT * FROM implied_vol_surface
        WHERE ts = (SELECT MAX(ts) FROM implied_vol_surface)
        ORDER BY name, tenor_months, strike_delta
    """)


def latest_realized_vol() -> pd.DataFrame:
    """Return the most recent realized vol history (last 252 trading days)."""
    return query_df("""
        SELECT * FROM realized_vol_history
        WHERE ts >= (SELECT MAX(ts) FROM realized_vol_history) - INTERVAL '252 days'
        ORDER BY ts, name
    """)


def latest_correlation_matrix() -> pd.DataFrame:
    """Return the most recent correlation snapshot as a pivot table."""
    df = query_df("""
        SELECT asset_1, asset_2, correlation
        FROM correlation_snapshots
        WHERE ts = (SELECT MAX(ts) FROM correlation_snapshots)
    """)
    if df.empty:
        return df
    return df.pivot(index="asset_1", columns="asset_2", values="correlation")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def close_pool():
    """Shut down the connection pool."""
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        logger.info("TimescaleDB connection pool closed")
        _pool = None
