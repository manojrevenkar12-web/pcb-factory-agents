"""PostgreSQL access. The schema is applied idempotently on start."""

from __future__ import annotations

from importlib.resources import files

import psycopg
from psycopg.rows import dict_row


def connect(url: str, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(url, autocommit=autocommit, row_factory=dict_row)


def migrate(url: str) -> None:
    sql = files("pfa").joinpath("sql/schema.sql").read_text()
    with connect(url, autocommit=True) as conn:
        # Serialise concurrent starts: CREATE ... IF NOT EXISTS is not race-free on its own.
        conn.execute("SELECT pg_advisory_lock(726101)")
        try:
            conn.execute(sql)
        finally:
            conn.execute("SELECT pg_advisory_unlock(726101)")


def reset(url: str) -> None:
    """Drops and recreates every schema. Tests and demos only."""
    with connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS agents, ops, erp, mailprovider CASCADE")
    migrate(url)
