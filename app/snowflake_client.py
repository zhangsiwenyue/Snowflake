"""Thin wrapper around snowflake-connector-python.

A single connection is shared per process (Streamlit re-runs the script for
every user interaction; we cache via st.cache_resource in the UI layer).
The connector itself is thread-safe at the connection level for sequential
cursors, which is sufficient for a single Streamlit user session.
"""
from __future__ import annotations

import logging
from typing import Any

import snowflake.connector
from snowflake.connector import DictCursor
from snowflake.connector.errors import DatabaseError, ProgrammingError

from .config import Settings

# Use ? placeholders (qmark) instead of %s (pyformat). pyformat tries to
# str.format() the whole query *before* binding, which breaks any legitimate
# '%' character in a literal or ILIKE pattern (e.g. '%california%').
snowflake.connector.paramstyle = "qmark"

logger = logging.getLogger(__name__)


class SnowflakeClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._conn: snowflake.connector.SnowflakeConnection | None = None

    def _connect(self) -> snowflake.connector.SnowflakeConnection:
        if self._conn is None or self._conn.is_closed():
            logger.info("Opening Snowflake connection")
            self._conn = snowflake.connector.connect(
                user=self.settings.snowflake_user,
                password=self.settings.snowflake_password,
                account=self.settings.snowflake_account,
                warehouse=self.settings.snowflake_warehouse,
                role=self.settings.snowflake_role,
                database=self.settings.snowflake_database,
                schema="PUBLIC",
                client_session_keep_alive=True,
                login_timeout=15,
                network_timeout=30,
            )
            cur = self._conn.cursor()
            try:
                cur.execute(
                    f"ALTER WAREHOUSE {self.settings.snowflake_warehouse} RESUME IF SUSPENDED"
                )
            except (DatabaseError, ProgrammingError) as e:
                # Already running, or no permission — fine either way.
                logger.debug("Warehouse resume note: %s", e)
            finally:
                cur.close()
        return self._conn

    def execute(
        self, sql: str, params: dict | list | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Run a SQL statement and return (rows, column_names).

        Applies a statement-level timeout so a runaway query cannot exceed our
        60-second SLA. Raises the underlying Snowflake exception on error;
        callers (the agent tool) translate those into model-friendly messages.
        """
        conn = self._connect()
        cur = conn.cursor(DictCursor)
        try:
            cur.execute(
                f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {self.settings.query_timeout_seconds}"
            )
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            return rows, cols
        finally:
            cur.close()

    def close(self) -> None:
        if self._conn is not None and not self._conn.is_closed():
            self._conn.close()
            self._conn = None
