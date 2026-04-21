"""Configuration loaded from env vars or Streamlit secrets.

We read from os.environ first, then fall back to st.secrets so the same code
runs locally (with .env) and on Streamlit Community Cloud (with secrets.toml).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def _get(key: str, default: str | None = None) -> str | None:
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st

        return st.secrets.get(key, default)  # type: ignore[no-any-return]
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    snowflake_user: str
    snowflake_password: str
    snowflake_account: str
    snowflake_warehouse: str
    snowflake_role: str
    snowflake_database: str
    anthropic_api_key: str
    anthropic_model: str
    query_row_limit: int
    query_timeout_seconds: int


def load_settings() -> Settings:
    missing = [
        k
        for k in (
            "SNOWFLAKE_USER",
            "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_ACCOUNT",
            "ANTHROPIC_API_KEY",
        )
        if not _get(k)
    ]
    if missing:
        raise RuntimeError(
            f"Missing required configuration: {', '.join(missing)}. "
            "Set them in .env (local) or .streamlit/secrets.toml (deployed)."
        )
    return Settings(
        snowflake_user=_get("SNOWFLAKE_USER") or "",
        snowflake_password=_get("SNOWFLAKE_PASSWORD") or "",
        snowflake_account=_get("SNOWFLAKE_ACCOUNT") or "",
        snowflake_warehouse=_get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH") or "COMPUTE_WH",
        snowflake_role=_get("SNOWFLAKE_ROLE", "ACCOUNTADMIN") or "ACCOUNTADMIN",
        snowflake_database=_get(
            "SNOWFLAKE_DATABASE",
            "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
        )
        or "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
        anthropic_api_key=_get("ANTHROPIC_API_KEY") or "",
        anthropic_model=_get("ANTHROPIC_MODEL", "claude-sonnet-4-6") or "claude-sonnet-4-6",
        query_row_limit=int(_get("QUERY_ROW_LIMIT", "200") or "200"),
        query_timeout_seconds=int(_get("QUERY_TIMEOUT_SECONDS", "30") or "30"),
    )
