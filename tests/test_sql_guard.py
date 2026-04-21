"""SQL guardrail tests — these are the cheapest, highest-signal tests.

A bug here means the agent could mutate or expose data outside the dataset.
"""
from __future__ import annotations

import pytest

from app.sql_guard import validate_and_rewrite


@pytest.mark.parametrize(
    "sql",
    [
        # DDL
        "DROP TABLE foo",
        "CREATE TABLE foo AS SELECT 1",
        "ALTER TABLE foo DROP COLUMN x",
        "TRUNCATE TABLE foo",
        # DML
        "INSERT INTO foo VALUES (1)",
        "UPDATE foo SET x = 1",
        "DELETE FROM foo",
        "MERGE INTO foo USING bar ON foo.x=bar.x WHEN MATCHED THEN DELETE",
        # Multi-statement
        "SELECT 1; DROP TABLE foo",
        # Foreign DB
        'SELECT * FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER',
        # Pure garbage
        "this is not sql",
        "",
    ],
)
def test_unsafe_queries_are_rejected(sql: str) -> None:
    result = validate_and_rewrite(sql)
    assert not result.safe, f"Expected reject for: {sql!r}"
    assert result.reason


def test_select_is_allowed() -> None:
    sql = 'SELECT B01001e1 FROM "2020_CBG_B01" WHERE CENSUS_BLOCK_GROUP = \'010010201001\''
    result = validate_and_rewrite(sql)
    assert result.safe
    assert result.rewritten_sql is not None


def test_limit_is_injected_when_missing() -> None:
    sql = 'SELECT B01001e1 FROM "2020_CBG_B01"'
    result = validate_and_rewrite(sql, default_limit=42)
    assert result.safe
    assert "LIMIT 42" in (result.rewritten_sql or "").upper()


def test_existing_limit_is_preserved() -> None:
    sql = 'SELECT B01001e1 FROM "2020_CBG_B01" LIMIT 7'
    result = validate_and_rewrite(sql, default_limit=200)
    assert result.safe
    assert "LIMIT 7" in (result.rewritten_sql or "").upper()
    assert "LIMIT 200" not in (result.rewritten_sql or "").upper()


def test_qualified_census_table_is_allowed() -> None:
    sql = (
        'SELECT B01001e1 FROM '
        '"US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"."PUBLIC"."2020_CBG_B01" '
        'LIMIT 5'
    )
    result = validate_and_rewrite(sql)
    assert result.safe, result.reason


def test_qualified_other_db_is_rejected() -> None:
    sql = 'SELECT * FROM "OTHER_DB"."PUBLIC"."FOO" LIMIT 5'
    result = validate_and_rewrite(sql)
    assert not result.safe
    assert "OTHER_DB" in (result.reason or "")


def test_cte_select_is_allowed() -> None:
    sql = (
        'WITH t AS (SELECT B01001e1 AS pop FROM "2020_CBG_B01") '
        'SELECT SUM(pop) FROM t'
    )
    result = validate_and_rewrite(sql)
    assert result.safe, result.reason


def test_aggregate_with_group_by_allowed() -> None:
    sql = (
        'SELECT LEFT(CENSUS_BLOCK_GROUP, 2) AS state_fips, SUM(B01001e1) AS pop '
        'FROM "2020_CBG_B01" GROUP BY 1 ORDER BY 2 DESC'
    )
    result = validate_and_rewrite(sql, default_limit=100)
    assert result.safe
    assert "LIMIT 100" in (result.rewritten_sql or "").upper()
