"""Explore Census DB: list schemas, tables, and sample a few rows/columns."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import snowflake.connector

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DB = "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"


def main() -> None:
    conn = snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        role=os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    )
    cur = conn.cursor()
    cur.execute(f'USE DATABASE "{DB}"')
    cur.execute("ALTER WAREHOUSE COMPUTE_WH RESUME IF SUSPENDED")
    cur.execute("USE WAREHOUSE COMPUTE_WH")

    print("=== schemas ===")
    cur.execute(f'SHOW SCHEMAS IN DATABASE "{DB}"')
    schemas = [row[1] for row in cur.fetchall()]
    for s in schemas:
        print(s)

    for s in schemas:
        if s in ("INFORMATION_SCHEMA",):
            continue
        print(f"\n=== tables/views in {s} ===")
        cur.execute(f'SHOW TABLES IN SCHEMA "{DB}"."{s}"')
        tables = cur.fetchall()
        for t in tables:
            print("  T", t[1], t[3] if len(t) > 3 else "")
        cur.execute(f'SHOW VIEWS IN SCHEMA "{DB}"."{s}"')
        views = cur.fetchall()
        for v in views:
            print("  V", v[1])

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
