"""Inspect the metadata tables that describe the Census fields and FIPS codes."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import snowflake.connector

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DB = "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"


def run(cur, sql: str) -> list:
    cur.execute(sql)
    return cur.fetchall()


def main() -> None:
    conn = snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse="COMPUTE_WH",
        role="ACCOUNTADMIN",
    )
    cur = conn.cursor()
    cur.execute("ALTER WAREHOUSE COMPUTE_WH RESUME IF SUSPENDED")
    cur.execute(f'USE DATABASE "{DB}"')
    cur.execute('USE SCHEMA "PUBLIC"')

    for tbl in [
        "2020_METADATA_CBG_FIELD_DESCRIPTIONS",
        "2020_METADATA_CBG_FIPS_CODES",
        "2020_METADATA_CBG_GEOGRAPHIC_DATA",
        "2020_CBG_B01",
        "2020_CBG_B19",
        "2020_REDISTRICTING_METADATA_CBG_FIELD_DESCRIPTIONS",
    ]:
        print(f"\n=== DESCRIBE {tbl} ===")
        try:
            for row in run(cur, f'DESCRIBE TABLE "{tbl}"'):
                print(f"  {row[0]:40s}  {row[1]}")
        except Exception as e:
            print(f"  ERR: {e}")
        print(f"--- sample {tbl} ---")
        try:
            for row in run(cur, f'SELECT * FROM "{tbl}" LIMIT 3'):
                print(f"  {row}")
        except Exception as e:
            print(f"  ERR: {e}")

    print("\n=== row counts ===")
    for tbl in [
        "2020_METADATA_CBG_FIELD_DESCRIPTIONS",
        "2020_METADATA_CBG_FIPS_CODES",
        "2020_METADATA_CBG_GEOGRAPHIC_DATA",
        "2020_CBG_B01",
    ]:
        try:
            print(tbl, run(cur, f'SELECT COUNT(*) FROM "{tbl}"'))
        except Exception as e:
            print(tbl, "ERR", e)

    print("\n=== distinct table_id values in field_descriptions ===")
    try:
        for row in run(
            cur,
            'SELECT DISTINCT TABLE_ID, TABLE_TITLE FROM "2020_METADATA_CBG_FIELD_DESCRIPTIONS" ORDER BY 1 LIMIT 100',
        ):
            print(f"  {row[0]} | {row[1]}")
    except Exception as e:
        print("  ERR:", e)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
