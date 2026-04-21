"""One-off probe: verify Snowflake connection and list Cybersyn Census shares."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import snowflake.connector

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    conn = snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        role=os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        client_session_keep_alive=True,
    )
    cur = conn.cursor()
    print("=== version ===")
    cur.execute("SELECT CURRENT_VERSION(), CURRENT_ACCOUNT(), CURRENT_ROLE()")
    print(cur.fetchone())

    print("\n=== databases ===")
    cur.execute("SHOW DATABASES")
    for row in cur.fetchall():
        print(row[1], "-", row[5] if len(row) > 5 else "")

    print("\n=== warehouses ===")
    cur.execute("SHOW WAREHOUSES")
    for row in cur.fetchall():
        print(row[0], row[1] if len(row) > 1 else "")

    cur.close()
    conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
