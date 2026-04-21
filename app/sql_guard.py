"""SQL guardrail: parse with sqlglot, allow only read-only SELECTs.

We use sqlglot rather than a regex because regexes drown in edge cases
(comments, CTEs, sub-selects). The validator:
  * Requires exactly one top-level statement.
  * Rejects anything that is not a SELECT (or a CTE that ultimately SELECTs).
  * Rejects DDL/DML keywords anywhere in the AST, even if disguised.
  * Restricts table references to the configured Census database/schema.
  * Auto-injects LIMIT if absent (so the agent cannot accidentally fetch all
    242k CBG rows and blow the latency budget).
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

# Keywords whose appearance anywhere is unsafe regardless of position.
FORBIDDEN_NODES: tuple[type, ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.AlterColumn,
    exp.TruncateTable,
    exp.Merge,
    exp.Command,  # generic catch-all for unparsed ddl-like statements
)

ALLOWED_DB = "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"
ALLOWED_SCHEMAS = {"PUBLIC", "INFORMATION_SCHEMA"}


@dataclass(frozen=True)
class GuardResult:
    safe: bool
    reason: str | None
    rewritten_sql: str | None


def validate_and_rewrite(sql: str, default_limit: int = 200) -> GuardResult:
    sql = sql.strip().rstrip(";")
    if not sql:
        return GuardResult(False, "Empty SQL.", None)

    try:
        parsed = sqlglot.parse(sql, read="snowflake")
    except Exception as e:
        return GuardResult(False, f"Could not parse SQL: {e}", None)

    if len(parsed) != 1 or parsed[0] is None:
        return GuardResult(False, "Exactly one statement is allowed.", None)

    tree = parsed[0]

    # Reject anything that contains forbidden node types anywhere in the tree.
    for node in tree.walk():
        node_obj = node[0] if isinstance(node, tuple) else node
        if isinstance(node_obj, FORBIDDEN_NODES):
            return GuardResult(
                False,
                f"Statement type not allowed: {type(node_obj).__name__}.",
                None,
            )

    # Top-level must be a SELECT (CTEs wrap a Select via With).
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
        # `WITH ... SELECT` is parsed as Select with a `with` arg; check that too.
        if not (isinstance(tree, exp.Query)):
            return GuardResult(False, "Only SELECT statements are allowed.", None)

    # Restrict table references to our database/schema (or unqualified, which
    # we resolve at run time via USE DATABASE/USE SCHEMA on the connection).
    # Snowflake parses "DB.SCHEMA.TABLE" with catalog=DB, db=SCHEMA.
    for table in tree.find_all(exp.Table):
        catalog = table.args.get("catalog")
        schema = table.args.get("db")
        if catalog is not None and catalog.name.upper() != ALLOWED_DB:
            return GuardResult(
                False,
                f"Table references outside the Census database are not allowed: {catalog.name}",
                None,
            )
        if schema is not None and schema.name.upper() not in ALLOWED_SCHEMAS:
            return GuardResult(
                False,
                f"Schema not allowed: {schema.name}",
                None,
            )

    # Inject a LIMIT if the outermost SELECT lacks one, so we never stream
    # hundreds of thousands of rows back to the agent.
    if isinstance(tree, exp.Select) and tree.args.get("limit") is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(default_limit)))

    rewritten = tree.sql(dialect="snowflake")
    return GuardResult(True, None, rewritten)
