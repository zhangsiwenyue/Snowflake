"""The Census chat agent: a Claude tool-use loop with three tools.

Flow:
  user_message
    -> InputGuardrail.classify (Haiku)         [refuse if out-of-scope/unsafe]
    -> Claude (Sonnet) with tools:
        - search_census_fields(keyword, year)
        - get_state_fips(state)
        - run_census_sql(sql)
    -> loop until Claude returns text-only, or hits MAX_ITERATIONS

Memory: the full message history (user, assistant, tool_use, tool_result) is
passed back to Claude on every turn. Streamlit session_state owns the history.

Latency budget: 60 s. Per-tool Snowflake timeout is 30 s; we cap iterations
at 8 so a runaway tool loop can't exceed the budget.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import anthropic

from .census_catalog import TABLE_FAMILIES, YEARS_AVAILABLE, CensusCatalog
from .config import Settings
from .guardrails import GuardrailDecision, InputGuardrail, REFUSAL_MESSAGES
from .snowflake_client import SnowflakeClient
from .sql_guard import validate_and_rewrite

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
MAX_TOOL_RESULT_CHARS = 6000  # truncate big result sets so we don't blow context

SYSTEM_PROMPT = f"""You are a careful data analyst answering questions about the US population, grounded exclusively in the Cybersyn US Open Census dataset on Snowflake.

DATA AVAILABLE
- Granularity: Census Block Group (CBG). The CENSUS_BLOCK_GROUP column is a 12-digit FIPS string: first 2 chars = State FIPS, next 3 = County FIPS, next 6 = Tract, last digit = Block Group.
- Years: {", ".join(str(y) for y in YEARS_AVAILABLE)} (American Community Survey 5-year estimates ending those years). When the user does not specify a year, default to {max(YEARS_AVAILABLE)}.
- All wide data tables live in database "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET", schema "PUBLIC". They are named like "<YEAR>_CBG_<FAMILY>", e.g. "2020_CBG_B19".
- Estimate columns end in "e<n>" (e.g. B19013e1). Margin-of-error columns end in "m<n>" — usually you want the estimate.
- Lookup tables: "<YEAR>_METADATA_CBG_FIELD_DESCRIPTIONS" (column → description), "<YEAR>_METADATA_CBG_FIPS_CODES" (state/county lookup), "<YEAR>_METADATA_CBG_GEOGRAPHIC_DATA" (CBG → lat/long, area).

ACS TABLE FAMILIES (each is one physical table per year):
{chr(10).join(f"  {k} — {v}" for k, v in TABLE_FAMILIES.items())}

YOUR TOOLS
1. search_census_fields(keyword, year): use whenever you need to translate a human concept ("median household income", "households with food stamps", "Spanish speakers") into the exact column code. Always do this before writing SQL unless you are certain about the column.
2. get_state_fips(state): get the 2-digit state FIPS code from a state name or abbreviation.
3. run_census_sql(sql): execute a single read-only SELECT against the Census database. Limit yourself to safe queries — the validator will reject DDL, DML, and queries against other databases.

HOW TO ANSWER
- Aggregate to the geography the user asks for. Population of a state = SUM(B01001e1) over CBGs whose CENSUS_BLOCK_GROUP starts with that state's 2-digit FIPS. Use LEFT(CENSUS_BLOCK_GROUP, 2) for state, LEFT(CENSUS_BLOCK_GROUP, 5) for county.
- Median fields (e.g. median household income) cannot be summed — they are already medians at CBG level. For state-level medians, take a population-weighted average of CBG medians as a rough estimate, and clearly disclose the approximation. Or report the median of the CBG-level medians and explain the caveat.
- For population: prefer B01003e1 (Total Population, single column) over summing the B01001 sex-by-age detail when both work.
- Always cite the year and the column you used. When you give a number, say which CBGs/counties/states it covers.
- If a question is ambiguous (e.g. "what is the population of Springfield?"), ask which one or list the top matches. Don't guess.
- If a question cannot be answered from this dataset (e.g. asks about non-US data, future projections, or individual-level information), say so plainly and suggest a related question that *can* be answered.
- Keep answers concise. Lead with the number, then a one-line explanation of what was computed."""


@dataclass
class AgentTrace:
    """Lightweight log of what the agent did, for debugging in the UI sidebar."""
    events: list[dict[str, Any]] = field(default_factory=list)

    def log(self, kind: str, **payload: Any) -> None:
        self.events.append({"t": time.time(), "kind": kind, **payload})


@dataclass
class AgentResponse:
    text: str
    refused: bool
    refusal_kind: str | None
    trace: AgentTrace
    elapsed_seconds: float


class CensusAgent:
    def __init__(
        self,
        settings: Settings,
        anthropic_client: anthropic.Anthropic | None = None,
        snowflake_client: SnowflakeClient | None = None,
    ):
        self.settings = settings
        self.anthropic = anthropic_client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )
        self.snowflake = snowflake_client or SnowflakeClient(settings)
        self.catalog = CensusCatalog(self.snowflake)
        self.guard = InputGuardrail(self.anthropic)
        self.tool_specs = self._build_tool_specs()

    def _build_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_census_fields",
                "description": (
                    "Search the Census field-description metadata to translate a human "
                    "concept (e.g. 'median household income') into one or more exact "
                    "column codes (e.g. 'B19013e1') and the physical table that holds "
                    "them (e.g. '2020_CBG_B19'). Returns up to 25 matches, ranked by "
                    "shortest column id first."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Whitespace-separated terms; ALL must match the field hierarchy (case-insensitive)."
                        },
                        "year": {
                            "type": "integer",
                            "enum": list(YEARS_AVAILABLE),
                            "description": "Year of ACS estimates to search.",
                        },
                    },
                    "required": ["keyword"],
                },
            },
            {
                "name": "get_state_fips",
                "description": (
                    "Resolve a US state name or two-letter abbreviation to its 2-digit "
                    "FIPS code. Use this before filtering CENSUS_BLOCK_GROUP by state."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "description": "State name (e.g. 'California') or USPS abbreviation (e.g. 'CA').",
                        }
                    },
                    "required": ["state"],
                },
            },
            {
                "name": "run_census_sql",
                "description": (
                    "Execute a single read-only SELECT against the Census database. "
                    "Returns up to 200 rows (the validator auto-injects a LIMIT if "
                    "you don't). Statement timeout is 30 seconds. Only tables in the "
                    "Census database are reachable."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "A single SELECT statement. Do not include a trailing semicolon.",
                        }
                    },
                    "required": ["sql"],
                },
            },
        ]

    # ------------------- Tool implementations -------------------

    def _tool_search_census_fields(self, args: dict[str, Any]) -> str:
        keyword = (args.get("keyword") or "").strip()
        year = int(args.get("year") or max(YEARS_AVAILABLE))
        if not keyword:
            return json.dumps({"error": "keyword is required"})
        if year not in YEARS_AVAILABLE:
            return json.dumps({"error": f"year must be one of {list(YEARS_AVAILABLE)}"})
        hits = self.catalog.search_fields(keyword, year=year, limit=25)
        if not hits:
            return json.dumps({
                "matches": [],
                "hint": "No matches. Try different keywords, or fewer/broader terms.",
            })
        return json.dumps({
            "matches": [
                {
                    "column": h.column_id,
                    "table_id": h.table_id,
                    "table_title": h.table_title,
                    "description": h.description,
                    "physical_table": h.physical_table,
                }
                for h in hits
            ]
        })

    def _tool_get_state_fips(self, args: dict[str, Any]) -> str:
        state = (args.get("state") or "").strip()
        if not state:
            return json.dumps({"error": "state is required"})
        try:
            mapping = self.catalog.state_fips_map()
        except Exception as e:
            return json.dumps({"error": f"could not load FIPS map: {e}"})

        # Direct abbreviation lookup.
        if state.upper() in mapping:
            return json.dumps({"state": state.upper(), "state_fips": mapping[state.upper()]})

        # Name → abbreviation. Build once, here, since we don't have it cached.
        name_to_abbr = {
            "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
            "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
            "district of columbia": "DC", "washington dc": "DC", "washington d.c.": "DC",
            "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
            "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
            "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
            "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
            "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
            "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
            "new york": "NY", "north carolina": "NC", "north dakota": "ND",
            "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
            "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
            "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
            "virginia": "VA", "washington": "WA", "west virginia": "WV",
            "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
        }
        abbr = name_to_abbr.get(state.lower())
        if abbr and abbr in mapping:
            return json.dumps({"state": abbr, "state_fips": mapping[abbr]})
        return json.dumps({
            "error": f"Could not resolve '{state}' to a state.",
            "hint": "Try the full state name or the 2-letter postal code.",
        })

    def _tool_run_census_sql(self, args: dict[str, Any]) -> str:
        raw_sql = (args.get("sql") or "").strip()
        if not raw_sql:
            return json.dumps({"error": "sql is required"})
        guard = validate_and_rewrite(raw_sql, default_limit=self.settings.query_row_limit)
        if not guard.safe:
            return json.dumps({"error": f"SQL rejected: {guard.reason}"})
        try:
            rows, cols = self.snowflake.execute(guard.rewritten_sql or raw_sql)
        except Exception as e:
            return json.dumps({"error": f"Snowflake error: {e}"})
        out = json.dumps({
            "columns": cols,
            "row_count": len(rows),
            "rows": rows,
            "executed_sql": guard.rewritten_sql,
        }, default=str)
        if len(out) > MAX_TOOL_RESULT_CHARS:
            # Truncate the rows but keep enough that the model can still summarise.
            kept = []
            running = 0
            for r in rows:
                rs = json.dumps(r, default=str)
                if running + len(rs) > MAX_TOOL_RESULT_CHARS - 500:
                    break
                kept.append(r)
                running += len(rs)
            out = json.dumps({
                "columns": cols,
                "row_count": len(rows),
                "rows": kept,
                "truncated": True,
                "note": f"Showing {len(kept)} of {len(rows)} rows; result was too large.",
                "executed_sql": guard.rewritten_sql,
            }, default=str)
        return out

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name == "search_census_fields":
            return self._tool_search_census_fields(args)
        if name == "get_state_fips":
            return self._tool_get_state_fips(args)
        if name == "run_census_sql":
            return self._tool_run_census_sql(args)
        return json.dumps({"error": f"unknown tool: {name}"})

    # ------------------- Main loop -------------------

    def respond(
        self,
        history: list[dict[str, Any]],
        user_message: str,
    ) -> AgentResponse:
        """Run one turn. `history` is the existing conversation in Anthropic
        message format (list of {"role": ..., "content": ...}); we append the
        new user message and any assistant/tool messages produced this turn.
        Returns the final assistant text and a trace.
        """
        start = time.time()
        trace = AgentTrace()
        trace.log("guardrail_check", message=user_message[:200])

        decision: GuardrailDecision = self.guard.classify(user_message)
        trace.log("guardrail_verdict", verdict=decision.verdict, reason=decision.reason)

        if not decision.allowed:
            text = REFUSAL_MESSAGES.get(
                decision.verdict,
                "I can only help with questions about the US Census dataset.",
            )
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": text})
            return AgentResponse(
                text=text,
                refused=True,
                refusal_kind=decision.verdict,
                trace=trace,
                elapsed_seconds=time.time() - start,
            )

        history.append({"role": "user", "content": user_message})
        final_text = ""

        for iteration in range(MAX_ITERATIONS):
            trace.log("model_call", iteration=iteration)
            try:
                resp = self.anthropic.messages.create(
                    model=self.settings.anthropic_model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    tools=self.tool_specs,
                    messages=history,
                )
            except anthropic.APIError as e:
                final_text = (
                    "I hit an error talking to the language model "
                    f"({type(e).__name__}). Please try again in a moment."
                )
                trace.log("model_error", error=str(e))
                history.append({"role": "assistant", "content": final_text})
                break

            assistant_blocks = resp.content
            history.append({"role": "assistant", "content": assistant_blocks})

            if resp.stop_reason != "tool_use":
                final_text = "".join(
                    b.text for b in assistant_blocks if b.type == "text"
                ).strip() or "(no answer produced)"
                trace.log("final", chars=len(final_text))
                break

            # Execute every tool_use block in this assistant turn, append all
            # tool_result blocks in a single user message (Anthropic requires
            # them to be siblings, not separate messages).
            tool_results: list[dict[str, Any]] = []
            for block in assistant_blocks:
                if block.type != "tool_use":
                    continue
                trace.log("tool_call", name=block.name, input=block.input)
                try:
                    result = self._dispatch(block.name, block.input or {})
                    is_error = False
                except Exception as e:
                    result = json.dumps({"error": f"tool crashed: {e}"})
                    is_error = True
                trace.log(
                    "tool_result",
                    name=block.name,
                    is_error=is_error,
                    chars=len(result),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                })

            history.append({"role": "user", "content": tool_results})
        else:
            final_text = (
                "I wasn't able to converge on an answer within my step budget. "
                "Could you rephrase or narrow the question?"
            )
            trace.log("max_iterations_hit")
            history.append({"role": "assistant", "content": final_text})

        return AgentResponse(
            text=final_text,
            refused=False,
            refusal_kind=None,
            trace=trace,
            elapsed_seconds=time.time() - start,
        )
