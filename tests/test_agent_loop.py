"""Agent loop integration tests with stubbed Anthropic + Snowflake clients.

Validates the orchestration logic (tool dispatch, message-history shape,
iteration cap) without hitting any external service.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.agent import MAX_ITERATIONS, CensusAgent
from app.config import Settings


# --- Stubs --------------------------------------------------------------

@dataclass
class _ToolUseBlock:
    type: str
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class _TextBlock:
    type: str
    text: str


@dataclass
class _StubMsgResp:
    content: list[Any]
    stop_reason: str


class _ScriptedAnthropic:
    """Replays a list of pre-canned responses, ignoring the actual prompt."""

    def __init__(self, responses: list[_StubMsgResp]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # so .messages.create works

    def create(self, **kwargs: Any) -> _StubMsgResp:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Scripted Anthropic ran out of responses")
        return self._responses.pop(0)


class _FakeSnowflake:
    def __init__(self, rows_by_sql_substring: dict[str, list[dict[str, Any]]]):
        self.rows_by_sql_substring = rows_by_sql_substring
        self.executed: list[str] = []

    def execute(self, sql: str, params: Any = None) -> tuple[list[dict[str, Any]], list[str]]:
        self.executed.append(sql)
        for substring, rows in self.rows_by_sql_substring.items():
            if substring in sql:
                cols = list(rows[0].keys()) if rows else []
                return rows, cols
        return [], []


# --- Fixtures -----------------------------------------------------------

def _make_settings() -> Settings:
    return Settings(
        snowflake_user="u",
        snowflake_password="p",
        snowflake_account="a",
        snowflake_warehouse="w",
        snowflake_role="r",
        snowflake_database="US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
        anthropic_api_key="k",
        anthropic_model="claude-sonnet-4-6",
        query_row_limit=200,
        query_timeout_seconds=30,
    )


def _agent(scripted: list[_StubMsgResp], snowflake: _FakeSnowflake) -> CensusAgent:
    a = CensusAgent.__new__(CensusAgent)
    a.settings = _make_settings()
    a.anthropic = _ScriptedAnthropic(scripted)  # type: ignore[assignment]
    a.snowflake = snowflake  # type: ignore[assignment]
    from app.census_catalog import CensusCatalog
    from app.guardrails import GuardrailDecision, InputGuardrail

    a.catalog = CensusCatalog(snowflake)  # type: ignore[arg-type]

    class _AlwaysAllowGuard:
        def classify(self, _msg: str) -> GuardrailDecision:
            return GuardrailDecision("in_scope", "test")

    a.guard = _AlwaysAllowGuard()  # type: ignore[assignment]
    a.tool_specs = a._build_tool_specs()
    return a


# --- Tests --------------------------------------------------------------

def test_simple_text_answer_returns_directly() -> None:
    """Model returns plain text on first call → agent should return it."""
    scripted = [
        _StubMsgResp(
            content=[_TextBlock("text", "California's population is ~39M.")],
            stop_reason="end_turn",
        )
    ]
    agent = _agent(scripted, _FakeSnowflake({}))
    history: list[dict[str, Any]] = []
    resp = agent.respond(history, "Population of California?")
    assert not resp.refused
    assert "39M" in resp.text
    # History contains user + assistant.
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_tool_use_then_text_answer() -> None:
    """Model asks for FIPS, then returns text. Agent dispatches and continues."""
    scripted = [
        _StubMsgResp(
            content=[
                _ToolUseBlock("tool_use", "tu1", "get_state_fips", {"state": "CA"})
            ],
            stop_reason="tool_use",
        ),
        _StubMsgResp(
            content=[_TextBlock("text", "California FIPS is 06.")],
            stop_reason="end_turn",
        ),
    ]
    sf = _FakeSnowflake({
        "FROM \"2020_METADATA_CBG_FIPS_CODES\"": [
            {"STATE": "CA", "STATE_FIPS": "06"},
            {"STATE": "TX", "STATE_FIPS": "48"},
        ],
    })
    agent = _agent(scripted, sf)
    history: list[dict[str, Any]] = []
    resp = agent.respond(history, "What FIPS code is California?")
    assert resp.text == "California FIPS is 06."
    # User msg, assistant (tool_use), user (tool_result), assistant (text) → 4 entries.
    assert len(history) == 4
    assert history[2]["role"] == "user"
    tool_results = history[2]["content"]
    assert tool_results[0]["type"] == "tool_result"
    payload = json.loads(tool_results[0]["content"])
    assert payload["state_fips"] == "06"


def test_unsafe_sql_returns_error_without_hitting_snowflake() -> None:
    scripted = [
        _StubMsgResp(
            content=[
                _ToolUseBlock("tool_use", "tu1", "run_census_sql",
                              {"sql": "DROP TABLE foo"})
            ],
            stop_reason="tool_use",
        ),
        _StubMsgResp(
            content=[_TextBlock("text", "I cannot run that.")],
            stop_reason="end_turn",
        ),
    ]
    sf = _FakeSnowflake({})
    agent = _agent(scripted, sf)
    history: list[dict[str, Any]] = []
    agent.respond(history, "Drop the table")
    # Snowflake was never executed.
    assert sf.executed == []
    payload = json.loads(history[2]["content"][0]["content"])
    assert "rejected" in payload["error"].lower()


def test_max_iterations_hit_returns_message() -> None:
    """If the model keeps asking for tool calls, we bail at MAX_ITERATIONS."""
    scripted = [
        _StubMsgResp(
            content=[_ToolUseBlock("tool_use", f"tu{i}", "get_state_fips", {"state": "CA"})],
            stop_reason="tool_use",
        )
        for i in range(MAX_ITERATIONS)
    ]
    sf = _FakeSnowflake({
        "FROM \"2020_METADATA_CBG_FIPS_CODES\"": [{"STATE": "CA", "STATE_FIPS": "06"}],
    })
    agent = _agent(scripted, sf)
    history: list[dict[str, Any]] = []
    resp = agent.respond(history, "loop forever?")
    assert "step budget" in resp.text.lower()


def test_guardrail_refusal_short_circuits() -> None:
    from app.agent import REFUSAL_MESSAGES
    from app.guardrails import GuardrailDecision

    scripted: list[_StubMsgResp] = []  # Should never be called.
    agent = _agent(scripted, _FakeSnowflake({}))

    class _RefusingGuard:
        def classify(self, _msg: str) -> GuardrailDecision:
            return GuardrailDecision("out_of_scope", "off topic")

    agent.guard = _RefusingGuard()  # type: ignore[assignment]
    history: list[dict[str, Any]] = []
    resp = agent.respond(history, "tell me a joke")
    assert resp.refused
    assert resp.text == REFUSAL_MESSAGES["out_of_scope"]
    # Conversation still recorded so follow-ups have context.
    assert len(history) == 2
