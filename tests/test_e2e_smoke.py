"""End-to-end smoke tests against real Snowflake + Anthropic.

These are slow and cost money. Skipped unless RUN_E2E=1 is set.

Goal: catch breakage that pure unit tests miss — schema drift in the Census
share, broken network creds, model regressions on basic questions. Not a
correctness eval — just "the wires are connected".
"""
from __future__ import annotations

import os

import pytest

from app.agent import CensusAgent
from app.config import load_settings

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1",
    reason="Set RUN_E2E=1 to run real-API smoke tests",
)


@pytest.fixture(scope="module")
def agent() -> CensusAgent:
    return CensusAgent(load_settings())


def test_total_population_query(agent: CensusAgent) -> None:
    history: list = []
    resp = agent.respond(history, "What is the total population of California in 2020?")
    assert not resp.refused
    # Should mention something close to 39M (CA pop ~39M).
    assert any(s in resp.text for s in ("39", "40")), resp.text
    assert resp.elapsed_seconds < 60


def test_off_topic_is_refused(agent: CensusAgent) -> None:
    resp = agent.respond([], "What's the recipe for chocolate chip cookies?")
    assert resp.refused
    assert resp.refusal_kind == "out_of_scope"


def test_follow_up_uses_context(agent: CensusAgent) -> None:
    history: list = []
    agent.respond(history, "What is the population of Texas in 2020?")
    resp = agent.respond(history, "And what about California?")
    assert not resp.refused
    # Follow-up should still be a population number, not "what do you mean?"
    assert any(c.isdigit() for c in resp.text)
