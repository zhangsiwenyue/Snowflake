"""Verify multi-turn memory works end-to-end: ask a question, then a follow-up
that depends on context ('what about Texas?'), then another follow-up."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import CensusAgent
from app.config import load_settings


def main() -> None:
    agent = CensusAgent(load_settings())
    history: list = []
    turns = [
        "What is the total population of California in 2020?",
        "And what about Texas?",
        "Which of the two has more people?",
    ]
    for i, q in enumerate(turns, 1):
        print(f"\n--- TURN {i}: {q} ---")
        resp = agent.respond(history, q)
        print(f"[{resp.elapsed_seconds:.1f}s, "
              f"{sum(1 for e in resp.trace.events if e['kind']=='tool_call')} tool calls]")
        print(resp.text)
        print(f"(history length: {len(history)})")


if __name__ == "__main__":
    main()
