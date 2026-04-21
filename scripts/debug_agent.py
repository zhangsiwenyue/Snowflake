"""Run the agent end-to-end from the CLI with verbose tracing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import CensusAgent
from app.config import load_settings


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the total Asian population in California?"
    agent = CensusAgent(load_settings())
    history: list = []
    resp = agent.respond(history, question)
    print(f"\n\n====== FINAL ({resp.elapsed_seconds:.1f}s) ======")
    print(resp.text)
    print("\n====== TRACE ======")
    for e in resp.trace.events:
        kind = e["kind"]
        if kind == "tool_call":
            print(f"  → {e['name']}({json.dumps(e.get('input'))})")
        elif kind == "tool_result":
            print(f"    ⬅ {'ERR' if e.get('is_error') else 'OK'} {e['chars']}ch")
        elif kind == "model_call":
            print(f"  [iter {e['iteration']}]")
        elif kind == "final":
            print(f"  ✅ final ({e['chars']}ch)")
        elif kind == "max_iterations_hit":
            print("  ❌ MAX_ITERATIONS")
        else:
            print(f"  · {kind}")

    print("\n====== FULL ASSISTANT MESSAGES ======")
    for i, m in enumerate(history):
        if m["role"] != "assistant":
            continue
        content = m["content"]
        if isinstance(content, str):
            print(f"[msg {i} text] {content[:400]}")
        else:
            for b in content:
                if getattr(b, "type", None) == "text":
                    print(f"[msg {i} text] {b.text[:400]}")
                elif getattr(b, "type", None) == "tool_use":
                    print(f"[msg {i} tool_use] {b.name}({json.dumps(b.input)[:200]})")


if __name__ == "__main__":
    main()
