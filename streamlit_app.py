"""Streamlit chat UI for the US Census Agent.

Run locally:    streamlit run streamlit_app.py
Deployed:       Streamlit Community Cloud reads .streamlit/secrets.toml
"""
from __future__ import annotations

import logging
import time
from typing import Any

import streamlit as st

from app.agent import CensusAgent
from app.config import Settings, load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="US Census Agent",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Connecting to Snowflake…")
def get_agent() -> CensusAgent:
    settings: Settings = load_settings()
    return CensusAgent(settings)


def render_trace(trace_events: list[dict[str, Any]]) -> None:
    if not trace_events:
        return
    for e in trace_events:
        kind = e["kind"]
        if kind == "guardrail_verdict":
            st.markdown(f"🛡️ **guardrail** → `{e['verdict']}` — {e.get('reason','')}")
        elif kind == "tool_call":
            with st.expander(f"🛠️ tool_call · `{e['name']}`", expanded=False):
                st.json(e.get("input"))
        elif kind == "tool_result":
            tag = "❌ error" if e.get("is_error") else "✅"
            st.markdown(f"⬅️ **tool_result** {tag} ({e['chars']} chars)")
        elif kind == "model_call":
            st.markdown(f"🧠 model call (iter {e['iteration']})")
        elif kind == "final":
            st.markdown(f"📤 final answer ({e['chars']} chars)")
        elif kind == "model_error":
            st.error(f"model error: {e.get('error')}")
        elif kind == "max_iterations_hit":
            st.warning("hit max iterations")


# --- Session state ---
# `agent_history` is the raw Anthropic-format list threaded through `agent.respond`.
# `display_messages` is a parallel list of {role, text, meta} for rendering only —
# it holds just the user's question and the assistant's final answer per turn,
# never the intermediate tool_use/tool_result blocks. Keeping these separate
# means every rerun re-renders a clean transcript, while the agent still sees
# the full tool-use history it needs for context.

if "agent_history" not in st.session_state:
    st.session_state.agent_history: list[dict[str, Any]] = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages: list[dict[str, Any]] = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


# ---------------------------- Sidebar ----------------------------

with st.sidebar:
    st.markdown("### 🇺🇸 US Census Chat Agent")
    st.markdown(
        "Grounded on the **Cybersyn US Open Census** Snowflake share "
        "(2019 & 2020 ACS 5-year estimates, Census Block Group level)."
    )
    st.markdown("---")
    st.markdown("**Try asking:**")
    suggestions = [
        "What is the total population of California?",
        "Which 5 US counties have the highest median household income?",
        "How many households in Texas receive food stamps?",
        "Compare the median home value in Manhattan vs. Brooklyn.",
        "What share of households in San Francisco have no internet access?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{hash(s)}", use_container_width=True):
            st.session_state.pending_input = s
            st.rerun()

    st.markdown("---")
    show_trace = st.checkbox("Show agent trace", value=False)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.agent_history = []
        st.session_state.display_messages = []
        st.session_state.last_trace = None
        st.session_state.pending_input = None
        st.rerun()

# ---------------------------- Main ----------------------------

st.title("US Census Chat Agent")
st.caption(
    "Ask natural-language questions about US population, demographics, income, "
    "housing, education, and more. Answers are computed live against Snowflake. "
    "I remember the conversation — ask follow-ups like *“what about Texas?”* without repeating yourself."
)

try:
    agent = get_agent()
except Exception as e:
    st.error(
        "Could not connect to the underlying services. The site administrator "
        "should check the Snowflake / Anthropic credentials."
    )
    st.exception(e)
    st.stop()

# Render the existing transcript from display_messages (clean, no tool noise).
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        meta = msg.get("meta")
        if meta:
            st.caption(meta)

# Always render chat_input so it stays visible on suggestion-click reruns.
typed = st.chat_input("Ask a question about US Census data…")
pending = st.session_state.pending_input
st.session_state.pending_input = None
user_input = typed or pending

if user_input:
    # Echo the user turn immediately.
    st.session_state.display_messages.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        start = time.time()
        try:
            response = agent.respond(st.session_state.agent_history, user_input)
        except Exception as e:
            placeholder.error(
                "Something went wrong while answering your question. Please try again."
            )
            st.exception(e)
            # Roll back the user turn so they can retry cleanly.
            st.session_state.display_messages.pop()
            st.stop()
        elapsed = time.time() - start
        placeholder.markdown(response.text)
        meta = (
            f"⏱️ {elapsed:.1f}s · "
            f"{sum(1 for e in response.trace.events if e['kind']=='tool_call')} tool calls"
            + (" · 🛡️ refused" if response.refused else "")
        )
        st.caption(meta)
        st.session_state.display_messages.append(
            {"role": "assistant", "text": response.text, "meta": meta}
        )
        st.session_state.last_trace = response.trace.events

    # Rerun so the chat_input box clears and the new turn is rendered via the
    # normal display_messages loop (keeps the DOM consistent across turns).
    st.rerun()

if show_trace and st.session_state.last_trace:
    with st.expander("🔍 Last-turn agent trace", expanded=True):
        render_trace(st.session_state.last_trace)
