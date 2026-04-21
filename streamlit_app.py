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


def render_history(messages: list[dict[str, Any]]) -> None:
    """Render only the user-visible turns. Tool_use / tool_result blocks
    that live inside `messages` are shown in the trace expander, not the
    main chat stream."""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            with st.chat_message(role):
                st.markdown(content)
        elif isinstance(content, list):
            # Anthropic-style content blocks. Show only text blocks here.
            text_parts = [
                getattr(b, "text", b.get("text", "")) if not isinstance(b, str) else b
                for b in content
                if (isinstance(b, str)
                    or getattr(b, "type", b.get("type") if isinstance(b, dict) else None) == "text")
            ]
            text = "\n\n".join(t for t in text_parts if t)
            if text and role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(text)


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
            st.session_state["pending_input"] = s
            st.rerun()

    st.markdown("---")
    show_trace = st.checkbox("Show agent trace", value=False)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_trace = None
        st.rerun()

# ---------------------------- Main ----------------------------

st.title("US Census Chat Agent")
st.caption(
    "Ask natural-language questions about US population, demographics, income, "
    "housing, education, and more. Answers are computed live against Snowflake."
)

if "history" not in st.session_state:
    st.session_state.history: list[dict[str, Any]] = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

try:
    agent = get_agent()
except Exception as e:
    st.error(
        "Could not connect to the underlying services. The site administrator "
        "should check the Snowflake / Anthropic credentials."
    )
    st.exception(e)
    st.stop()

# Render existing history.
render_history(st.session_state.history)

# Handle suggestion clicks.
pending = st.session_state.pop("pending_input", None)
user_input = pending or st.chat_input("Ask a question about US Census data…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        start = time.time()
        try:
            response = agent.respond(st.session_state.history, user_input)
        except Exception as e:
            placeholder.error(
                "Something went wrong while answering your question. Please try again."
            )
            st.exception(e)
            st.stop()
        elapsed = time.time() - start
        placeholder.markdown(response.text)
        st.caption(
            f"⏱️ {elapsed:.1f}s · {len([e for e in response.trace.events if e['kind']=='tool_call'])} tool calls"
            + (" · 🛡️ refused" if response.refused else "")
        )
        st.session_state.last_trace = response.trace.events

if show_trace and st.session_state.last_trace:
    with st.expander("🔍 Last-turn agent trace", expanded=True):
        render_trace(st.session_state.last_trace)
