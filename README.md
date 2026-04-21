# US Census Chat Agent

A production-quality, conversational chat agent that answers natural-language questions
about the US population, grounded in the **Cybersyn US Open Census** Snowflake share
(2019 & 2020 ACS 5-year estimates, Census Block Group level).

Built for the Snowflake Applied AI take-home assignment (April 2026).

---

## 🌐 Live demo

| | |
|---|---|
| **URL** | _https://census-agent-yue.streamlit.app_ <sub>(filled in after Streamlit Cloud deploy)</sub> |
| **Login** | Public — no authentication required |
| **Backing data** | Cybersyn US Open Census (Snowflake Marketplace) |
| **LLM** | `claude-sonnet-4-6` (planning + SQL) · `claude-haiku-4-5-20251001` (input guardrail) |

Try asking:
- *What is the total population of California in 2020?*
- *Which 5 US counties have the highest median household income?*
- *How many households in Texas receive food stamps?*
- *Compare the median home value in Manhattan vs. Brooklyn.*
- *What share of households in San Francisco have no internet access?*

The agent **preserves conversation context across turns**, so follow-ups like *"what about Texas?"* work.

---

## 🏗️ Architecture

```
┌────────────────────────┐
│  Streamlit web UI      │   chat history kept in st.session_state
│  (streamlit_app.py)    │
└───────────┬────────────┘
            │ user message + history
            ▼
┌────────────────────────┐
│ InputGuardrail         │   Haiku classifier:
│ (app/guardrails.py)    │   in_scope / out_of_scope / unsafe / prompt_injection
└───────────┬────────────┘
            │ allowed
            ▼
┌────────────────────────┐
│ CensusAgent loop       │   Claude Sonnet 4.6 with tool use,
│ (app/agent.py)         │   capped at 8 iterations
└─┬─────────┬─────────┬──┘
  │         │         │
  ▼         ▼         ▼
search_     get_      run_
census_     state_    census_
fields      fips      sql
  │         │         │
  └─────────┴─────────┘
            │ all tools route through
            ▼
┌────────────────────────┐
│ SQL guardrail          │   sqlglot validator:
│ (app/sql_guard.py)     │   SELECT-only, allowlist DB/schema,
│                        │   auto-LIMIT, single-statement
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ SnowflakeClient        │   30s STATEMENT_TIMEOUT,
│ (app/snowflake_client) │   keep-alive, lazy reconnect
└────────────────────────┘
```

### Why these choices

| Decision | Why | Tradeoff |
|---|---|---|
| **Live Snowflake queries (text-to-SQL) over pre-loading** | The Census share has ~70 wide tables and **8,164 column descriptions** — pre-aggregating everything would either be incomplete or take days. Live SQL keeps every CBG, every column reachable. | Each question costs 1–3 Snowflake queries + warm warehouse. Acceptable for a chat agent; not for a public scale-out service. |
| **Schema retrieval as a tool** instead of stuffing the schema in the prompt | 8,164 column descriptions × hierarchical labels = ~500k tokens. Doesn't fit. The LLM searches the metadata table on demand, exactly like a human analyst would. | Adds one tool round-trip for unfamiliar concepts. Worth it. |
| **Tool-use loop in Claude Sonnet 4.6** rather than a hand-coded planner | Sonnet's tool-use is reliable enough that explicit planning code becomes scaffolding that has to be maintained. The system prompt + curated `TABLE_FAMILIES` map gives it the prior knowledge it needs. | Less inspectability than a deterministic planner. The trace expander in the UI compensates. |
| **Two-layer guardrail** (input classifier + SQL validator) | Classifier catches off-topic / unsafe / injection at the front door. Validator is the last line of defense — even if a future prompt change loosens the agent, no DDL/DML/cross-DB query can ever reach Snowflake. | Classifier adds ~500ms. Acceptable inside the 60s budget. |
| **Haiku for the classifier**, Sonnet for the agent | The classification task is easy and frequent. Pay for Sonnet only when reasoning is needed. | None observed. |
| **Streamlit + Streamlit Community Cloud** | Native chat components (`st.chat_message`, `st.chat_input`); session state for memory; one-click deploy from GitHub; HTTPS public URL with no infra. | Cold starts on idle apps (~10–30s). For a take-home, fine; for production, switch to Render/Fly with a warm instance. |
| **`MAX_ITERATIONS = 8`** | Hard cap on the tool loop. Each iteration is bounded by the 30s Snowflake timeout, so worst-case is well under 60s with margin for the LLM calls. | Theoretically a complex question could need >8 hops. None observed in testing. |
| **`STATEMENT_TIMEOUT_IN_SECONDS = 30`** | Catches accidentally expensive queries (cross-joins, missing predicates) before they blow the latency budget. | A legitimate big query is killed too. The agent returns a clear error and can retry with a smaller scope. |

---

## 🚀 Local development

### Prerequisites
- Python 3.11+
- A Snowflake trial account with the **US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET** share installed from Snowflake Marketplace
- An Anthropic API key

### Setup
```bash
git clone https://github.com/<owner>/snowflake-census-agent.git
cd snowflake-census-agent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Fill in your Snowflake + Anthropic credentials

streamlit run streamlit_app.py
```

The same secrets work in `.env` (for scripts and tests):
```bash
cp .streamlit/secrets.toml.example .env  # then convert TOML → KEY=value lines
```

---

## 🧪 Testing

**Tests are organized by what they protect, not by file structure.**

```bash
# Fast unit tests (mocked LLM and Snowflake) — run in <2s
pytest

# End-to-end smoke against real services — run in <60s, costs ~$0.05
RUN_E2E=1 pytest tests/test_e2e_smoke.py -v
```

| File | What it protects | Why this scope |
|---|---|---|
| `tests/test_sql_guard.py` (19 tests) | The SQL allowlist: no DDL, DML, multi-stmt, cross-DB; LIMIT auto-injection works | A bug here is the difference between a chat app and a DROP TABLE. Highest leverage tests. |
| `tests/test_guardrails.py` (8 tests) | Input classifier policy mapping & JSON parsing with a stub LLM | Verifies the boundary, not the LLM's judgment (which we trust the model card for). |
| `tests/test_agent_loop.py` (5 tests) | Tool-call dispatch, message-history shape, iteration cap, refusal short-circuit | Catches orchestration bugs without paying for LLM calls every test run. |
| `tests/test_e2e_smoke.py` (3 tests, opt-in) | The wires actually connect: Snowflake, Anthropic, schema hasn't drifted | Caught one schema column-name typo (`FIELD_LEVELl_9`) during development. |

What I deliberately **didn't** build, and why:
- **No LLM-graded answer-quality eval.** Setting up a golden dataset of (question, expected answer) pairs is a multi-day project and the answers move with each Census refresh. Out of scope for 24h.
- **No load test.** Streamlit Community Cloud serves single-user-style sessions; load testing would just measure their infra.
- **No mutation-testing of the SQL guardrail.** Worth it for v2; the parametrized test list covers the obvious vectors.

---

## ⚡ Performance

Measured with `RUN_E2E=1 pytest tests/test_e2e_smoke.py -v`:

| Question | Latency |
|---|---|
| "Population of California in 2020" | ~17s (1 schema search + 1 SUM query) |
| Off-topic refusal | <1s (classifier short-circuits before agent runs) |
| Follow-up "and what about California?" | ~17s (carries context) |

All comfortably inside the 60s budget. Cold-start the first time the warehouse resumes adds ~3s.

---

## 🛡️ Guardrails — what they catch

1. **InputGuardrail** (app/guardrails.py): Haiku classifies every user turn into `in_scope` / `out_of_scope` / `unsafe` / `prompt_injection`. Refusals never reach the main agent or Snowflake. **Fails open** on classifier errors — we'd rather answer a borderline query than break the UX on an Anthropic blip.
2. **SQL guard** (app/sql_guard.py): every SQL the model writes is parsed with `sqlglot` and rejected unless it's a single SELECT touching only the Census database. LIMIT is auto-injected.
3. **Snowflake-side timeout**: 30s statement timeout caps any one query. Combined with the iteration cap, the worst-case wall-clock is bounded.
4. **System-prompt grounding**: the agent is told to *say it can't answer* rather than guess when the dataset is insufficient (e.g. non-US data, future projections, individual-level questions).

---

## 📁 Repository layout

```
.
├── streamlit_app.py            # UI entry point
├── app/
│   ├── config.py               # env / Streamlit secrets loader
│   ├── snowflake_client.py     # connection + statement-level timeout
│   ├── census_catalog.py       # field search + state FIPS lookup
│   ├── sql_guard.py            # SELECT-only sqlglot validator
│   ├── guardrails.py           # input topic/safety classifier
│   └── agent.py                # tool-use loop, system prompt, dispatch
├── tests/
│   ├── test_sql_guard.py       # 19 unit tests
│   ├── test_guardrails.py      # 8 unit tests, mocked LLM
│   ├── test_agent_loop.py      # 5 orchestration tests, mocked LLM + SF
│   └── test_e2e_smoke.py       # 3 real-service tests, opt-in via RUN_E2E=1
├── scripts/                    # one-off Snowflake exploration helpers
├── .streamlit/secrets.toml.example
├── requirements.txt
├── REFLECTION.md               # written reflection (per assignment)
└── README.md                   # you are here
```

---

## 🐛 Known limitations

See [REFLECTION.md](REFLECTION.md) for the full list. Highlights:

- **Median fields** can't be aggregated by SUM. The agent is told to disclose the
  approximation when it weights or averages CBG-level medians, but a sufficiently
  twisted question could still produce a misleading number. Real fix: add a
  small library of correct aggregation patterns to the system prompt.
- **Geographies smaller than a CBG** (neighborhoods, ZIP codes) aren't directly addressable.
  We could join through `2020_CBG_GEOMETRY_WKT` to do point-in-polygon, but that's a v2.
- **Time-series questions across years** rely on the model joining 2019 and 2020 tables
  manually. Works for simple cases; would benefit from a `compare_years` tool.
