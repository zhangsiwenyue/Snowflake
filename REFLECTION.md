# Reflection

## Development process & key architectural decisions

I spent the first ~30 minutes on the data, not on code. The Cybersyn share is wide but
not self-describing — column names like `B19013e1` are opaque, and the only way to map
them to "median household income" is the `*METADATA_CBG_FIELD_DESCRIPTIONS` table, which
has 8,164 rows. That single observation drove the rest of the architecture:

1. **The LLM cannot be expected to memorize the schema.** Pre-stuffing the prompt with
   8k column descriptions would burn ~500k tokens per turn and still rot whenever the
   share is refreshed. Instead, schema discovery is a *tool* (`search_census_fields`),
   and the LLM uses it like a human analyst — search for the concept, get back the
   exact column codes, then write SQL.

2. **Curated table-family hints in the system prompt** (`TABLE_FAMILIES` in `app/census_catalog.py`)
   give the model enough prior knowledge to *skip* the schema search for common questions
   ("population", "median income"). This trades a small amount of prompt budget (~1k
   tokens) for one fewer round-trip on the most-common queries.

3. **Two-layer guardrails.** The input-side classifier (Haiku) is cheap, fast, and
   handles the easy cases (off-topic, unsafe, injection). The SQL-side validator
   (`sqlglot`) is the last line of defense — even if a future system-prompt change
   somehow allowed the agent to try a `DROP TABLE`, the validator would refuse before
   anything reaches Snowflake. Defense in depth, with the boundary that matters most
   (data integrity) protected by deterministic code, not an LLM.

4. **Streamlit + Streamlit Community Cloud** for UX/hosting. Native chat components
   meant zero front-end work. Session state cleanly handles conversation memory.
   GitHub-linked deploy means a public URL with no infrastructure to babysit. The
   tradeoff (cold starts on idle apps) is acceptable for a take-home; in a real
   product I'd switch to a warm always-on host.

5. **`claude-sonnet-4-6` for reasoning, `claude-haiku-4-5` for the classifier.** Pay
   for the smarter model only when reasoning is needed. The classifier task is easy
   enough that Haiku is indistinguishable from Sonnet for the cost of being 5× faster
   and ~10× cheaper.

6. **30s statement timeout + 8-iteration cap.** Combined, these put a hard ceiling on
   wall-clock under 60s with comfortable margin. The system prompt nudges the model
   toward narrowly-scoped queries, but the ceiling means a misbehaving query can't
   exhaust the budget.

I built bottom-up: SQL guard first (highest blast radius, easiest to test), then the
catalog helpers, then the agent loop, then the UI. Tests landed alongside each layer.
The first end-to-end run worked on the second try (the only fix was a sqlglot version
rename: `AlterTable` → `Alter`).

## What I'd improve with more time

**Higher leverage things I'd do next, roughly in priority order:**

1. **Streaming output to the UI.** Currently the user sees "Thinking…" until the full
   response is ready. Streaming the final assistant text via `st.write_stream` would
   feel much faster, and a cumulative tool-call indicator in the UI would let users
   see progress on multi-hop questions.
2. **A `compare_geographies` tool** that takes a list of geographies and a metric and
   returns a clean comparison. Today the agent does this with hand-written SQL. A
   typed tool would be more reliable and let me cache common comparisons.
3. **A small library of aggregation recipes** baked into the system prompt: how to
   compute population-weighted state-level medians, how to roll CBGs up to ZIP codes
   via the geometry table, etc. Most "wrong-looking" answers in stress-testing came
   from the model improvising its own aggregation.
4. **A correctness eval set** of ~30 (question, expected_answer_range) pairs that I
   could run on every commit. This is what would let me iterate on the system prompt
   with confidence.
5. **Move guardrails into a single `Anthropic Messages.create` call** using a
   structured-output system prompt or the new `tool_choice: any` shape, rather than a
   separate Haiku call. Saves ~500ms.
6. **Caching.** `search_census_fields("median household income", 2020)` returns the
   same rows every time. A small in-memory LRU + a Streamlit `@st.cache_data` for the
   FIPS map would shave latency for common queries.
7. **Authentication.** Currently the demo URL is public. For a real customer, I'd add
   the Streamlit Cloud SSO option or move to a host with native auth.
8. **Observability.** Each turn produces a trace dict; I'd ship those to a structured
   log (e.g., Logfire / Honeycomb) so I could see real-world failure modes instead of
   guessing.

## Edge cases & failure modes I identified but didn't fully fix

- **Median fields cannot be summed.** The system prompt warns the model, but a
  determined user could still elicit a misleading "weighted average of medians".
  Real fix: a typed tool that knows the metric type and returns "this is approximate"
  metadata.
- **Ambiguous place names.** "Springfield" matches dozens of cities. The system prompt
  tells the agent to ask which one or list matches; in practice it usually does, but
  there's no enforcement layer that would *prevent* a guess.
- **Sub-CBG geographies.** A neighborhood-level question (e.g. "the Mission in San
  Francisco") will get a county-level answer with a hedging note. The geometry table
  is there to support point-in-polygon joins, but I didn't build the tool.
- **Cross-year questions.** "How did Texas's population change between 2019 and 2020?"
  works, but is fragile — the model has to manually join two tables. A `time_series`
  helper tool would make this robust.
- **Tool-result truncation.** If a query genuinely returns >6kB of data, we truncate
  to fit context. The model is told it was truncated, but it may still be tempted to
  draw conclusions from a partial result. A better fix is a `summarize_query` follow-up
  tool that pushes aggregation back into Snowflake.
- **Snowflake credential leak surface.** Credentials live in `.streamlit/secrets.toml`
  (gitignored) and Streamlit Cloud's secret store. Rotating them after this evaluation
  ends is the right move.
- **Cold-start latency on Streamlit Community Cloud.** First request to an idle app
  can take 10–30s before the agent loop even begins. Within budget, but ugly. A warm
  host is the right answer for production.
- **Classifier fail-open behavior.** If the Haiku call errors, we let the message
  through. This is the safer UX choice for a candidate demo, but in a real product
  fail-closed (with a "service unavailable" message) is more defensible.

## Testing approach & what I'd add

**What's there now (32 unit tests + 3 opt-in e2e):**
- The SQL guardrail has the heaviest test coverage (19 tests). It's the layer where
  a regression would be most damaging (writes to data) and the easiest to fully cover
  with unit tests.
- The input classifier is tested with a stub Anthropic client — I verify the JSON
  parsing, fail-open behavior, and refusal-message coverage. Whether Haiku itself
  classifies correctly is a separate question I trust the model card for.
- The agent loop is tested with both the LLM and Snowflake stubbed. This catches
  orchestration bugs (message-history shape, tool dispatch, max-iterations) cheaply.
- The e2e smoke (`RUN_E2E=1`) hits real Snowflake and Anthropic and is the only thing
  that would catch schema drift in the share. I keep it minimal so it stays fast and
  cheap.

**What I'd add next:**
- **Correctness regression suite.** ~30 (question, expected_value_or_range) pairs
  exercised against the real services on each PR. Tolerates the inherent fuzziness of
  LLM phrasing by checking only that the right number appears in the response.
- **Adversarial-prompt suite.** A library of prompt-injection attempts and known-bad
  off-topic queries. Verifies the classifier *and* the system prompt's resistance to
  jailbreak attempts.
- **SQL fuzz-test.** Property-based tests with `hypothesis` that generate random
  syntactic variations of allowed and disallowed SQL and assert the validator
  classifies them correctly. Cheap insurance against parser quirks.
- **Latency budget test.** Assert that no single turn in the e2e set exceeds 30s, with
  a warning at 20s. Catches a regression before it bites a user.
- **Snowshake-down test.** Periodically (CI cron) walk every CBG table and verify the
  expected metadata columns are present. Catches schema drift in the Cybersyn share
  before users do.

## What I deliberately left out

- **A custom front-end.** Streamlit's chat components are good enough for a chat agent;
  rebuilding them in React would be vanity work that doesn't move any of the four
  evaluation dimensions.
- **A vector index over the field descriptions.** The metadata table is small enough
  (8k rows) that an `ILIKE` search returns in <500ms, and the model already knows the
  concept names well enough that lexical matching is sufficient. A vector index would
  be premature optimization here.
- **Multi-user session storage.** Streamlit's `session_state` gives per-browser-tab
  memory, which matches the assignment requirement. Persisting conversations across
  sessions is a product feature, not a take-home requirement.
- **Per-user rate limiting.** The Streamlit Cloud free tier handles light traffic; a
  serious deploy would put the agent behind an API gateway with per-key quotas.
