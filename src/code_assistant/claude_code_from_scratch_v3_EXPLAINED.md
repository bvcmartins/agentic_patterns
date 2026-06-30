# `claude_code_from_scratch_v3` — Code Walkthrough & Design Rationale

A detailed, **function-by-function** reference for `claude_code_from_scratch_v3.ipynb`,
with the *why* behind each implementation choice. Every public class, function, and node in
the notebook has its own entry below: real signature, what it does mechanically, and the
design rationale.

v3 is **the same coding agent as v2, rebuilt on LangGraph**. `v1` hand-rolls a single
tool loop; `v2` adds the article's reliability stack (test-time compute, planning, a
bounded context window, a five-subagent architecture) — all on a hand-written
`master_loop` over Ollama's raw `/api/chat`. v3 keeps the *same capabilities* and the
*same phase numbering*, but every piece of hand-rolled machinery is re-expressed as a
**LangGraph** idiom. Reading v2 and v3 side-by-side is the point: each section below says
what the framework replaced and what that buys.

Two framing commitments carry over from v2 and shape everything:

- **One model-construction chokepoint.** Every chat model is built by the `llm(role, …)`
  factory. Swap that one function + the `MODELS` map and the whole notebook retargets to
  any OpenAI-compatible backend. **The graphs never change** — they operate on
  `ChatOllama` through LangChain's `Runnable` interface, not on a backend.
- **Nothing is claimed without a runnable artifact.** Code is gated by a linter, then by
  real test execution, then by an *independent* re-verification. The agent never
  "believes" its own output — `spec_verify` is run again, by us, after the team says done.

The single sentence that captures the whole notebook: **in v3, the loop *is* a graph, the
branches *are* edges, structured output *is* a Pydantic schema, and parallelism *is*
`.batch()`.** Everything else is the v2 design, preserved.

---

## How to read this document

Each phase below is split into its constituent functions/classes. An entry looks like:

> ### `function_name(args) -> ReturnType` *(cell N)*
> **Does:** the mechanics — what it reads, computes, and returns.
> **Why:** the design choice behind it.

The **cell N** tags match the companion graph (below) and the notebook's own cell order, so
you can jump straight from a function name to its source.

---

## Companion: the interactive code-block graph

This document has a sibling: **[`claude_code_from_scratch_v3_GRAPH.html`](claude_code_from_scratch_v3_GRAPH.html)**
— open it in any browser (no network needed). It renders the notebook as a graph where **every
node is one code cell, showing that cell's real code**, and **every edge is a real dependency**:
an arrow `A → B` means cell `B` *uses* a symbol that cell `A` *defines* (the edge is labelled
with the symbol names). The edges are computed mechanically from the notebook's AST, so the
picture is the notebook's actual wiring, not a hand-drawn approximation.

Read the two together: this `.md` is the *prose* (what each function does and why); the `.html`
is the *map* (how the blocks feed each other). The cell numbers match — every "cell N" reference
below is node `#N` in the graph. In the graph you can:

- **pan** (drag the background), **zoom** (mouse wheel), and **drag** any cell to rearrange it;
- **click a cell** to open a side panel with its description, the symbols it *defines*, what it
  *depends on*, what *uses it*, and its full syntax-highlighted code;
- **filter** by phase (the coloured chips, top-right) or **search** code/titles (top-left box).

Regenerate it any time with `python3 _build_v3_graph.py` (re-reads the notebook).

> Reading order that works well: skim the graph end-to-end to see the spine
> (`Phase 0 setup → tools → the tool-loop graph → hardening primitives → the team`), notice how
> almost everything points back at the `llm()` factory (cell 5) and the tracer (cell 7), then
> drop into the prose below for any cell whose role isn't obvious from its code.

---

## Table of contents

- [Phase 0 — Imports, logging, config, the model factory](#phase-0)
- [Phase 0.5 — Observability (callback handler + graph views)](#phase-05)
- [Phase 1 — Cognitive substrate (thinking, structured routing, test-time compute)](#phase-1)
- [Phase 2 — Tools (`@tool`-decorated, sandboxed)](#phase-2)
- [Phase 3 — The tool loop, as a graph (+ agent-as-tool subagents)](#phase-3)
- [Phase 4 — The hardening stack, as small graphs](#phase-4)
- [Phase 5 — Planning & durable state](#phase-5)
- [Phase 6 — Context engineering: a `pre_model_hook`](#phase-6)
- [Phase 7 — The five-subagent team, as one graph](#phase-7)
- [Phase 8 — Running the team](#phase-8)
- [Phase 9 — v3 vs v2: what LangGraph buys you](#phase-9)
- [Phase 10 — Offline self-tests](#phase-10)
- [Phase 11 — A harder end-to-end build](#phase-11)
- [Cross-cutting design themes](#themes)
- [Notes on changes since first draft](#known-issue)

---

<a name="phase-0"></a>
## Phase 0 — Imports, logging, config, the model factory

### Imports *(cell 2)*
**Does:** pulls in the standard-library toolkit (`ast`, `py_compile`, `sqlite3`, `subprocess`,
`tempfile`, `uuid`, `Counter`, `dataclass`, `Path`, `typing`) and the LangChain/LangGraph
stack. The load-bearing imports name every seam the notebook uses downstream:
- **messages** — `AIMessage`, `HumanMessage`, `SystemMessage`, `ToolMessage`, `BaseMessage`,
  `RemoveMessage`, `trim_messages` (the message vocabulary the graphs pass around).
- **`tool`** — the decorator that turns a typed function into a LangChain tool.
- **`BaseCallbackHandler`** — the base class for the tracer (Phase 0.5).
- **`ChatOllama`** — the only concrete model class, built exclusively inside `llm()`.
- **graph primitives** — `StateGraph`, `START`, `END`, `MessagesState`, `add_messages`.
- **prebuilt loop pieces** — `ToolNode`, `tools_condition`, `create_react_agent`.
- **persistence** — `InMemorySaver` (checkpointer), `InMemoryStore` (long-term store).

**Why:** the *only* new dependencies vs v2 are `langgraph`/`langchain-core`/`langchain-ollama`.
Everything else is the same standard library as v2 — a deliberate signal that **LangGraph
replaces the orchestration, not the underlying tools or sandboxing.** `from __future__ import
annotations` lets the type hints (used by `@tool` schema derivation) stay lazy.

### `class _Fmt(logging.Formatter)` *(cell 3)*
**Does:** a colour-per-level log formatter. `format()` looks up an ANSI colour by `levelname`,
trims the logger name to its child suffix (`agent3.tool` → `tool`), and renders
`[LEVEL] subsystem | message`. The cell then builds the root `agent3` logger (handlers cleared,
`propagate = False`) and four children: `log_llm`, `log_tool`, `log_graph`, `log_sub`.
**Why:** identical in spirit to v2's formatter; only the subsystem names shift (`ollama`→`llm`,
`loop`→`graph`) to match LangGraph vocabulary. `propagate = False` stops duplicate root lines.
Verbosity is the single env var `AGENT_LOG_LEVEL` (default `INFO`).

### Configuration cell *(cell 4)*
**Does:** centralises every knob. The load-bearing values:
- **`OLLAMA_HOST`** — defaults to `http://localhost:8080` (the user's local Ollama proxy).
- **`MODELS`** — the *role → model* map: `reasoning` (`qwen3:32b`) for hard thinking
  (architect, verifier, planner, adversary); `fast` (`qwen3:8b`) for high-volume routine
  work; `summarizer` (`qwen3:8b`) for distillation. Each is independently overridable by env
  var. `MODEL_REASONING`/`MODEL_FAST` are convenience aliases.
- **Sandbox paths** — `WORKSPACE` (`v3_workspace/`, resolved + created eagerly),
  `AGENT_CODE_DIR` (`agent_code/`), `DB_PATH` (`dag.db`).
- **Limits** — `MAX_TOOL_OUTPUT=12_000`, `BASH_TIMEOUT_S=60`, `TEST_TIMEOUT_S=120`,
  `REQUEST_TIMEOUT_S=900` (the 32B model is slow on big contexts), `MAX_ITERATIONS=20`.
- **`BASH_BLOCKLIST`** — the destructive-fragment denylist (`rm -rf /`, `sudo`, `shutdown`,
  `mkfs`, fork-bomb, …); still a *speed bump, not a sandbox*.
- **`_HAS_PYTEST`** — `importlib.util.find_spec("pytest") is not None`, feature-detected once so
  the test runner can degrade to plain python.

**Why:** the single most important lever here is the **two-tier `MODELS` split** — cheap model
for volume, expensive model reserved for judging/architecting — carried over from v1/v2 intact.
Everything created eagerly at import means later cells can assume the sandbox exists.

### `_client(model, reasoning, temperature, num_predict) -> ChatOllama` *(cell 5)*
**Does:** the cached constructor. `@lru_cache(maxsize=32)` keyed on the full parameter tuple;
builds a `ChatOllama` with `base_url=OLLAMA_HOST`, the `reasoning` flag, `temperature`, a
`client_kwargs={"timeout": REQUEST_TIMEOUT_S}`, and `num_predict` only when a token cap is given.
**Why:** caching on the parameter tuple is a free, dependency-light **connection-reuse**
optimization — repeated calls with the same shape reuse one client and its connection pool.

### `llm(role="fast", *, reasoning=True, temperature=0.2, max_tokens=None) -> ChatOllama` *(cell 5)*
**Does:** the **v3 chokepoint** — v3's analogue of v2's `chat_complete()`. Resolves a *role*
(`reasoning`/`fast`/`summarizer`) or a literal model name through `MODELS.get(role, role)`, then
returns the cached `_client`. Graphs then call `.invoke` / `.batch` / `.bind_tools` /
`.with_structured_output` on the returned Runnable.
**Why:** the **`reasoning` flag is the knob that matters most.** qwen3 is a thinking model, and
`langchain-ollama` surfaces the thinking channel separately — `reasoning=True` routes `<think>`
into `msg.additional_kwargs["reasoning_content"]` and keeps `msg.content` clean (used for
free-text calls); `reasoning=False` disables thinking entirely (faster, used for JSON/structured
calls). **This is the v3 equivalent of v2's "JSON mode suppresses `<think>`" trick** — same
effect, expressed as a model flag rather than an API format field. One factory means backend
portability, uniform tracing, and a single place to retune everything.

### `ollama_healthcheck() -> bool` *(cell 5)*
**Does:** a *tags-only* probe. `GET /api/tags` (no generation), then for each role's model checks
presence by exact name or family-prefix match (`t.startswith(name.split(':')[0] + ":")`). Logs
each role's `[OK]`/`[MISSING]` and returns `True`/`False`.
**Why:** a fail-fast gate before live runs, exactly as v2 — far better than a cryptic mid-run
404. No generation means it's instant and free.

---

<a name="phase-05"></a>
## Phase 0.5 — Observability

v2 wrapped *every* model and tool call in a bespoke `rich` tracer. In LangGraph the idiomatic
seam is a **`BaseCallbackHandler`**: LangChain itself calls your hooks on every model/tool start
and stop, *no matter how deeply the graph nests*. So instrumentation moves from "wrap each call
site" to "register one handler and pass it in the run config."

### `_clip(s, n=None) -> str` *(cell 7)*
**Does:** truncates `s` to `n` chars (default `TRACE_PREVIEW=1600`), appending a
`\n... [+N chars]` marker. Coerces `None`/non-strings safely.
**Why:** the same "don't dump the world into the terminal" readability helper as v2.

### `thinking_of(msg: BaseMessage) -> str` *(cell 7)*
**Does:** the v3-specific accessor for qwen3's reasoning channel — reads
`additional_kwargs["reasoning_content"]`, falling back to `["thinking"]`, else `""`.
**Why:** every place that needs the model's `<think>` channel goes through this *one* function,
so the rest of the code never hard-codes the `additional_kwargs` key.

### `class RichTracer(BaseCallbackHandler)` *(cell 7)*
The live, streaming view — but now **driven by LangChain**, not manual wrapping. Because
LangChain fires the hooks, the *same* tracer works identically for a bare `llm.invoke()`, a
`ToolNode`, a `create_react_agent`, and the deep five-subagent graph — that uniformity is the
whole reason to use the callback seam. Its methods:

- **`__init__`** — a `threading.Lock` plus thread-safe counters (`calls`, `tokens`,
  `tool_calls`) and a `_starts` dict keyed by `run_id` for per-call timing.
- **`on` / `full`** *(properties)* — read `AGENT_TRACE` (`full`/`compact`/`off`); `off` makes
  every hook a no-op so production runs pay nothing.
- **`_emit(renderable, plain)`** — the dispatch: print a `rich` renderable if `rich` is present,
  else log the plain string. The single place the "degrade gracefully" choice lives.
- **`on_chat_model_start(serialized, messages, *, run_id)`** — increments `calls` under the lock,
  records `_starts[run_id]`, and (in full mode) panels the *prompt tail* (the latest message,
  `messages[0][-1]`) rather than re-dumping the whole transcript every turn — the same
  "don't re-print the world" choice as v2.
- **`on_llm_end(response, *, run_id)`** — pops the start time for latency, pulls the message out
  of `response.generations[0][0]`, splits it into `<think>` (via `thinking_of`) and answer,
  accumulates `usage_metadata["output_tokens"]` under the lock, and lists any requested
  `tool_calls`. Panels thinking in grey, answer in green with latency + token count, tool
  requests in yellow. The whole extraction is wrapped in `try/except` so a malformed response
  can't crash the run.
- **`on_tool_start(serialized, input_str, *, run_id)`** — increments `tool_calls`; panels the
  tool name + args in yellow.
- **`on_tool_end(output, *, run_id)`** — panels the result; **heuristically reddens** it if the
  text starts with `error`/`[error`/`reverted`/`traceback`, matching v2's colour convention.
- **`event(title, body="", style)`** — a *generic decision marker* the cognitive primitives call
  directly (routing choices, verifier scores, adversary findings, plan shapes). **This is the
  one place the notebook still narrates *imperatively* into the tracer**, because those are
  decisions, not model/tool calls LangChain would hook.
- **`summary()`** — prints the `calls / tokens / tool_calls` ledger as a `rich` table (or a log
  line) at the end of a run.

**Why:** like v2 it degrades gracefully (panels with `rich`, log lines without). The split
between hooked methods (model/tool calls) and the explicit `event()` is the key design point: the
framework narrates the *mechanical* calls for free; the agent narrates only its *decisions*.

### `tracer`, `CB`, `run_config(label="run", **extra) -> dict` *(cell 7)*
**Does:** `tracer` is the single module-level `RichTracer` instance. `CB = {"callbacks":
[tracer]}` is passed as `config=CB` to plain `.invoke`/`.batch` calls. `run_config(label)` is the
richer version for graphs — it bundles the callbacks **and** a unique
`configurable.thread_id` (`f"{label}-{uuid…}"`), merging any `**extra` (e.g.
`recursion_limit`).
**Why:** the `thread_id` is what the checkpointer keys persistence on, so **every live graph run
gets its own isolated, independently resumable/inspectable thread**. Plain calls don't need a
thread, so they use the lighter `CB`.

### `show_graph(app, title="")` *(cell 8)*
**Does:** renders a compiled graph as a Mermaid PNG via `app.get_graph().draw_mermaid_png()`,
falling back to `draw_ascii()` if rendering fails.
**Why:** the topology becomes a *picture* — used throughout to show the loop, the team, the
self-correcting cycle. Something v2 simply could not do.

### `stream_run(app, inputs, config=None, *, mode="updates")` *(cell 8)*
**Does:** runs `app.stream(...)` and, for each `{node: update}` chunk, logs `· node «name» -> …`,
tagging any message that carries a thinking channel (`[+think]`). Returns the final update.
**Why:** the post-`master_loop` replacement for v2's per-iteration logging — you watch the run
cross the graph **node-by-node, live**, instead of reading after-the-fact logs.

**Net effect:** v2 had one observability mechanism (the tracer). v3 has three feeding off the
framework — callback narration, graph streaming, and graph diagrams — and all three come from
instrumenting *once* (the handler) or for *free* (stream/draw are built in).

---

<a name="phase-1"></a>
## Phase 1 — Cognitive substrate

The "how the model thinks" layer, re-expressed with LangChain idioms: thinking via
`reasoning=True`, structured routing via `with_structured_output`, parallel sampling via
`.batch()`.

### `STRONG_SYSTEM_PROMPT` *(cell 10)*
**Does:** a constant encoding the agent's epistemics as five rules of engagement: never claim
behaviour without a runnable artifact; defer all questions to execution; a failing test/linter
is correct until proven otherwise; say "I don't know" rather than guess; the spec is the source
of truth.
**Why:** identical to v2 — the "thoughtful response" idea baked into a constant rather than
re-prompted ad hoc.

### `strip_think(text) -> str` *(cell 10)*
**Does:** regex-removes a literal `<think>…</think>` block (via `_THINK_RE`, `DOTALL`) and strips.
**Why:** even though qwen3 *usually* puts thinking in `additional_kwargs` now, some paths still
emit literal `<think>` in content, so this defensive parser stays.

### `split_think(msg) -> (thinking, answer)` *(cell 10)*
**Does:** **dual-mode.** Given a `BaseMessage`, it prefers the structural channel (`thinking_of`)
and only falls back to a regex search of the content; given a raw string, it regex-splits.
Returns `(thinking, answer)` in both cases.
**Why:** this is what lets the rest of the code treat "thinking" uniformly whether it arrived
*structurally* (new qwen3) or *inline* (`<think>` tags). One accessor, two wire formats.

> Note: `strip_code_fences` is defined later (Phase 4, cell 25) but serves the same defensive
> role — pulling raw source out of a stray markdown fence.

### `@dataclass ThoughtfulResponse` + `think_then_answer(...)` *(cell 11)*
**`ThoughtfulResponse`** carries `thinking`, `answer`, `output_tokens`.
**`think_then_answer(query, role="fast", temperature=0.3, max_tokens=2048,
system=STRONG_SYSTEM_PROMPT) -> ThoughtfulResponse`**
**Does:** the basic single-shot call. Builds `[SystemMessage, HumanMessage]`, invokes the model
with `reasoning=True`, separates channels with `split_think`, and reads real token use from
`msg.usage_metadata`.
**Why:** v2's primitive, re-pointed at `ChatOllama`. Returning real `output_tokens` lets callers
budget on actual usage, not guesses.

### Structured routing — `with_structured_output` replaces hand-parsed JSON *(cell 12)*
The cleanest single win of the rebuild. v2 hand-wrote JSON prompts and *tolerantly parsed* the
result. v3 hands the framework a **Pydantic schema** and lets it constrain *and* validate.

- **`class Difficulty`** (`Literal["trivial"…"extreme"]`) — the difficulty schema.
- **`estimate_difficulty(query) -> str`** — a cheap classifier on the **fast** model with
  `reasoning=False`, `temperature=0.0`, `.with_structured_output(Difficulty)`. Mapped downstream
  through `THINKING_BUDGETS` (`trivial:256 … extreme:6000`) to a `num_predict` budget. Wrapped in
  `try/except` returning `"medium"`.
- **`class ProblemKind`** (`type` + a one-sentence `reason`) — the problem-kind schema.
- **`classify_problem(query) -> dict`** — returns `convergent / divergent / exploratory /
  structural` with a reason, the system prompt spelling out each type. Mapped through
  `TYPE_STRATEGY` to a strategy name. `try/except` returns `{"type": "convergent", …}`.

**Why:** both classifiers use `temperature=0.0` + `reasoning=False` for cheap determinism, and
both keep a safe-default `try/except` — because even with schema enforcement a backend hiccup
shouldn't halt a run. **Budgets are scaled up vs the article** because a thinking model spends
tokens on `<think>` too. **Honest framing from v2 survives:** for spec-driven *coding* almost
everything is convergent/structural, so the type axis is near-constant here, but it's kept as the
bridge to the non-coding domains this notebook leads toward.

### Test-time compute — `.batch()` replaces `ThreadPoolExecutor` *(cell 13)*

- **`class Verdict`** — `score: int (ge=1, le=10)` + `reason`. The verifier's output schema.
- **`self_consistency(query, k=3, role="fast") -> dict`**
  **Does:** builds `k` identical `[System, Human]` message lists, `.batch()`es them at
  `temperature=0.7` for diversity, strips thinking from each, buckets the answers by their first
  60 lowercased chars, and returns the majority bucket with `votes`, `agreement`, and
  `all_samples`. Emits a `tracer.event` with the agreement ratio.
  **Why:** the 60-char prefix is the same cheap, embedding-free clustering trick as v2; agreement
  doubles as a confidence signal. `.batch()` runs the samples concurrently *for you* — no manual
  thread pool.
- **`verifier_score(question, candidate, role="reasoning") -> dict`**
  **Does:** one structured 1–10 score from the **reasoning** model
  (`with_structured_output(Verdict)`, temp 0, `reasoning=False`). System prompt: score *facts and
  correctness, not style*. `try/except` returns `score:0` on failure. Emits a `tracer.event`.
  **Why:** the judge is where the strong model's quality matters; structured output guarantees a
  numeric score you can branch on.
- **`class Ranking`** — `best_index: int` + `reason`. The ranker's output schema.
- **`asymmetric_solve(query, n_candidates=3) -> dict`**
  **Does:** the **verifier-asymmetry** pattern. Generates `n` cheap candidates on the **fast**
  model via `.batch()` (temp 0.7), then spends **one** structured call on the **reasoning** model
  to pick the best index (clamped into range). Falls back to candidate #0 on parse failure. Emits
  a `tracer.event`.
  **Why it's the key cost trick:** generation is expensive per-token and parallelizable;
  *judging* is where the strong model matters most, and you pay for it **once**.
- **`adaptive_think(query, route=True) -> dict`**
  **Does:** the dispatcher tying both axes together — `estimate_difficulty` → budget,
  `classify_problem` → strategy, then runs the chosen strategy (`self_consistency` /
  `asymmetric_solve` / `decompose` / `wide_pass` / single pass). `route=False` collapses to a
  budget-only single pass. Each branch emits a `tracer.event` so the routing decision is visible.
  **Why:** `route=False` is the built-in ablation handle ("does routing actually help?"). The
  function is the single entry point that makes the two routing axes act together.

---

<a name="phase-2"></a>
## Phase 2 — Tools

Everything the agent can *do*. v2 registered tools as a `name→callable` dict and hand-wrote their
JSON schemas. **In v3 each tool is a `@tool`-decorated function, and LangChain derives the JSON
schema from the type hints + docstring.** The *bodies* are the same sandboxed v2 tools; only the
registration changes. Two design rules still dominate: **paths are sandboxed to `WORKSPACE`** and
**outputs are truncated**.

### `_safe_path(path) -> Path` *(cell 15)*
**Does:** resolves `path` (joining onto `WORKSPACE` if relative), then `relative_to(WORKSPACE)`
— raising `ValueError` if it escapes.
**Why:** the containment boundary for every file op. (It resolves *after* joining, so symlink
games are the residual risk — acceptable for a local single-user tool.)

### `_truncate(s, limit=MAX_TOOL_OUTPUT) -> str` *(cell 15)*
**Does:** clips with a `[truncated N chars]` marker.
**Why:** keeps a single runaway file dump or test log from blowing the context window.

### `SNAPSHOTS` *(cell 15)*
**Does:** a module-level `Dict[str, Optional[str]]` — the in-memory undo stack mapping a file
path to its prior content (or `None` if the file was new).
**Why:** chosen over git because the workspace already lives inside a repo and the notebook
refuses to nest one.

### File/shell tools (all `@tool`) *(cell 15)*
- **`read_file(path, start_line=None, end_line=None) -> str`** — reads with **1-indexed line
  numbers prefixed**, optional range, `errors="replace"` so binary junk doesn't crash. Returns
  friendly `Error: …` strings instead of raising (the model reads them as observations).
- **`write_file(path, content) -> str`** — **snapshots prior content into `SNAPSHOTS` before
  writing**, creates parent dirs, reports created-vs-updated. The snapshot is what makes
  `revert_file` possible.
- **`revert_file(path) -> str`** — pops the snapshot and restores it, or **deletes** the file if
  it was new (`prev is None`). The in-memory undo stack in action.
- **`grep(pattern, path=".", recursive=True) -> str`** — shells out to real `grep -rn`, clamped
  to `_safe_path`, 30s timeout, truncated to 8000 chars.
- **`glob_files(pattern) -> str`** — glob scoped to `WORKSPACE`, filtered through
  `is_relative_to(WORKSPACE)`, capped at 200 hits.
- **`bash(command) -> str`** — runs in `WORKSPACE` with `BASH_TIMEOUT_S`, after scanning
  `BASH_BLOCKLIST`. **Honest caveat:** the blocklist is a *speed bump, not a sandbox* —
  `shell=True` means a determined model could evade it; acceptable because the workspace is
  throwaway and local.

**Why (whole group):** every tool returns a string (errors included) rather than raising, because
the loop feeds tool output back to the model as a `ToolMessage` — a thrown exception would break
the graph, but a `"Error: file not found"` string is something the model can *react* to.

### Coding-specific tools (the quality gates) *(cell 16)*
- **`lint_python(code) -> {passed, errors}`** *(plain helper, not `@tool`)* — writes to a temp
  file, `py_compile`s it (syntax gate), then walks the AST to flag bare `except:`. Deliberately
  minimal — a fast, dependency-free *must-pass* filter, not a full linter. Kept plain because
  `write_code` and the spec layer call it directly.
- **`_run_tests(test_code, timeout=TEST_TIMEOUT_S) -> dict`** *(plain helper)* — writes a test
  module and runs it with **pytest if available, else plain python**, then regex-parses
  `N passed`/`N failed` into a structured dict, inferring pass/fail from the return code when
  counts are absent. Times out gracefully into a structured failure. Kept plain because the
  `run_tests` tool **and** the spec layer both call it.
- **`write_code(filename, content) -> str` (`@tool`)** — the **lint-gated write**: rejects unless
  the filename is a bare `*.py` (no `/`, no `..`) *and* the content lints clean; only then
  persists to `agent_code/`. **This is the central reliability mechanism — broken code never
  reaches disk**, so downstream test runs never fail for trivial syntax reasons.
- **`run_python(code) -> str` (`@tool`)** — writes the snippet with `agent_code/` prepended to
  `sys.path`, runs it as a subprocess, returns `exit=… ` + output. A fresh subprocess per run
  gives real isolation and a hard timeout — the no-Docker stand-in for the article's sandbox.
- **`run_tests(test_code) -> str` (`@tool`)** — the `@tool` wrapper over `_run_tests`, returning
  a one-line `all_passed=… passed=… failed=…` summary plus output for the model to read.
- **`TOOLS_BASE` / `TOOLS_BY_NAME`** — the base toolset every coding agent gets, and a name→tool
  map. This list is what gets handed to every `ToolNode` and every subagent — it is, in effect,
  v3's tool registry (see the Phase 7 note on "MCP").

**Why the plain-vs-`@tool` split:** `lint_python` and `_run_tests` are *internal machinery* that
multiple layers call directly with Python types; the `@tool` versions exist only so the *model*
can invoke them through the loop with JSON args. Same body, two callers.

---

<a name="phase-3"></a>
## Phase 3 — The tool loop, as a graph

v2's `master_loop()` was a hand-written *perception → action → observation* while-loop. **In
LangGraph that loop *is* the graph.**

### `build_agent_graph(tools, system=STRONG_SYSTEM_PROMPT, role="fast", reasoning=True, checkpointer=None)` *(cell 18)*
**Does:** the v3 replacement for `master_loop`. Binds the tools to the model
(`llm(role, reasoning).bind_tools(tools)`) and assembles a two-node `StateGraph` over the
built-in `MessagesState`:

```
START → agent → (tools_condition) → tools → agent → … → END
```

Its internals:
- **`agent_node(state)`** — prepends `STRONG_SYSTEM_PROMPT` at call time *if* the first message
  isn't already a `SystemMessage`, then invokes the tool-bound model and returns the new message.
  *Choice — prepend, don't store:* the system prompt always leads without being persisted
  repeatedly into state.
- **`ToolNode(tools)`** — the prebuilt node that executes whatever tools the model requested,
  appending their results as `ToolMessage`s. Replaces v2's hand-written `_run_tool_call` +
  dispatch table entirely.
- **`tools_condition`** — the prebuilt conditional edge: if the last AI message has tool calls,
  route to `tools`; otherwise end. This is v2's "did the model request tools?" check, now a
  library function.
- **Compiled with a checkpointer** (`InMemorySaver` by default) — so every run is a **resumable
  thread** keyed by `run_config`'s `thread_id`. Durable state and time-travel come *for free*
  from compiling the graph; v2 had nothing equivalent for the loop itself.

**Why:** `.bind_tools(tools)` replaces v2's manual `_fn()` schema builder; `ToolNode` +
`tools_condition` replace the hand-written dispatch + "should I loop?" check. The entire
perception/action/observation cycle is now three library calls and three edges.

### `coding_agent` *(cell 18)*
**Does:** the module-level lead agent — the full base toolset + the strong system prompt on the
**fast** model. Cells 20–21 visualise it (`show_graph`) and run it on a tiny `hello.txt`
round-trip (`stream_run`), so you see the `agent ⇄ tools` cycle live.

### Subagent discipline = the agent-as-tool pattern *(cell 19)*
v2's `spawn_subagent` was a recursive call into `master_loop`. v3 keeps the same *discipline* but
packages it the LangGraph way.

- **`SUBAGENT_SYSTEM`** — the focused prompt: one subtask, no clarifying questions (make a
  reasonable assumption), and *"your final message is the ONLY thing the parent sees — make it
  self-contained."*
- **`spawn_subagent(prompt, tools=TOOLS_BASE, system=SUBAGENT_SYSTEM, role="fast") -> str`**
  **Does:** builds a fresh `build_agent_graph`, invokes it on the prompt with a unique thread id
  and a generous `recursion_limit=2*MAX_ITERATIONS`, then walks the messages **in reverse** and
  returns the **last non-empty AI message** (think-stripped). Returns a sentinel if the subagent
  produced nothing.
  **Why:** this enforces the **context-isolation** property — the parent never sees the
  subagent's tool transcript, only its distilled summary.
- **`make_subagent_tool(name, description, system, tools=TOOLS_BASE, role="fast") -> StructuredTool`**
  **Does:** wraps a `spawn_subagent` call as a `StructuredTool` via `from_function`. **This is the
  agent-as-tool pattern:** a parent agent can now *delegate* by calling the subagent like any
  other tool, and LangGraph runs the entire sub-graph inside that one tool call.

**Why this is cleaner than v2:** in v2, context isolation was a *convention* you had to respect
(return only the last message). In v3 it's *structural* — the sub-graph is a black box exposing a
single string return, so its internal steps are physically incapable of leaking into the parent's
context.

---

<a name="phase-4"></a>
## Phase 4 — The hardening stack, as small graphs

v2's four hardening primitives were plain functions with internal loops. **In v3 each becomes a
tiny graph — which is exactly where LangGraph earns its keep: loops and branches are edges**,
inspectable and drawable.

### `architect_editor_solve` — a linear two-node chain *(cell 23)*
Separation of deliberation from transcription. Schemas: `Section` (`section`, `intent`,
`key_constraints`) and `ArchitectPlan` (`plan: List[Section]`). State `_AEState`
(`task`, `plan`, `output`).
- **`_architect_node(state)`** — the **reasoning** model with
  `with_structured_output(ArchitectPlan)` produces a *structured* plan (sections with intents and
  constraints) but **explicitly not code**. Degrades to an empty plan on failure; emits a
  `tracer.event` naming the sections.
  **Why:** the Pydantic schema is what guarantees the architect's constraints travel verbatim to
  the editor.
- **`_editor_node(state)`** — the **fast** model with **`reasoning=False`**,
  `max_tokens=3072`, executes the plan into the final artifact ("do NOT redesign").
  **Why disable thinking here:** the architect already deliberated, so the editor just
  transcribes — turning off `<think>` makes it dramatically faster *and* stops thinking from
  eating the token budget and truncating generated code. **v3's spelled form of v2's `/no_think`
  editor trick** (a model flag instead of a prompt prefix).
- **`architect_editor_app`** — `START → architect → editor → END`, compiled.
- **`architect_editor_solve(task) -> dict`** — the thin function wrapper returning
  `{plan, output}`.

### `self_refine` — generate → critique → refine, as a loop *(cell 24)*
State `_RefineState` carries `query`, `current`, `critique`, `iteration`, `max_iter`, `history`.
- **`_gen_node(state)`** — initial generation via `think_then_answer`; seeds `iteration=0` and
  the history.
- **`_critique_node(state)`** — the **fast** model (reasoning on, `max_tokens=600`) plays a
  *strict reviewer* against the current output, listing 2–5 specific issues.
- **`_refine_node(state)`** — produces a refined version addressing every critique point,
  increments `iteration`, appends to `history`.
- **`_refine_route(state) -> "critique" | END`** — the **conditional edge**: loop
  `refine → critique` while `iteration < max_iter`, else end.
- **`self_refine_app`** — wired `generate → critique → refine`, with the conditional loop back.
- **`self_refine(query, iterations=2) -> dict`** — wrapper returning `{final, history,
  iterations_run}`.

**Why:** critique and refine are **separate calls** (find flaws, then fix them) because
separating the two produces sharper critiques — a choice carried from v2. What's new is that the
loop is a *visible cycle in the graph*, not a hidden `for`.

### `code_with_tests` — generate → verify, as a loop *(cell 25)*
The single most important pattern for code reliability, now a two-node graph with a feedback
cycle. State `_CWTState` carries the task, test code, current code, feedback, round counter,
status, and history.
- **`_cwt_generate(state)`** — generates code (stripping any markdown fence via
  `strip_code_fences`), appending the previous failure as `PREVIOUS ATTEMPT FAILED:` when
  present; increments `round`.
- **`_cwt_verify(state)`** — **lint-gates first** (short-circuits without wasting a test run when
  lint fails), then writes a `_candidate.py` and runs the **real** `_run_tests`. Records pass/fail
  into history and emits a `tracer.event`.
- **`_cwt_route(state) -> "generate" | END`** — the conditional edge: end on `status == "passed"`
  or when `round >= max_rounds`, else loop back to `generate`.
- **`strip_code_fences(text) -> str`** *(defined in this cell)* — strips thinking, then pulls the
  inner block out of a ```` ```python ... ``` ```` fence if present.
- **`code_with_tests_app`** + **`code_with_tests(code_gen_task, test_code, max_rounds=3) -> dict`**
  — the compiled graph and its wrapper (`{final_code, rounds_used, status, history}`).

**The defining choice, preserved verbatim from v2:** the feedback fed back is the **verbatim test
stdout**, never a paraphrase — the ground-truth error is the most useful possible signal for the
next attempt. Cells 27–28 visualise the self-refine loop and run `code_with_tests` for real on an
`inc(n)=n+1` task that loops on failure.

### `adversarial_probe` — red-teaming, one structured call *(cell 26)*
Schemas: `Attack` (`category`, `scenario`, `why_it_breaks`, `severity`) and `AttackList`.
- **`adversarial_probe(target_description, candidate_output, n_max=4) -> list`**
  **Does:** the **reasoning** model plays "hostile adversary" with
  `with_structured_output(AttackList)`, at **higher temperature (0.4)** to encourage creative
  attacks, returning a typed list of `Attack` dicts (capped at `n_max`). Emits a `tracer.event`
  listing severities. `try/except` returns `[]`.
  **Why:** purely *advisory* — it surfaces risks for the reviewer rather than gating, because a
  generated attack might be a false alarm. The Pydantic schema is what makes the attacks
  structured and iterable instead of prose to re-parse.

---

<a name="phase-5"></a>
## Phase 5 — Planning & durable state

Same durable substrate as v2; the only change is that `make_plan` now returns a *validated
Pydantic object*. LangGraph then adds two persistence layers on top (checkpointer + Store), wired
in Phases 6–7.

### `make_plan(goal, role="reasoning") -> Plan` *(cell 30)*
Schemas: `PlanStep` (`step_id`, `description`, `depends_on`, `expected_artifact`) and `Plan`
(`goal` + `steps`).
**Does:** asks the **reasoning** model (temp 0, `max_tokens=2000`) with
`with_structured_output(Plan)` for a dependency-ordered plan and gets back a *typed, validated*
object — no hand-parsing, no `None`. Emits a `tracer.event` showing the dependency edges.
Degrades to an empty `Plan` on failure so callers never crash.
**Why:** structured output makes the plan a first-class object the team can iterate over slice by
slice.

### `class TaskDAG` *(cell 31)*
**Does:** a **SQLite-backed** DAG (`node_id, title, status, attempts, depends_on`):
- `add_node` uses `INSERT OR REPLACE` (idempotent re-seeding).
- `all_nodes()` dumps the table.
- `ready_nodes()` returns pending nodes whose deps are all `done` — the scheduler primitive.
- `set_status` also increments `attempts`, a free retry counter.

`isolation_level=None` (autocommit) persists each update immediately, so the DAG survives a kernel
restart.
**Why:** carried over **unchanged** from v2. *Note:* in v3 the team's control flow is the **graph
topology** (Phase 7), so `TaskDAG` is retained mainly as a durable record / carried-over
substrate rather than the live scheduler it was in v2.

### `class BiTemporalMemory` *(cell 31)*
**Does:** facts with `valid_from`/`valid_to` intervals. `store` appends a record; `invalidate`
sets `valid_to` + a reason (so superseded facts are **invalidated, not deleted** — you can always
ask "what did the agent believe *then*"); `query_valid(kind)` returns currently-valid facts;
`recall(query, k)` does **keyword-overlap** ranking (set-intersection of word tokens), no
embeddings.
**Why:** unchanged from v2 — the same deliberate trade of recall quality for zero dependencies
and full transparency, fine because the corpus is one run's worth of facts.

### The spec layer — definition-of-done as executable contract *(cell 32)*
Unchanged from v2 and still the linchpin of the notebook's epistemics:
- **`write_definition_of_done(criteria, import_line="") -> dict`** — persists the contract to
  `DEFINITION_OF_DONE.json` and returns it.
- **`compile_test_suite(criteria, import_line="") -> str`** — **codegens a real pytest module** —
  prepends `agent_code/` to `sys.path`, adds the `import_line`, and turns each
  `{"name", "check"}` into a `def test_name(): assert <check>`.
- **`spec_verify(contract) -> dict`** — compiles the suite and runs it via `_run_tests`.

**Why:** "Done" is **not** prose the model self-grades against — it is *compiled into tests that
execute against the agent's code*. The suite is green or the work isn't done, full stop.

---

<a name="phase-6"></a>
## Phase 6 — Context engineering: a `pre_model_hook`

The insight from v2 still holds: in a coding tool loop the context grows by accumulated **tool
observations** (file dumps, test logs), not by user turns. v2 trimmed *inside* its loop. LangGraph
gives the loop a dedicated seam — a **`pre_model_hook`** that runs immediately before every model
call, receives the full state, and returns what the model should actually see.

### `make_context_hook(max_recent=6, memory=None)` *(cell 34)*
**Does:** returns a `hook(state)` that implements **trim → reinject** *non-destructively*:
- Short-circuits (returns `{}`) until there are more than `max_recent + 2` messages.
- Keeps `head` (the system/first message), `anchor` (the original human task, found in
  `msgs[1:3]` — never trimmed, because losing the task is catastrophic), and the last
  `max_recent` messages verbatim.
- Everything between is `dropped` and replaced by a single `[context note]` `SystemMessage`
  recording how many steps were elided.
- If a `BiTemporalMemory` is passed, it `recall`s facts relevant to the anchor task and injects
  them inside a `<durable_memory>` block — the *reinject* half of the pattern. Emits a
  `tracer.event` showing the new window size.
- **Returns `{"llm_input_messages": …}`, not `{"messages": …}`.** This is the crucial LangGraph
  mechanic: `llm_input_messages` changes only *what the model sees this turn*; the full transcript
  stays untouched in state and in the checkpointer. **v3 gets non-destructive trimming for free**
  — v2's manual trim actually dropped messages from the working list.

### `build_managed_agent(...)` *(cell 34)*
**Does:** `create_react_agent(llm(role, reasoning=True), tools, prompt=system,
pre_model_hook=make_context_hook(max_recent, memory), checkpointer=InMemorySaver())`.
`create_react_agent` is LangGraph's **prebuilt** version of `build_agent_graph`; it accepts a
`pre_model_hook` directly, so wiring in a bounded context window is a one-liner. `managed_agent`
is the module-level instance; cell 35 draws it.
**Why this is the cleanest section of the rebuild:** v2 needed a whole `ContextManager` class
(split / trim / distill / render_block / consolidate) plus a `managed_loop` to thread it through.
v3 expresses the same *trim + reinject* behaviour as a single hook function on a prebuilt agent.
(The heavier v2 machinery — LLM-based distillation and the writer+critic consolidation guard — is
**not** reproduced here; v3's hook does the deterministic trim + keyword recall, trading the
distillation step for framework simplicity. If you need the distill/consolidate layer, that's the
one place v3 is *lighter* than v2 by design.)

---

<a name="phase-7"></a>
## Phase 7 — The five-subagent team, as one graph

v2 hand-routed five subagents through a `TaskDAG` scheduler (`agent_run` pulling
`ready_nodes()`). **In v3 the DAG *is* the graph topology** and each subagent is a node; the
control flow v2 expressed imperatively is now *declared as edges*.

### `class TeamState` *(cell 37)*
**Does:** one shared `TypedDict` that flows through every node: `task`, `target_filename`,
`contract`, `plan`, `test_result`, `review`, `report`, `attempts`, `max_attempts`, `notes`.
**Why:** a single shared state (vs v2's `CodingAgent` container passed between subagents) is the
LangGraph way — nodes read and write slices of it, and the framework merges their returns.
`_note(state, msg)` is a tiny helper that appends to the running `notes` list (the run's audit
trail).

### The five nodes — each reuses a Phase-4/5 primitive *(cell 37)*
- **`planner_node(state)`** — runs `make_plan` and stores the steps into state. Produces the
  roadmap.
- **`implementer_node(state)`** — the workhorse. If the previous attempt failed, prepends the
  **verbatim** prior test stdout to the task; drafts the target file with **architect/editor**
  (`architect_editor_solve`), strips fences, and persists via the **lint-gated** `write_code`.
  Increments `attempts`.
  **Why:** the same external-feedback discipline as `code_with_tests`, here split across the
  graph's retry loop.
- **`tester_node(state)`** — runs `spec_verify` **independently** of the implementer; records
  pass/fail into `test_result`. The clean separation between "the builder thinks it passes" and
  "an independent step confirms it passes."
- **`reviewer_node(state)`** — reads the file, scores it with `verifier_score`, red-teams it with
  `adversarial_probe`, stores the review. Advisory, not gating. Handles the missing-file case.
- **`report_node(state)`** — uses `self_refine` (1 iteration) to write a concise `REPORT.md`
  grounded in the run's `notes` (so the report describes what *actually happened*, not a
  hallucinated narrative).

### `tester_route(state) -> "implementer" | "reviewer"` — the self-correcting loop *(cell 37)*
**Does:** the one **conditional edge** that makes the team more than a pipeline: if the tester
passed, go to `reviewer`; if it failed *and* `attempts < max_attempts`, go **back to
`implementer`** (carrying the failure); if the budget is spent, give up and review anyway.

```
START → planner → implementer → tester ─(pass)→ reviewer → report_writer → END
                       ▲                 │
                       └──(fail, < max)──┘
```

**Why:** this is v2's implementer↔tester retry, now a *visible cycle in the graph*.

### `build_team_graph(checkpointer=None)` / `team_app` / `run_team(...)` *(cell 37)*
**Does:** `build_team_graph` wires the five nodes and the conditional edge and compiles with a
checkpointer; `team_app` is the module-level instance. `run_team(task, target_filename, contract,
max_attempts=2, stream=True) -> dict` seeds the initial state and either streams the run
(`stream_run`, then reads back the final checkpoint via `get_state`) or invokes it, with
`recursion_limit=50` to accommodate the retry loop.
**Why a graph instead of a hand-routed DAG:** the dependency graph is now *data the framework
executes and can draw*, not imperative scheduling code. Re-wiring the pipeline means editing
edges, and `show_graph(team_app)` (cell 38) renders the whole team — including the
self-correcting loop — as a diagram.

**On the "MCP-style registry":** v2 built an explicit `MCPTool`/`mcp_registry` to demonstrate the
pattern. v3 notes (in the Phase 7 markdown) that **the registry already exists** — the typed
`@tool` set / `ToolNode` from Phase 2 *is* a uniform registry with names, descriptions, and JSON
schemas, which is exactly what MCP provides. So v3 doesn't reimplement it; the framework's tool
abstraction subsumes it.

---

<a name="phase-8"></a>
## Phase 8 — Running the team

The driver runs the team on a deliberately simple, fully-deterministic task (FizzBuzz — *not*
dengue, to prove the engine is general):

1. **`ollama_healthcheck()`** (cell 40, tags-only, fail-fast).
2. Define `TASK_8` + five `CONTRACT_8` criteria and persist the **definition of done** via
   `write_definition_of_done` (cell 41).
3. **`run_team(TASK_8, "solution.py", CONTRACT_8, max_attempts=2)`** (cell 42) — streamed, so you
   watch `planner → implementer → tester` loop back on a red test, then `reviewer →
   report_writer`.
4. Print attempts / test result / review, read back `REPORT.md`, and dump `tracer.summary()`.

---

<a name="phase-9"></a>
## Phase 9 — v3 vs v2: what LangGraph buys you

The notebook's own comparison table, plus a structural **census** (`_census`, cell 44) that
introspects each compiled graph's nodes and edges with **no model calls**. The mapping, in one
place:

| capability | v2 (from scratch) | v3 (LangGraph) |
|---|---|---|
| the tool loop | `master_loop()` while-loop | `StateGraph` + `ToolNode` + `tools_condition` |
| structured output | prompt + tolerant JSON parse | `.with_structured_output(PydanticModel)` |
| parallel sampling | `ThreadPoolExecutor` | `llm.batch([...])` |
| subagents | recursive `spawn_subagent` | compiled sub-graph as a `@tool` (agent-as-tool) |
| context window | manual trim in the loop | `pre_model_hook` returning `llm_input_messages` (non-destructive) |
| the team | imperative `TaskDAG` routing | conditional edges (the DAG *is* the graph) |
| **persistence** | sqlite DAG only | **checkpointer** (resume, time-travel) + **Store** |
| **streaming** | custom logging | `graph.stream(...)` node-by-node |
| **observability** | bespoke `rich` tracer wrapping every call | a `BaseCallbackHandler` LangChain drives |
| **visualisation** | — | `graph.draw_mermaid_png()` |
| **human-in-the-loop** | — | `interrupt()` / `interrupt_before` (available, not yet used) |

### `_census(app) -> dict` *(cell 44)*
**Does:** pure introspection — `app.get_graph()`, then the node names (minus `__start__`/`__end__`)
and edge count, run over `coding_agent`, `managed_agent`, `self_refine_app`,
`code_with_tests_app`, and `team_app`. Also prints the base toolset and `team_app.checkpointer`'s
type.
**Why:** confirms the base toolset (the de-facto "MCP registry") and that `team_app` carries a
checkpointer — all without burning a single model call.

---

<a name="phase-10"></a>
## Phase 10 — Offline self-tests

### `check(name, cond)` / `section(title)` *(cell 46)*
**Does:** a tiny harness — `check` records `(name, bool)` into `_results` and prints PASS/FAIL;
`section` prints a header. The cell then exercises everything that needs **no model calls**:
- **Tools** — round-trip write/read, `bash` echo, `bash` blocklist, lint-gated `write_code`
  (clean + rejected), `run_python`, and a `_safe_path` path-escape block.
- **Parsers + lint + tests** — `strip_think`, `split_think`, `strip_code_fences`, `lint_python`
  (good + bad), `_run_tests`.
- **DAG + memory + spec** — `TaskDAG` dependency gating + unlock, `BiTemporalMemory` recall +
  invalidation, `spec_verify` on a trivial contract.
- **Schemas + topology** — the Pydantic schemas, `make_subagent_tool`, and that **every graph
  compiles with the expected topology** (`coding_agent` has a `tools` node, the team has its five
  worker nodes, `managed_agent` has ≥3 nodes).

### Results roll-up *(cell 47)*
**Does:** sums passes, prints any failures, and `assert`s no failures.
**Why:** the plumbing is testable in seconds without burning GPU time — the same "the plumbing is
testable offline" discipline as v2.

---

<a name="phase-11"></a>
## Phase 11 — A harder end-to-end build

A harder task (`BoundedCounter`, a small LRU-ish data structure with `add`/`top`/`keys` and
capacity eviction) driven through the **same** Phase-7 team graph, to show the engine scales past
toy tasks.
- **`TASK_11` + `CONTRACT_11`** *(cell 49)* — the criteria are written as **one-line lambda
  checks** (e.g. `(lambda c=BoundedCounter(2): (c.add('a'), c.add('b'), c.add('c'), 'a' not in
  c.keys()))()[-1] is True`) so the contract stays a pure data structure.
- **`run_team(..., max_attempts=3)`** *(cell 50)* — streamed; loops implementer↔tester until the
  suite is green or the budget is spent.
- **Independent re-verification** *(cell 51)* — we recompile the contract and run `spec_verify`
  **ourselves**, not trusting the team's own word — the no-claim-without-evidence ethos applied to
  the *whole run*. Then inspect `counter.py` and `REPORT.md`.
- **Replay** *(cell 52)* — prints the run's per-node `notes`, and — because the team compiled with
  a checkpointer — the full per-node state history of the thread is available to inspect.

---

<a name="themes"></a>
## Cross-cutting design themes

The real lessons, mostly inherited from v2 and re-grounded in the framework:

1. **One model-construction chokepoint (`llm`).** Backend portability, uniform tracing, and a
   single place to retune `reasoning`/temperature/budget. The graphs operate on a `Runnable`, so
   swapping backends never touches them.

2. **Two-tier model economics.** Cheap model for high-volume generation and routine work;
   expensive model reserved for *judging/architecting*. `asymmetric_solve` is the purest
   expression — parallel cheap generation via `.batch()`, one strong ranking call.

3. **No claim without a runnable artifact.** Lint gate → real test execution → *independent*
   re-verification. The definition of done is *compiled into tests*, not self-graded; test
   failures feed back **verbatim**.

4. **Structure replaces tolerance where the framework allows it.** v2 hand-parsed JSON tolerantly
   everywhere; v3 uses `with_structured_output(PydanticModel)` so the framework constrains *and*
   validates. The tolerant parsers (`strip_think`, `strip_code_fences`) survive only for the
   free-text paths that genuinely need them.

5. **Loops and branches are edges.** Every place v2 had a hidden `for`/`while`/`if` (the tool
   loop, self-refine, code-with-tests, the team's retry), v3 has an explicit, drawable,
   inspectable graph edge. The control flow is *data the framework executes*.

6. **Context as a managed resource — non-destructively.** The `pre_model_hook` returns
   `llm_input_messages`, bounding the model's *view* while the checkpointer keeps the full
   history. Trimming no longer means forgetting.

7. **Persistence and observability come from compiling the graph.** A checkpointer makes every run
   a resumable, inspectable thread; `graph.stream` and `draw_mermaid_png` give live and visual
   views — all for free, none of it hand-built.

8. **Subagents are structurally isolated.** Agent-as-tool means a subagent's transcript *cannot*
   leak into the parent — context isolation is enforced by the architecture, not by convention.

9. **Honest about where it simplifies.** Like v2's caveats (no Docker, no ChromaDB, no git,
   near-constant type axis, `bash` is a speed bump), v3 openly drops v2's LLM-based context
   *distillation* and the writer+critic *consolidation* guard in favour of a deterministic trim +
   keyword recall in the hook. The intent is a *teaching* engine you can read end-to-end against
   its v2 sibling — not a hardened product.

---

<a name="known-issue"></a>
## Notes on changes since first draft

An earlier draft of this walkthrough flagged a corrupted line in **cell 7**
(`RichTracer.on_llm_end`) where a stray URL had been pasted into an identifier
(`gen = responshttp://…e.generations[0][0]`), which was a `SyntaxError`. **That has since been
fixed in the notebook** — cell 7 now reads the correct `gen = response.generations[0][0]`, so the
tracer's per-response panel (thinking / answer / token accounting) works as intended. No
outstanding correctness issues are known in the current notebook; the offline self-tests in
Phase 10 (cell 46) are written to pass without the backend and gate the live phases.
