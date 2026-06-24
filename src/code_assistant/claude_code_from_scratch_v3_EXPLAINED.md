# `claude_code_from_scratch_v3` — Code Walkthrough & Design Rationale

A detailed reference for every class and function in
`claude_code_from_scratch_v3.ipynb`, with the *why* behind each implementation choice.

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

### Imports
The only new dependencies vs v2 are the LangChain/LangGraph stack: `langgraph`,
`langchain-core`, `langchain-ollama`. Everything else (`sqlite3`, `subprocess`, `ast`,
`py_compile`) is still standard library, *exactly as v2* — a deliberate signal that
LangGraph replaces the orchestration, not the underlying tools or sandboxing. The notable
imports name the seams the rest of the notebook uses: `StateGraph/START/END/MessagesState`
(the graph primitives), `ToolNode/tools_condition/create_react_agent` (the prebuilt loop
pieces), `InMemorySaver` (the checkpointer), `InMemoryStore` (the long-term store), and
`trim_messages` (used conceptually by the context hook).

### `_Fmt(logging.Formatter)`
Identical in spirit to v2's formatter: ANSI colour per level, logger name trimmed to its
child suffix. The root logger is renamed `agent3` (v2 used `agent2`) and fans out into
`agent3.llm`, `agent3.tool`, `agent3.graph`, `agent3.subagent`. **Why keep this verbatim:**
the two notebooks are meant to read the same; only the subsystem names shift (`ollama`→`llm`,
`loop`→`graph`) to match the LangGraph vocabulary. `propagate = False` again stops duplicate
root lines; verbosity is the single env var `AGENT_LOG_LEVEL`.

### Configuration cell
The load-bearing decisions, unchanged from v2 in intent:
- **`OLLAMA_HOST`** — defaults to `http://localhost:8080` (the user's local Ollama proxy).
- **`MODELS`** — the *role → model* map: `reasoning` (`qwen3:32b`) for hard thinking
  (architect, verifier, planner, adversary); `fast` (`qwen3:8b`) for high-volume routine
  work; `summarizer` for distillation. Each is independently overridable by env var. This
  two-tier split is the single most important cost/latency lever, carried over intact.
- **Sandbox paths** — `WORKSPACE` (`v3_workspace/`), `AGENT_CODE_DIR` (`agent_code/`),
  `DB_PATH` (the SQLite DAG). Created eagerly at import.
- **Limits** — `MAX_TOOL_OUTPUT`, `BASH_TIMEOUT_S`, `TEST_TIMEOUT_S`,
  `REQUEST_TIMEOUT_S=900` (the 32B model is slow on big contexts), `MAX_ITERATIONS`.
- **`BASH_BLOCKLIST`** — the same destructive-fragment denylist; still a *speed bump, not a
  sandbox*.
- **`_HAS_PYTEST`** — feature-detected once so the test runner can degrade to plain python.

### `llm(...)` — the model factory (the v3 chokepoint)
This is v3's analogue of v2's `chat_complete()`. Where v2 had one HTTP function, v3 has one
**factory** that returns a configured `ChatOllama` Runnable; the graphs then call
`.invoke` / `.batch` / `.bind_tools` / `.with_structured_output` on it.

- **`_client(model, reasoning, temperature, num_predict)`** — `@lru_cache`d so repeated
  calls with the same shape reuse one client (and its connection pool). *Choice:* caching on
  the parameter tuple is a free, dependency-light connection-reuse optimization.
- **`llm(role, *, reasoning=True, temperature=0.2, max_tokens=None)`** — resolves a *role*
  (`reasoning`/`fast`/`summarizer`) or a literal model name through `MODELS`, then returns
  the cached client. **The one knob that matters most here is `reasoning`:** qwen3 is a
  thinking model, and `langchain-ollama` surfaces the thinking channel separately —
  `reasoning=True` routes `<think>` into `msg.additional_kwargs["reasoning_content"]` and
  leaves `msg.content` clean (used for free-text calls); `reasoning=False` disables thinking
  entirely (faster, used for JSON/structured calls). **This is the v3 equivalent of v2's
  "JSON mode suppresses `<think>`" trick** — same effect, expressed as a model flag rather
  than an API format field.

### `ollama_healthcheck()`
A *tags-only* probe (`GET /api/tags`, no generation): server up, and is each role's model
present (exact or family-prefix match)? Called as a fail-fast gate before live runs, exactly
as v2 — far better than a cryptic mid-run 404.

---

<a name="phase-05"></a>
## Phase 0.5 — Observability

v2 wrapped *every* model and tool call in a bespoke `rich` tracer. In LangGraph the
idiomatic seam is a **`BaseCallbackHandler`**: LangChain itself calls your hooks on every
model/tool start and stop, *no matter how deeply the graph nests*. So instrumentation moves
from "wrap each call site" to "register one handler and pass it in the run config."

### `_clip(s, n)` / `thinking_of(msg)`
`_clip` truncates with a `[+N chars]` marker (same readability helper as v2). `thinking_of`
is the v3-specific accessor that pulls qwen3's reasoning channel out of
`additional_kwargs["reasoning_content"]` (falling back to `"thinking"`). Every place that
needs the model's `<think>` channel goes through this one function.

### `class RichTracer(BaseCallbackHandler)`
The live, streaming view — but now **driven by LangChain**, not by manual wrapping. Because
LangChain fires the hooks, the *same* tracer works identically for a bare `llm.invoke()`, a
`ToolNode`, a `create_react_agent`, and the deep five-subagent graph — that uniformity is the
whole reason to use the callback seam.

- **`__init__`** — a `Lock` plus thread-safe counters (`calls`, `tokens`, `tool_calls`) and a
  `_starts` map keyed by `run_id` for per-call timing.
- **`on` / `full`** — read `AGENT_TRACE` (`full`/`compact`/`off`); `off` makes every hook a
  no-op so production runs pay nothing.
- **`on_chat_model_start`** — increments the call counter, records the start time, and (in
  full mode) panels the *prompt tail* (the latest message) rather than re-dumping the whole
  transcript every turn — the same "don't re-print the world" choice as v2.
- **`on_llm_end`** — splits the response into `<think>` (via `thinking_of`) and the answer,
  accumulates output tokens under the lock, and lists any requested tool calls. Panels the
  thinking in grey, the answer in green with its latency and token count.
- **`on_tool_start` / `on_tool_end`** — panel the args and the result; results are
  heuristically reddened if they look like an error (`startswith error/traceback/reverted…`),
  matching v2's colour convention.
- **`event(title, body)`** — a generic decision marker the cognitive primitives call directly
  (routing choices, verifier scores, adversary findings, plan shapes). This is the one place
  the notebook still narrates *imperatively* into the tracer, because those are decisions, not
  model/tool calls LangChain would hook.
- **`summary()`** — prints the `calls / tokens / tool_calls` ledger at the end.

Like v2, it **degrades gracefully**: with `rich` you get panels; without it the same info
goes through the `agent3` logger.

### `CB` and `run_config(label, **extra)`
The two ways to attach the tracer to a run. `CB = {"callbacks": [tracer]}` is passed as
`config=CB` to plain `.invoke`/`.batch` calls. `run_config(label)` is the richer version for
graphs: it bundles the callbacks **and** a unique `configurable.thread_id` — which is what
the checkpointer keys persistence on. *Choice:* every live graph run gets its own thread id,
so its checkpoints are isolated and independently resumable/inspectable.

### `show_graph(app)` / `stream_run(app, inputs, …)` — the LangGraph-native views
Two things v2 simply could not do:
- **`show_graph`** renders any compiled graph as a Mermaid PNG (ASCII fallback). The
  topology becomes a *picture* — used throughout to show the loop, the team, the
  self-correcting cycle.
- **`stream_run`** runs a graph with `app.stream(...)` and prints each **node's** update as it
  arrives, tagging messages that carry a thinking channel. This is the post-`master_loop`
  replacement for v2's per-iteration logging: you watch the run cross the graph node-by-node,
  live.

**Why this matters:** v2 had one observability mechanism (the tracer). v3 has three feeding
off the framework — callback narration, graph streaming, and graph diagrams — and all three
come from instrumenting *once* (the handler) or for *free* (stream/draw are built in).

---

<a name="phase-1"></a>
## Phase 1 — Cognitive substrate

The "how the model thinks" layer, re-expressed with LangChain idioms: thinking via
`reasoning=True`, structured routing via `with_structured_output`, parallel sampling via
`.batch()`.

### `STRONG_SYSTEM_PROMPT` + the tolerant parsers
`STRONG_SYSTEM_PROMPT` encodes the agent's epistemics as five rules of engagement: never
claim behaviour without a runnable artifact; defer to execution; a failing test/linter is
correct until proven otherwise; say "I don't know" rather than guess; the spec is the source
of truth. Identical to v2 — this is the "thoughtful response" idea baked into a constant.

Even though qwen3 *usually* puts thinking in `additional_kwargs` now, some paths still emit
literal `<think>…</think>` in the content, so the tolerant parsers stay:
- **`strip_think(text)`** — regex-removes a literal `<think>` block.
- **`split_think(msg)`** — returns `(thinking, answer)`. Crucially it's **dual-mode**: given a
  `BaseMessage` it prefers the `additional_kwargs` channel (`thinking_of`) and only falls back
  to regex; given a raw string it regex-splits. This is what lets the rest of the code treat
  "thinking" uniformly whether it arrived structurally or inline.

(`strip_code_fences` is defined later, in Phase 4's `code_with_tests` cell, but serves the
same defensive role — pulling raw source out of a stray markdown fence.)

### `think_then_answer(query, …) -> ThoughtfulResponse`
The basic single-shot call. Builds a `[SystemMessage, HumanMessage]` pair, invokes the model
with `reasoning=True`, and uses `split_think` to separate channels. The `@dataclass
ThoughtfulResponse` carries `thinking`, `answer`, and `output_tokens` (read from
`msg.usage_metadata`) so callers can budget on real token use. This is v2's primitive,
re-pointed at `ChatOllama`.

### Structured routing — `with_structured_output` replaces hand-parsed JSON
This is the cleanest single win of the rebuild. v2 hand-wrote JSON prompts and then
*tolerantly parsed* the result (try `json.loads`, fall back to a regex `{…}` span). v3 hands
the framework a **Pydantic schema** and lets it constrain *and validate* the output:

- **`class Difficulty`** (`Literal["trivial"…"extreme"]`) and **`estimate_difficulty(query)`**
  — a cheap classifier on the fast model with `reasoning=False`, mapped through
  `THINKING_BUDGETS` to a `num_predict` budget. Budgets are scaled up vs the article because a
  thinking model spends tokens on `<think>` too.
- **`class ProblemKind`** (`type` + a one-sentence `reason`) and **`classify_problem(query)`**
  — returns `convergent / divergent / exploratory / structural`, mapped through
  `TYPE_STRATEGY` to a strategy. The system prompt spells out what each type means.

Both classifiers use `temperature=0.0` + `reasoning=False` for cheap determinism, and both
keep a `try/except` returning a safe default (`medium` / `convergent`) — because even with
schema enforcement a backend hiccup shouldn't halt a run. **The honest framing from v2
survives:** for spec-driven *coding* almost everything is convergent/structural, so the type
axis is near-constant here, but it's deliberately kept as the bridge to the non-coding domains
this notebook leads toward.

### Test-time compute — `.batch()` replaces `ThreadPoolExecutor`
The two-axis routing engine, with parallel sampling now expressed as `llm.batch([...])`
instead of a manual thread pool (LangChain runs the batch concurrently for you):

- **`self_consistency(query, k=3)`** — builds `k` identical message lists, `.batch()`es them
  at temp 0.7 for diversity, buckets the answers by their first 60 lowercased chars, and
  returns the majority bucket with an agreement ratio. The 60-char prefix is the same cheap,
  embedding-free clustering trick as v2; agreement doubles as a confidence signal.
- **`class Verdict` + `verifier_score(question, candidate)`** — a structured 1–10 score from
  the **reasoning** model (`with_structured_output(Verdict)`, temp 0, `reasoning=False`).
  Scores facts/correctness, not style.
- **`class Ranking` + `asymmetric_solve(query, n=3)`** — the **verifier-asymmetry** pattern:
  generate `n` cheap candidates on the *fast* model via `.batch()`, then spend **one**
  structured call on the *reasoning* model to pick the best index. *Why it's the key cost
  trick:* generation is expensive per-token and parallelizable; *judging* is where the strong
  model's quality matters most, and you pay for it once. Falls back to candidate #0 on a parse
  failure.
- **`adaptive_think(query, route=True)`** — the dispatcher tying both axes together: estimate
  difficulty → budget, classify type → strategy, run the chosen strategy. `route=False`
  collapses to a budget-only single pass, useful for the "does routing actually help?"
  ablation. Each branch emits a `tracer.event` so the routing decision is visible in the
  narrative.

---

<a name="phase-2"></a>
## Phase 2 — Tools

Everything the agent can *do*. v2 registered tools as a `name→callable` dict and hand-wrote
their JSON schemas. **In v3 each tool is a `@tool`-decorated function, and LangChain derives
the JSON schema from the type hints + docstring.** The *bodies* are the same sandboxed v2
tools; only the registration changes. Two design rules still dominate: **paths are sandboxed
to `WORKSPACE`** and **outputs are truncated**.

### Path & output safety
- **`_safe_path(path)`** — resolves and `relative_to(WORKSPACE)`-checks, raising if it escapes.
  The containment boundary for every file op. (Resolves *after* joining, so symlink games are
  the residual risk — acceptable for a local single-user tool.)
- **`_truncate(s, limit)`** — clip with a `[truncated N chars]` marker.

### File/shell tools (all `@tool`)
- **`read_file(path, start_line, end_line)`** — reads with 1-indexed line numbers prefixed,
  optional range, `errors="replace"` so binary junk doesn't crash.
- **`write_file(path, content)`** — **snapshots prior content into `SNAPSHOTS` before
  writing**, enabling `revert_file`. Reports created-vs-updated.
- **`revert_file(path)`** — pops the in-memory snapshot and restores (or deletes if the file
  was new). The same **in-memory undo stack** as v2 — chosen over git because the workspace
  already lives inside a repo and the notebook refuses to nest one.
- **`grep(pattern, path, recursive)`** — shells out to real `grep -rn`, clamped to `WORKSPACE`.
- **`glob_files(pattern)`** — glob scoped to `WORKSPACE` via `is_relative_to`, capped at 200
  hits.
- **`bash(command)`** — runs in `WORKSPACE` with a timeout after scanning `BASH_BLOCKLIST`.
  **Same honest caveat:** the blocklist is a speed bump, not a sandbox — `shell=True` means a
  determined model could evade it; acceptable because the workspace is throwaway and local.

### Coding-specific tools (the quality gates)
- **`lint_python(code) -> {passed, errors}`** — a plain (non-`@tool`) helper reused internally:
  writes to a temp file, `py_compile`s it (syntax), then walks the AST to flag bare `except:`.
  Deliberately minimal — a fast, dependency-free *must-pass* filter, not a full linter.
- **`_run_tests(test_code, timeout)`** — a plain helper: writes a test module and runs it with
  **pytest if available, else plain python**, then regex-parses `N passed` / `N failed` into a
  structured dict, inferring pass/fail from the return code when counts are absent. *Choice:*
  kept as a plain function (not a `@tool`) because both the `run_tests` tool **and** the spec
  layer call it directly.
- **`write_code(filename, content)` (`@tool`)** — the **lint-gated write**: rejects unless the
  filename is a bare `*.py` and the content lints clean; only then does it persist to
  `agent_code/`. This is the central reliability mechanism — **broken code never reaches
  disk**, so downstream test runs never fail for trivial syntax reasons.
- **`run_python(code)` (`@tool`)** — writes the snippet with `agent_code/` prepended to
  `sys.path`, runs it as a subprocess, returns exit code + output. A fresh subprocess per run
  gives real isolation and a hard timeout — the no-Docker stand-in for the article's sandbox.
- **`run_tests(test_code)` (`@tool`)** — the `@tool` wrapper over `_run_tests`, returning a
  one-line pass/fail summary plus output for the model to read.
- **`TOOLS_BASE` / `TOOLS_BY_NAME`** — the base toolset every coding agent gets, and a
  name→tool map. This list is what gets handed to every `ToolNode` and every subagent — it is,
  in effect, v3's tool registry (see the Phase 7 note on "MCP").

---

<a name="phase-3"></a>
## Phase 3 — The tool loop, as a graph

v2's `master_loop()` was a hand-written *perception → action → observation* while-loop. **In
LangGraph that loop *is* the graph.**

### `build_agent_graph(tools, system, role, reasoning, checkpointer)`
The v3 replacement for `master_loop`. A two-node `StateGraph` over the built-in
`MessagesState`:

```
START → agent → (tools_condition) → tools → agent → … → END
```

- **`agent_node`** — prepends `STRONG_SYSTEM_PROMPT` at call time if the first message isn't
  already a system message, then invokes the tool-bound model. *Choice — prepend, don't
  store:* the system prompt always leads without being persisted repeatedly into state.
- **`ToolNode(tools)`** — the prebuilt node that executes whatever tools the model requested,
  appending their results as `ToolMessage`s. This replaces v2's hand-written `_run_tool_call`
  + dispatch table entirely.
- **`tools_condition`** — the prebuilt conditional edge: if the last AI message has tool
  calls, route to `tools`; otherwise end. This is v2's "did the model request tools?" check,
  now a library function.
- **`.bind_tools(tools)`** — attaches the tool schemas to the model so it can emit tool calls.
  Replaces v2's manual `_fn()` schema builder.
- **Compiled with a checkpointer** (`InMemorySaver` by default) — so every run is a
  **resumable thread** keyed by `run_config`'s `thread_id`. Durable state and time-travel come
  *for free* from compiling the graph; v2 had nothing equivalent for the loop itself.

`coding_agent` is the module-level lead agent: the full base toolset + the strong system
prompt on the fast model. Cells 20–21 visualise it (`show_graph`) and run it on a tiny
`hello.txt` round-trip (`stream_run`), so you see the `agent ⇄ tools` cycle live.

### Subagent discipline = the agent-as-tool pattern
v2's `spawn_subagent` was a recursive call into `master_loop`. v3 keeps the same *discipline*
but packages it the LangGraph way:

- **`SUBAGENT_SYSTEM`** — the same focused prompt: one subtask, no clarifying questions (make
  a reasonable assumption), and *"your final message is the ONLY thing the parent sees — make
  it self-contained."*
- **`spawn_subagent(prompt, tools, system, role)`** — builds a fresh `build_agent_graph`,
  invokes it on the prompt with a unique thread id and a generous `recursion_limit`, and
  returns the **last non-empty AI message** (think-stripped). This enforces the
  **context-isolation** property: the parent never sees the subagent's tool transcript, only
  its distilled summary.
- **`make_subagent_tool(name, description, system, …)`** — wraps such a sub-graph as a
  `StructuredTool`. **This is the agent-as-tool pattern:** a parent agent can now *delegate* by
  calling the subagent like any other tool, and LangGraph runs the entire sub-graph inside that
  one tool call. The subagent's internal steps are physically incapable of leaking into the
  parent's context, because they happen one `Runnable` layer down.

*Why this is cleaner than v2:* in v2, context isolation was a convention you had to respect
(return only the last message). In v3 it's structural — the sub-graph is a black box exposing
a single string return.

---

<a name="phase-4"></a>
## Phase 4 — The hardening stack, as small graphs

v2's four hardening primitives were plain functions with internal loops. **In v3 each becomes
a tiny graph — which is exactly where LangGraph earns its keep: loops and branches are
edges**, inspectable and drawable.

### `architect_editor_solve(task)` — a linear two-node chain
Separation of deliberation from transcription. `_AEState` (`task`, `plan`, `output`) flows
through two nodes:
- **`_architect_node`** — the **reasoning** model with `with_structured_output(ArchitectPlan)`
  produces a *structured* plan (a list of `Section`s with intents and constraints) but
  **explicitly not code**. The Pydantic schema (`Section`, `ArchitectPlan`) is what guarantees
  the architect's constraints travel verbatim; a failure degrades to an empty plan.
- **`_editor_node`** — the **fast** model with **`reasoning=False`** executes that plan into
  the final artifact. *Why disable thinking here:* the architect already deliberated, so the
  editor just transcribes — and turning off `<think>` makes it dramatically faster *and* stops
  thinking from eating the token budget and truncating the generated code. This is v3's spelled
  form of v2's `/no_think` editor trick (a model flag instead of a prompt prefix).

The graph is `START → architect → editor → END`, compiled to `architect_editor_app`;
`architect_editor_solve` is the thin function wrapper.

### `self_refine(query, iterations=2)` — generate → critique → refine, as a loop
`_RefineState` carries `current`, `critique`, `iteration`, `max_iter`, and full `history`.
Three nodes — `_gen_node`, `_critique_node`, `_refine_node` — wired
`generate → critique → refine`, with a **conditional edge** `_refine_route` that loops
`refine → critique` until the iteration budget is spent. *Choice carried from v2:* critique
and refine are **separate calls** (find flaws, then fix them) because separating the two
produces sharper critiques. What's new is that the loop is now a *visible cycle in the graph*,
not a hidden `for`.

### `code_with_tests(task, test_code, max_rounds=3)` — generate → verify, as a loop
The single most important pattern for code reliability, now a two-node graph with a feedback
cycle. `_CWTState` carries the task, the test code, the current code, the feedback, the round
counter, and status/history.
- **`_cwt_generate`** — generates code (stripping any markdown fence via `strip_code_fences`),
  appending the previous failure as `PREVIOUS ATTEMPT FAILED:` when present.
- **`_cwt_verify`** — **lint-gates first** (short-circuits without wasting a test run), then
  writes a candidate and runs the **real** `_run_tests`. Records pass/fail into history.
- **`_cwt_route`** — the conditional edge: end on `passed` or when `round >= max_rounds`, else
  loop back to `generate`.

*The defining choice, preserved verbatim from v2:* the feedback fed back is the **verbatim
test stdout**, never a paraphrase — the ground-truth error is the most useful possible signal
for the next attempt.

### `adversarial_probe(target, candidate, n_max=4)` — red-teaming, one structured call
The **reasoning** model plays "hostile adversary" with
`with_structured_output(AttackList)` and returns a typed list of `Attack`s (category,
scenario, why_it_breaks, severity). Higher temperature (0.4) to encourage creative attacks;
purely *advisory* (it surfaces risks for the reviewer rather than gating, because a generated
attack might be a false alarm). The Pydantic schema is what makes the attacks structured and
iterable instead of prose to re-parse.

Cells 27–28 visualise the self-refine loop and run `code_with_tests` for real on a trivial
`inc(n)=n+1` task that loops on failure.

---

<a name="phase-5"></a>
## Phase 5 — Planning & durable state

Same durable substrate as v2; the only change is that `make_plan` now returns a *validated
Pydantic object*. LangGraph then adds two persistence layers on top (checkpointer + Store),
wired in Phases 6–7.

### `make_plan(goal)` → a validated `Plan`
`PlanStep` (`step_id`, `description`, `depends_on`, `expected_artifact`) and `Plan` (`goal` +
`steps`) are Pydantic models. `make_plan` asks the **reasoning** model (temp 0) with
`with_structured_output(Plan)` for a dependency-ordered plan and gets back a *typed, validated*
object — no hand-parsing, no `None`. A failure degrades to an empty `Plan` so callers never
crash.

### `class TaskDAG` — durable, dependency-aware work tracking
Carried over **unchanged** from v2. A **SQLite-backed** DAG (`node_id, title, status,
attempts, depends_on`):
- `add_node` uses `INSERT OR REPLACE` (idempotent re-seeding).
- `ready_nodes()` returns pending nodes whose deps are all `done` — the scheduler primitive.
- `set_status` also increments `attempts`, a free retry counter.

`isolation_level=None` (autocommit) persists each update immediately, so the DAG survives a
kernel restart. *Note:* in v3 the team's control flow is the **graph topology** (Phase 7), so
`TaskDAG` is retained mainly as a durable record / the carried-over substrate rather than the
live scheduler it was in v2.

### `class BiTemporalMemory` — facts with validity intervals
Also unchanged. Each fact has `valid_from`/`valid_to`; superseded facts are **invalidated, not
deleted** (so you can always ask "what did the agent believe *then*"). `query_valid(kind)`
returns currently-valid facts; `recall(query, k)` does **keyword-overlap** ranking — no
embeddings, no ChromaDB. The same deliberate trade of recall quality for zero dependencies and
full transparency, fine because the corpus is one run's worth of facts.

### The spec layer — definition-of-done as executable contract
Unchanged from v2 and still the linchpin of the notebook's epistemics:
- `write_definition_of_done(criteria, import_line)` persists the contract to
  `DEFINITION_OF_DONE.json`.
- `compile_test_suite(criteria, import_line)` **codegens a real pytest module** — each
  `{"name", "check"}` becomes a `def test_name(): assert <check>`.
- `spec_verify(contract)` compiles the suite and runs it via `_run_tests`.

"Done" is **not** prose the model self-grades against — it is *compiled into tests that
execute against the agent's code*. The suite is green or the work isn't done, full stop.

---

<a name="phase-6"></a>
## Phase 6 — Context engineering: a `pre_model_hook`

The insight from v2 still holds: in a coding tool loop the context grows by accumulated **tool
observations** (file dumps, test logs), not by user turns. v2 trimmed *inside* its loop.
LangGraph gives the loop a dedicated seam — a **`pre_model_hook`** that runs immediately before
every model call, receives the full state, and returns what the model should actually see.

### `make_context_hook(max_recent=6, memory=None)`
Returns a `hook(state)` that implements **trim → reinject** *non-destructively*:
- Keeps `head` (the system/first message), `anchor` (the original human task — never trimmed,
  because losing the task is catastrophic), and the last `max_recent` messages verbatim.
- Everything older is **`dropped`** and replaced by a single `[context note]` system message
  recording how many steps were elided.
- If a `BiTemporalMemory` is passed, it `recall`s facts relevant to the anchor task and injects
  them inside a `<durable_memory>` block — the *reinject* half of the pattern.
- **It returns `{"llm_input_messages": …}`, not `{"messages": …}`.** This is the crucial
  LangGraph mechanic: `llm_input_messages` changes only *what the model sees this turn*; the
  full transcript stays untouched in state and in the checkpointer. **v3 gets non-destructive
  trimming for free** — v2's manual trim actually dropped messages from the working list.

### `build_managed_agent(...)`
`create_react_agent(model, tools, prompt=…, pre_model_hook=make_context_hook(…),
checkpointer=…)`. `create_react_agent` is LangGraph's **prebuilt** version of
`build_agent_graph`; it accepts a `pre_model_hook` directly, so wiring in a bounded context
window is a one-liner. `managed_agent` is the module-level instance; cell 35 draws it.

*Why this is the cleanest section of the rebuild:* v2 needed a whole `ContextManager` class
(split / trim / distill / render_block / consolidate) plus a `managed_loop` to thread it
through. v3 expresses the same *trim + reinject* behaviour as a single hook function on a
prebuilt agent. (The heavier v2 machinery — LLM-based distillation and the writer+critic
consolidation guard — is **not** reproduced here; v3's hook does the deterministic trim +
keyword recall, trading the distillation step for framework simplicity. If you need the
distill/consolidate layer, that's the one place v3 is *lighter* than v2 by design.)

---

<a name="phase-7"></a>
## Phase 7 — The five-subagent team, as one graph

v2 hand-routed five subagents through a `TaskDAG` scheduler (`agent_run` pulling
`ready_nodes()`). **In v3 the DAG *is* the graph topology** and each subagent is a node; the
control flow v2 expressed imperatively is now *declared as edges*.

### `class TeamState`
One shared `TypedDict` that flows through every node: `task`, `target_filename`, `contract`,
`plan`, `test_result`, `review`, `report`, `attempts`, `max_attempts`, `notes`. *Choice:* a
single shared state (vs v2's `CodingAgent` container passed between subagents) is the LangGraph
way — nodes read and write slices of it, and the framework merges their returns.

### The five nodes — each reuses a Phase-4/5 primitive
- **`planner_node`** — runs `make_plan` and stores the steps into state. Produces the roadmap.
- **`implementer_node`** — the workhorse. Drafts the target file with **architect/editor**
  (`architect_editor_solve`), strips fences, and persists via the **lint-gated** `write_code`.
  On a retry it feeds the *verbatim* prior test failure back into the task — the same
  external-feedback discipline as `code_with_tests`, here split across the graph's retry loop.
- **`tester_node`** — runs `spec_verify` **independently** of the implementer. The clean
  separation between "the builder thinks it passes" and "an independent step confirms it
  passes."
- **`reviewer_node`** — reads the file, scores it with `verifier_score`, red-teams it with
  `adversarial_probe`, stores the review. Advisory, not gating.
- **`report_node`** — uses `self_refine` to write a concise `REPORT.md` grounded in the run's
  `notes` (so the report describes what *actually happened*, not a hallucinated narrative).

### `tester_route` — the self-correcting loop
The one **conditional edge** that makes the team more than a pipeline: if the tester passed,
go to `reviewer`; if it failed *and* attempts remain, go **back to `implementer`** (carrying
the failure); if the budget is spent, give up and review anyway. This is v2's
implementer↔tester retry, now a visible cycle in the graph:

```
START → planner → implementer → tester ─(pass)→ reviewer → report_writer → END
                       ▲                 │
                       └──(fail, < max)──┘
```

### `build_team_graph()` / `team_app` / `run_team(...)`
`build_team_graph` wires the nodes and edges and compiles with a checkpointer; `team_app` is
the module-level instance. `run_team(task, target_filename, contract, …)` seeds the initial
state and either streams the run (`stream_run`) or invokes it, with a raised `recursion_limit`
to accommodate the retry loop.

**On the "MCP-style registry":** v2 built an explicit `MCPTool`/`mcp_registry` to demonstrate
the pattern. v3 notes (in the Phase 7 markdown) that **the registry already exists** — the
typed `@tool` set / `ToolNode` from Phase 2 *is* a uniform registry with names, descriptions,
and JSON schemas, which is exactly what MCP provides. So v3 doesn't reimplement it; the
framework's tool abstraction subsumes it.

*Why a graph instead of a hand-routed DAG:* the dependency graph is now *data the framework
executes and can draw*, not imperative scheduling code. Re-wiring the pipeline means editing
edges, and `show_graph(team_app)` (cell 38) renders the whole team — including the
self-correcting loop — as a diagram.

---

<a name="phase-8"></a>
## Phase 8 — Running the team

The driver runs the team on a deliberately simple, fully-deterministic task (FizzBuzz — *not*
dengue, to prove the engine is general):

1. **`ollama_healthcheck()`** (tags-only, fail-fast).
2. Define `TASK_8` + five `CONTRACT_8` criteria and persist the **definition of done** via
   `write_definition_of_done`.
3. **`run_team(TASK_8, "solution.py", CONTRACT_8, max_attempts=2)`** — streamed, so you watch
   `planner → implementer → tester` loop back on a red test, then `reviewer → report_writer`.
4. Print attempts / test result / review, read back `REPORT.md`, and dump `tracer.summary()`.

---

<a name="phase-9"></a>
## Phase 9 — v3 vs v2: what LangGraph buys you

The notebook's own comparison table, plus a structural **census** (`_census`, cell 44) that
introspects each compiled graph's nodes and edges with **no model calls**. The mapping, in
one place:

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

The census also confirms the base toolset (the de-facto "MCP registry") and that `team_app`
carries a checkpointer.

---

<a name="phase-10"></a>
## Phase 10 — Offline self-tests

A tiny `check`/`section` harness exercises everything that needs **no model calls** — the
tools (round-trip write/read, bash echo, blocklist, lint-gated `write_code`, path-escape
block), the parsers (`strip_think`, `split_think`, `strip_code_fences`), `lint_python` /
`_run_tests`, the `TaskDAG` dependency gating, `BiTemporalMemory` recall + invalidation,
`spec_verify`, the Pydantic schemas, and that **every graph compiles with the expected
topology** (`coding_agent` has a `tools` node, the team has its five worker nodes,
`managed_agent` has ≥3 nodes). Cell 47 rolls the results up and `assert`s no failures, so the
plumbing is testable in seconds without burning GPU time — the same "the plumbing is testable
offline" discipline as v2.

---

<a name="phase-11"></a>
## Phase 11 — A harder end-to-end build

A harder task (`BoundedCounter`, a small LRU-ish data structure with `add`/`top`/`keys` and
capacity eviction) driven through the **same** Phase-7 team graph, to show the engine scales
past toy tasks. The criteria are written as one-line lambda checks so the contract stays a
pure data structure. After `run_team(..., max_attempts=3)`:
- **Independent re-verification** (cell 51): we recompile the contract and run `spec_verify`
  **ourselves**, not trusting the team's own word — the no-claim-without-evidence ethos applied
  to the *whole run*.
- Inspect `counter.py` and `REPORT.md`.
- Replay the run's per-node `notes` (cell 52) — and, because the team compiled with a
  checkpointer, the full per-node state history of the thread is available to inspect.

---

<a name="themes"></a>
## Cross-cutting design themes

The real lessons, mostly inherited from v2 and re-grounded in the framework:

1. **One model-construction chokepoint (`llm`).** Backend portability, uniform tracing, and a
   single place to retune `reasoning`/temperature/budget. The graphs operate on a `Runnable`,
   so swapping backends never touches them.

2. **Two-tier model economics.** Cheap model for high-volume generation and routine work;
   expensive model reserved for *judging/architecting*. `asymmetric_solve` is the purest
   expression — parallel cheap generation via `.batch()`, one strong ranking call.

3. **No claim without a runnable artifact.** Lint gate → real test execution → *independent*
   re-verification. The definition of done is *compiled into tests*, not self-graded; test
   failures feed back **verbatim**.

4. **Structure replaces tolerance where the framework allows it.** v2 hand-parsed JSON
   tolerantly everywhere; v3 uses `with_structured_output(PydanticModel)` so the framework
   constrains *and validates*. The tolerant parsers (`strip_think`, `strip_code_fences`)
   survive only for the free-text paths that genuinely need them.

5. **Loops and branches are edges.** Every place v2 had a hidden `for`/`while`/`if` (the tool
   loop, self-refine, code-with-tests, the team's retry), v3 has an explicit, drawable,
   inspectable graph edge. The control flow is *data the framework executes*.

6. **Context as a managed resource — non-destructively.** The `pre_model_hook` returns
   `llm_input_messages`, bounding the model's *view* while the checkpointer keeps the full
   history. Trimming no longer means forgetting.

7. **Persistence and observability come from compiling the graph.** A checkpointer makes every
   run a resumable, inspectable thread; `graph.stream` and `draw_mermaid_png` give live and
   visual views — all for free, none of it hand-built.

8. **Subagents are structurally isolated.** Agent-as-tool means a subagent's transcript
   *cannot* leak into the parent — context isolation is enforced by the architecture, not by
   convention.

9. **Honest about where it simplifies.** Like v2's caveats (no Docker, no ChromaDB, no git,
   near-constant type axis, `bash` is a speed bump), v3 openly drops v2's LLM-based context
   *distillation* and the writer+critic *consolidation* guard in favour of a deterministic
   trim + keyword recall in the hook. The intent is a *teaching* engine you can read end-to-end
   against its v2 sibling — not a hardened product.

---

<a name="known-issue"></a>
## Notes on changes since first draft

An earlier draft of this walkthrough flagged a corrupted line in **cell 7**
(`RichTracer.on_llm_end`) where a stray URL had been pasted into an identifier
(`gen = responshttp://…e.generations[0][0]`), which was a `SyntaxError`. **That has since been
fixed in the notebook** — cell 7 now reads the correct `gen = response.generations[0][0]`, so
the tracer's per-response panel (thinking / answer / token accounting) works as intended. No
outstanding correctness issues are known in the current notebook; the 25 offline self-tests in
Phase 10 (cell 46) are written to pass without the backend and gate the live phases.
