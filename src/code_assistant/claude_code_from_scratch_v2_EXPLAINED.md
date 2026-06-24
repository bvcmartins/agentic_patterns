# `claude_code_from_scratch_v2` — Code Walkthrough & Design Rationale

A detailed reference for every class and function in
`claude_code_from_scratch_v2.ipynb`, with the *why* behind each implementation
choice.

The notebook lifts the **reusable reliability engine** from the article *"Building
Claude from Scratch: 62 Components…"* (which wired all its patterns into a single
dengue-paper reproduction) and re-points it at **general coding tasks**, running on
**local Ollama / qwen3** so it is directly comparable to the flat `v1` harness in the
same folder.

Two design commitments shape everything below:

- **One model-call chokepoint.** Every LLM call goes through `chat_complete()`. Swap
  that one function + the `MODELS` map and the whole notebook retargets to DeepSeek or
  Gemini. The rest is backend-agnostic.
- **Nothing is claimed without a runnable artifact.** Code is gated by a linter, then
  by real test execution, then by an independent re-verification. The agent never
  "believes" its own output.

---

## Table of contents

- [Phase 0 — Imports, logging, configuration](#phase-0)
- [Phase 0.5 — Observability (`Tracer`, `_TraceRecorder`)](#phase-05)
- [Phase 1 — Cognitive substrate (thinking, routing, voting, verifiers)](#phase-1)
- [Phase 2 — Tools (file/shell, lint-gated writes, runners)](#phase-2)
- [Phase 3 — The tool loop & subagents](#phase-3)
- [Phase 4 — The hardening stack](#phase-4)
- [Phase 5 — Planning & durable state (Plan, DAG, memory, spec)](#phase-5)
- [Phase 6 — Context engineering (`ContextManager`)](#phase-6)
- [Phase 7 — MCP registry & the five-subagent architecture](#phase-7)
- [Phase 8 — How it all runs](#phase-8)
- [Cross-cutting design themes](#themes)

---

<a name="phase-0"></a>
## Phase 0 — Imports, logging, configuration

### `_Fmt(logging.Formatter)`
A custom log formatter that adds ANSI colour per level and trims the logger name to
its child suffix (`agent2.ollama` → `ollama`).

**Why it's built this way.** The notebook deliberately mirrors `v1`'s logging style so
the two notebooks read identically side-by-side. A single root logger `agent2` fans out
into per-subsystem children — `ollama`, `loop`, `tool`, `subagent`, `dag`, `chat`, `ctx`
— so you can reason about *which layer* emitted a line. `log.propagate = False` stops
duplicate lines from the root handler. Verbosity is one env var (`AGENT_LOG_LEVEL`),
which is the recurring config idiom: **every knob is an environment variable with a
sane default.**

### Configuration cell (no functions, but the load-bearing decisions live here)
- **`OLLAMA_HOST`** — defaults to `http://localhost:8080` (the user's local Ollama proxy port).
- **`MODELS`** — a *role → model* map, not a single model. `reasoning` (`qwen3:32b`) is
  reserved for the hard thinking steps (architect, verifier, planner, adversary);
  `fast` (`qwen3:8b`) does the high-volume routine work (editor, subagents,
  classifiers); `summarizer` does distillation. This **two-tier split mirrors v1's
  lead/subagent split** and is the single most important cost/latency lever in the
  notebook — the expensive model is spent only where asymmetry pays off (see verifier
  asymmetry below).
- **Sandbox paths** — `WORKSPACE` (where tests run), `AGENT_CODE_DIR` (`agent_code/`,
  where the agent's modules land), and `DB_PATH` (the SQLite DAG). Created eagerly at
  import so later code is declarative.
- **Limits** — `MAX_TOOL_OUTPUT` (truncate giant tool results), `BASH_TIMEOUT_S`,
  `TEST_TIMEOUT_S`, `REQUEST_TIMEOUT_S=900` (the 32B model is *slow* on big contexts, so
  the HTTP timeout is generous), `MAX_ITERATIONS`.
- **`BASH_BLOCKLIST`** — a denylist of obviously destructive command fragments
  (`rm -rf /`, `sudo`, fork bombs…). This is a *speed bump, not a sandbox* (see the
  honest caveat under `tool_bash`).
- **`_HAS_PYTEST`** — feature-detected once so the test runner can degrade to plain
  `python` if pytest isn't installed.

---

<a name="phase-05"></a>
## Phase 0.5 — Observability

The loops only emit one-line metadata; to *understand* how the agent solves a problem
you need to see, for each call, the `<think>` channel, the answer, the exact tool args
and results, and how all of that nests across subagents and DAG nodes. Two
complementary views provide this. **Both only observe — they change no behaviour.**

### `_clip(s, n)`
Truncate a string to `n` chars (default `TRACE_PREVIEW`) with a `[+N chars elided]`
marker. Used everywhere to keep panels readable.

### `class Tracer` — the live, streaming view
A **depth-aware, thread-safe** tracer. Every model call, tool call, and subagent runs
"inside" it, so a run reads top-to-bottom as a nested narrative.

- **`__init__`** — sets up a `threading.local()` for depth and a `Lock` for the shared
  counters (`calls`, `tokens`, `by_label`, `tool_counts`).
- **`_depth` (property + setter)** — depth is **thread-local**. *Why:* the parallel
  self-consistency / best-of-N samples run on a `ThreadPoolExecutor`; without
  per-thread depth their indentation would tangle into each other.
- **`on` / `full`** — read `AGENT_TRACE` (`full`/`compact`/`off`). `off` makes the
  whole tracer a no-op so production runs pay nothing.
- **`_print` / `_line`** — low-level emit; `_print` indents by `depth*3` via rich
  `Padding`, falling back to space-indented `log.info` when rich isn't installed.
- **`span(kind, title, meta)`** — the **context manager that creates nesting**. Emits a
  styled rule header, bumps depth on enter, restores it and prints an elapsed-time
  footer on exit. `kind` (`agent`/`subagent`/`dag`/`iter`/`strategy`) selects colour +
  icon from `_KIND_STYLE`.
- **`model_request` / `model_response`** — fed by `chat_complete`. The request panel
  shows the system prompt *only on kickoff* (≤2 messages) plus the latest message — a
  deliberate choice to avoid re-dumping the whole transcript every turn. The response
  panel splits `<think>` from the answer and lists any requested tool calls; it also
  accumulates the per-label tally under the lock.
- **`tool` / `tool_result`** — panel the args and the result; results are heuristically
  coloured red if they look like an error (`startswith error/traceback/reverted…`).
- **`event(title, body)`** — a generic decision marker for routing choices, verifier
  scores, adversarial attacks, plan shapes — anything that isn't a model/tool call but
  matters to the narrative.
- **`summary()`** — prints the per-label `calls / tokens / time` table at the end.
  This is the cost ledger you read after a run.

**Design choice:** the tracer degrades gracefully. With `rich` you get panelled,
colour-nested output; without it the *same* information goes through the `agent2`
logger as depth-indented lines. No hard dependency.

### `class _TraceRecorder` + `render_trace()` — the navigable tree view
Where `Tracer` *streams* the run live, `_TraceRecorder` *records* it into a tree and
`render_trace()` renders that tree as a **LangSmith-style, collapsible HTML view** with
a timing waterfall and click-to-expand inputs/outputs.

- The recorder mirrors the tracer's API (`open_span`/`close_span`, `model_start`/
  `model_end`, `tool_start`/`tool_finish`, `event`) but appends nodes to a parent stack
  instead of printing.
- The module-level `_span`, `_model_request`, … wrappers **fan a single event out to
  both** the live tracer and the recorder, so instrumenting a call site once feeds both
  views.
- `render_trace`'s nested helpers (`bounds`, `agg`, `bar`, `render`, `detail`) compute
  each node's time span, aggregate child timings, and emit `<details>` blocks plus a
  proportional timing bar.

**Why two views?** They answer different questions. The live stream is for *watching* a
run unfold; the tree is for *post-hoc forensics* — "where did the time go, and what
exactly did the reviewer see?"

---

<a name="phase-1"></a>
## Phase 1 — Cognitive substrate

This is the "how the model thinks" layer: a single tolerant model wrapper, parsers for
qwen3's native thinking channel, and the compute-adaptive routing engine.

### `chat_complete(...)` — the one chokepoint
Hits Ollama `/api/chat` with `stream=False`, logs latency + token usage, hands the full
request/response to the tracer, and returns the raw `message` dict with the
completion-token count stashed under `eval_count`.

Key parameters and the reasoning behind them:
- **`json_mode`** → sets Ollama's `format: "json"`, which constrains output to valid
  JSON. A crucial qwen3-specific effect: **JSON mode also suppresses the `<think>`
  block.** So structured calls (classifiers, verifiers, planners) are clean JSON with
  no thinking to strip, while free-text calls keep `<think>` as the thinking channel.
- **`label`** — every call is tagged (`think`, `verify`, `architect`, `editor`,
  `distill`…). This is what the tracer's per-label cost table keys on; it's how you see
  *which pattern* burned the tokens.
- **`timeout`** — optional per-call override. Best-effort callers (context distillation,
  consolidation) pass a short timeout so a slow backend **fails fast to their fallback**
  instead of blocking on the 900s global timeout.
- **No retry, no streaming.** Errors bubble up via `raise_for_status()`. The philosophy
  is that retries/streaming belong in a backend adapter, not the core; keeping this
  function tiny is what makes backend-swapping a one-function change.

### Parsing helpers (the "tolerant" layer)
qwen3 emits `<think>…</think>` inline and sometimes wraps code in markdown fences, so
every consumer needs defensive parsing:
- **`strip_think(text)`** — regex-removes the `<think>` block, returns the visible answer.
- **`split_think(text)`** — returns `(thinking, answer)` so the tracer can panel them separately.
- **`parse_json(text)`** — strips think, tries `json.loads`, then falls back to the
  first `{…}` span via regex. This tolerance is essential because even in JSON mode a
  model can prepend stray text.
- **`strip_code_fences(text)`** — pulls raw source out of a ` ```python … ``` ` block if
  the model added one despite being told not to. Defensive belt-and-suspenders for the
  code-generation paths.

### `ollama_healthcheck()`
GETs `/api/tags`, lists available models, and checks each role in `MODELS` is present
(exact match or same family prefix). Called at import and again before each live run as
a fail-fast sanity gate — far better than a cryptic mid-run 404.

### `STRONG_SYSTEM_PROMPT` + `think_then_answer()`
`STRONG_SYSTEM_PROMPT` encodes the agent's *epistemics* as rules of engagement: never
claim behaviour without a runnable artifact; defer to execution; a failing test/linter
is correct until proven otherwise; say "I don't know" rather than guess; the spec is the
source of truth. This is the article's "thoughtful response" idea baked into a constant.

`think_then_answer(query, …) -> ThoughtfulResponse` is the basic single-shot call. In
the article it forced `<thinking>`/`<answer>` tags; **here qwen3 thinks natively**, so
the function just calls `chat_complete` and uses `split_think` to separate channels. The
`@dataclass ThoughtfulResponse` carries `thinking`, `answer`, and `output_tokens` so
callers can budget on actual token use.

### The two-axis routing engine
This is the heart of "compute-adaptive effort." The article routes on **two orthogonal
axes**, and v2 keeps both:

**Axis 1 — difficulty → token budget.** `estimate_difficulty(query)` is a cheap JSON
classifier on the fast model returning `trivial…extreme`, mapped through
`THINKING_BUDGETS` to a `num_predict` budget. Budgets are **scaled up vs the article**
because a thinking model spends tokens on `<think>` too, not just the answer.

**Axis 2 — problem *type* → solving strategy.** `classify_problem(query)` returns one of
`convergent / divergent / exploratory / structural`, mapped through `TYPE_STRATEGY` to a
strategy:
- `convergent` → `self_consistency` (one defensible answer; trust agreement)
- `divergent` → `asymmetric_solve` (many valid answers; let a verifier rank)
- `exploratory` → `wide_pass` (one higher-temp, higher-budget pass)
- `structural` → `decompose` (plan the parts first)

**Honest framing in the comments:** for spec-driven *coding* almost everything is
convergent/structural, so the type axis is *near-constant here* — but it's deliberately
kept because it becomes load-bearing for the non-coding domains (financial analysis,
knowledge management) this notebook is a stepping-stone toward. Both classifiers use
`temperature=0.0` + `json_mode` + tiny token caps because they should be deterministic
and cheap, and both have try/except fallbacks to a safe default (`medium` / `convergent`)
so a parse failure never halts a run.

### The four strategies
- **`self_consistency(query, k=3)`** — samples `k` answers **in parallel** (temp 0.7 for
  diversity) via `ThreadPoolExecutor`, buckets them by their first 60 lowercased chars,
  and returns the majority bucket with an agreement ratio. *Choice:* the 60-char prefix
  is a cheap, embedding-free way to cluster "the same answer"; good enough for short
  convergent answers, and the agreement ratio doubles as a confidence signal.
- **`verifier_score(question, candidate)`** — a structured 1–10 score from the
  **reasoning** model (JSON, temp 0). Scores facts/correctness, not style.
- **`asymmetric_solve(query, n=3)`** — the **verifier-asymmetry** pattern: generate `n`
  cheap candidates on the *fast* model, then spend **one** call on the *reasoning* model
  to rank them. *Why this is the key cost trick:* generation is expensive per-token and
  parallelizable; *judging* is the part where the strong model's quality matters most,
  and you only pay for it once. Falls back to candidate #0 if ranking fails to parse.
- **`adaptive_think(query, route=True)`** — the dispatcher that ties both axes together:
  estimate difficulty → budget, classify type → strategy, then run the chosen strategy
  inside a tracer span. `route=False` collapses to the article's original budget-only
  single pass, which is useful for ablation ("does routing actually help?").

---

<a name="phase-2"></a>
## Phase 2 — Tools

Everything the agent can *do* to the world. Two design rules dominate: **paths are
sandboxed to `WORKSPACE`**, and **outputs are truncated** so a giant file never blows
the context.

### Path & output safety
- **`_safe_path(path)`** — resolves a path and `relative_to(WORKSPACE)`-checks it,
  raising if it escapes. This is the containment boundary for every file op. (Note it
  resolves *after* joining, so symlink games are the residual risk — acceptable for a
  local single-user research tool.)
- **`_truncate(s, limit)`** — clip with a `[truncated N chars]` marker.

### File/shell tools
- **`tool_bash(command)`** — runs in `WORKSPACE` with a timeout, after scanning the
  `BASH_BLOCKLIST`. **Honest caveat (stated in v1 lineage):** the blocklist is a speed
  bump, not a sandbox — `shell=True` means a determined model could evade it. It's
  acceptable because the workspace is a throwaway dir and the user runs this locally.
- **`tool_read(path, start, end)`** — reads with 1-indexed line numbers prefixed (so the
  model can refer to lines), optional range, `errors="replace"` so binary junk doesn't crash.
- **`tool_write(path, content)`** — **snapshots the prior content into `SNAPSHOTS`
  before writing**, enabling `tool_revert`. Reports created-vs-updated.
- **`tool_grep` / `tool_glob`** — regex search and glob, both clamped to `WORKSPACE`
  (glob filters results by `is_relative_to`, caps at 200 hits).
- **`tool_revert(path)`** — pops the in-memory snapshot and restores (or deletes if the
  file was newly created). *Choice:* an **in-memory undo stack** rather than git — the
  article used a git checkpointer, but the workspace already lives inside a repo and the
  notebook explicitly refuses to nest one.

### Coding-specific tools (the quality gates)
- **`lint_python(code) -> {passed, errors}`** — a lightweight static gate: writes to a
  temp file and `py_compile`s it (catches syntax errors), then walks the AST to flag
  bare `except:`. *Choice:* deliberately minimal — no ruff/flake8 dependency, just
  "does it compile + one anti-pattern." The point is a fast, dependency-free *must-pass*
  filter, not a full linter.
- **`safe_write_code_file(filename, content)`** — the **lint-gated write**: rejects
  unless the filename is a bare `*.py` and the content lints clean. This is the central
  reliability mechanism — **broken code never reaches disk**, so downstream test runs
  never fail for trivial syntax reasons.
- **`run_python(code, timeout)`** — writes the snippet to `_run.py` with `agent_code/`
  prepended to `sys.path`, runs it as a subprocess, returns `{exit_code, stdout}`.
  *Choice:* a fresh subprocess per run (not `exec`) gives real isolation and a hard
  timeout. This replaces the article's Docker `PersistentSandbox` — same intent,
  no Docker dependency.
- **`run_tests(test_code, timeout)`** — writes a test module and runs it with **pytest
  if available, else plain python**, then regex-parses `N passed` / `N failed` out of
  the output into a structured `{all_passed, passed, failed, stdout, exit_code}`. The
  regex fallback (infer pass/fail from return code when counts are absent) is what lets
  it work for both pytest and bare-assert modules.

---

<a name="phase-3"></a>
## Phase 3 — The tool loop & subagent discipline

### `_fn(name, description, properties, required)`
A tiny helper that builds an OpenAI/Ollama-style function-tool schema dict. Keeps the
`TOOLS_BASE` list readable instead of repeating the nested `{"type":"function",…}`
boilerplate seven times.

### `_run_tool_call(tc, dispatch)`
Executes **one** tool call: extracts name + args (tolerating args delivered as a JSON
*string*), looks the name up in `dispatch`, runs it, and wraps the result as a
`{"role":"tool", …}` message (truncated). **Every failure is caught and returned as an
`[error] …` string rather than raised** — *crucial choice:* a tool error becomes an
observation the model can react to and recover from, instead of crashing the loop.
Unknown tools return the list of available tools so the model can self-correct.

### `master_loop(messages, tools, dispatch, …)`
The classic **perception → action → observation** loop: call the model; if it requested
tools, run them all and append their results; repeat until the model stops calling tools
or `max_iterations` is hit. It prepends the system prompt if missing and re-appends a
*clean* assistant message (dropping the internal `eval_count` bookkeeping key so it
doesn't pollute the next request). Each iteration is wrapped in a tracer `iter` span.

### `TOOLS_BASE` / `DISPATCH_BASE`
The schema list and the matching **name → lambda** dispatch table. The lambdas adapt the
loop's `args` dict to each tool's positional signature and JSON-encode the structured
runner results. This pairing (schema + dispatch) is what gets passed to every loop and
subagent.

### `SUBAGENT_SYSTEM` + `spawn_subagent(prompt, model)`
A subagent is just `master_loop` run with a focused system prompt and a fresh
6-hex-char id. The system prompt enforces **subagent discipline**: one subtask, no
clarifying questions (make a reasonable assumption), and *"your final message is the
ONLY thing the parent sees — make it self-contained."* The function returns the last
non-empty assistant message (think-stripped). *Why this matters:* it enforces the
**context-isolation** property of subagents — the parent's context stays clean, paying
only for the distilled summary, not the subagent's entire tool transcript. (Phase 6
redefines this function to additionally bound the subagent's *own* context.)

---

<a name="phase-4"></a>
## Phase 4 — The hardening stack

Four independent reliability patterns, each a pure function so they compose.

### `architect_editor_solve(task)` — separation of deliberation from transcription
Two-model split: the **reasoning** model (`ARCHITECT_SYSTEM`, JSON) produces a
*structured plan* (sections + design decisions) but **explicitly not code**; the
**fast** model (`EDITOR_SYSTEM`) executes that plan into the final artifact.

Two sharp choices here:
- The editor prompt starts with **`/no_think`** to disable qwen3's reasoning. The
  rationale is spelled out in the comment: the architect already deliberated, so the
  editor just transcribes — and disabling thinking makes it ~50× faster *and* stops
  `<think>` from eating the token budget and truncating the generated code. This is a
  genuinely important qwen3-specific optimization.
- The plan is passed to the editor as JSON, so the architect's constraints travel
  verbatim. Parse failures degrade to an empty plan rather than crashing.

### `self_refine(query, iterations=2)` — critique-then-revise
A generate → self-critique → refine loop, all on one model. Each round asks the model to
critique its own output "as a strict reviewer" (2–5 specific issues), then rewrite
addressing every point. Keeps full `history` for inspection. *Choice:* the critique and
refine are **separate calls** rather than one "improve this" call — separating the
*finding* of flaws from the *fixing* of them produces sharper critiques.

### `code_with_tests(task, test_code, max_rounds=3)` — the external-feedback loop
The single most important pattern for code reliability: generate → **lint-gate** →
**run real tests** → on failure, feed the actual test output back as `PREVIOUS ATTEMPT
FAILED:` and regenerate. Returns the best attempt with status (`passed` /
`failed_after_max_rounds`) and per-round history. *Choice:* the feedback is the
**verbatim test stdout**, not a model's paraphrase of it — the ground-truth error is the
most useful possible signal for the next attempt. Lint failures short-circuit before
wasting a test run.

### `adversarial_probe(target, candidate, n_max=4)` — red-teaming
The **reasoning** model plays "hostile adversary" and returns structured attacks (edge
cases, counterexamples, unhandled failures) each with the exact triggering input and a
severity. *Choice:* slightly higher temperature (0.4) to encourage creative attacks, and
it's purely *advisory* — it surfaces risks for the reviewer rather than gating, because
a generated "attack" might be a false alarm.

---

<a name="phase-5"></a>
## Phase 5 — Planning & durable state

### `@dataclass PlanStep` / `@dataclass Plan` / `make_plan(goal)`
`make_plan` asks the **reasoning** model (JSON, temp 0) for a dependency-ordered plan and
parses it into typed `PlanStep`s (`step_id`, `description`, `depends_on`,
`expected_artifact`). Dataclasses give structure without ceremony; a parse failure
returns an empty `Plan` so callers never get `None`.

### `class TaskDAG` — durable, dependency-aware work tracking
A **SQLite-backed** DAG of work nodes (`node_id, title, status, attempts, depends_on`).
- `add_node` uses `INSERT OR REPLACE` (idempotent re-seeding).
- `ready_nodes()` returns pending nodes whose dependencies are all `done` — this is the
  scheduler the agent loop consumes.
- `set_status` also increments `attempts`, giving a retry counter for free.

*Why SQLite, not an in-memory list?* **Durability.** The DAG survives a kernel restart
or a crashed run; `isolation_level=None` (autocommit) means each status update is
persisted immediately. It's the article's task-graph idea made crash-safe with zero
extra dependencies.

### `class BiTemporalMemory` — facts with validity intervals
A dependency-free memory store. Each fact has `valid_from`/`valid_to`; superseded facts
are **invalidated, not deleted** (`valid_to` is timestamped + a reason recorded). This is
the "bi-temporal" idea: you can always ask "what did the agent believe *then*" vs "what
is true *now*."
- `query_valid(kind)` returns currently-valid facts (optionally filtered by kind).
- `recall(query, k)` does **keyword-overlap** ranking — set-intersection of word tokens
  — with *no embeddings and no ChromaDB*. *Choice:* the article used a four-tier ChromaDB
  store; v2 deliberately trades recall quality for zero dependencies and full
  transparency, which is fine because the corpus is one run's worth of facts.

### The spec layer — definition-of-done as executable contract
- `write_definition_of_done(criteria, import_line)` persists the contract to
  `DEFINITION_OF_DONE.json`.
- `compile_test_suite(criteria, import_line)` **codegens a real pytest module** from the
  criteria — each `{"name", "check"}` becomes a `def test_name(): assert <check>`.
- `spec_verify(contract)` compiles the suite and runs it via `run_tests`.

*This is the linchpin of the whole notebook's epistemics:* the "definition of done" is
not prose the model self-grades against — it is **compiled into tests that actually
execute against the agent's code**. "Done" means the suite is green, full stop.

---

<a name="phase-6"></a>
## Phase 6 — Context engineering: a bounded working window

The insight motivating this phase: in a *coding* tool loop the context doesn't grow by
user turns (there's one task) — it grows by accumulated **tool observations** (file
dumps, test logs). Left unbounded, a long run drowns the model in stale output. The fix
is **trim → distill → reinject → consolidate**.

### `@dataclass TrimResult`
Carries the outcome of a trim: the new `messages`, whether anything was `trimmed`, how
many messages were `dropped`, and how many facts were `distilled`.

### `class ContextManager`
Bounds a loop's working context and preserves what it drops as durable memory. Built on
the Phase-5 `BiTemporalMemory` so distilled facts live in the same store as everything
else.

- **`__init__`** — `max_steps` (how many recent assistant-led steps to keep verbatim),
  the summarizer model, `recall_k`, and a `call_timeout`. The distill/consolidate calls
  are **best-effort**, so they get a short timeout to degrade to a raw-summary fallback
  rather than hang the run.
- **`_split(messages)`** — partitions into `head` (system messages), `anchor` (the
  original task — the first user message), and `body`. *Choice:* the system prompt and
  the task are **never trimmed** — losing the task is catastrophic, so it's a permanent
  anchor.
- **`trim(messages)`** — finds assistant-message indices ("step starts"); if there are
  more than `max_steps`, it cuts everything before the last `max_steps` steps, distills
  the dropped span, and returns `anchor + recent`. Deterministic — no model call unless
  there's actually something to distill.
- **`_transcript(msgs)`** — flattens dropped messages into a compact text transcript
  (including which tools were called), clipping each to 600 chars.
- **`_distill(dropped)`** — one JSON call (`DISTILL_SYSTEM`) compressing the transcript
  into **high-signal reusable facts** (files written, test results, decisions,
  constraints) — explicitly dropping chit-chat, raw code bodies, and stack traces. On
  any failure it stores a raw truncated summary so *something* survives. Facts go into
  `BiTemporalMemory` tagged `source="distill"`.
- **`render_block(query)`** — recalls relevant facts and renders them as a
  `<durable_memory>` block prefixed with a `CONTEXT_POLICY` that *tells the model how to
  treat it*: "this is an accurate record of what already happened — don't redo work
  marked done; recent messages still take precedence." *Choice:* the policy text is as
  important as the facts — without it the model might ignore or distrust the injected block.
- **`consolidate(critic=True)`** — periodically merges/de-dupes the distilled facts via
  a **writer** call, then optionally a **critic** call that audits whether the rewrite
  dropped anything important; if the critic says unsafe, the consolidation is **rejected**
  and the originals are kept. Only on approval are the old facts invalidated and the
  merged set stored. *This writer+critic pair mirrors the article* and is a guard against
  the classic failure mode of memory-compaction silently losing a load-bearing fact.

### `managed_loop(...)`
`master_loop` plus the bounded window: each iteration calls `ctx.trim()`, and if it
trimmed, rebuilds the system message as `base_system + render_block(task)` — i.e. the
distilled facts are **re-injected through the system prompt**. Otherwise identical to
`master_loop`.

### `spawn_subagent(...)` — redefined
This phase **redefines** the Phase-3 `spawn_subagent` to run on `managed_loop` with its
own (or a shared) `ContextManager`. So every subagent now keeps its own tool transcript
bounded and distils its dropped steps. Passing a shared `ctx` lets multiple subagents
write into one memory. *Choice — redefinition over a flag:* the notebook teaches the
flat version first (Phase 3), then transparently upgrades it, which reads well
top-to-bottom; the cost is that cell-execution order matters.

---

<a name="phase-7"></a>
## Phase 7 — MCP-style registry + the five-subagent coding architecture

### `class MCPTool` + `mcp_registry`
A minimal **Model-Context-Protocol-style registry**: each `MCPTool` wraps a
`name`/`description`/`handler` and `execute(**kwargs)` calls the handler. The registry
maps coding capabilities (`read_code`, `write_code`, `run_python`, `run_tests`,
`list_code`, `query_memory`). *Choice:* this is a **demonstration of the MCP pattern**
(uniform tool registry with metadata) rather than a full MCP server — `query_memory` is
even left as a stub to be bound to the live agent, signalling the seam where a real MCP
backend would plug in.

### `class Subagent` and the five roles
A tiny base class (`name`, `parent`) subclassed into five specialists, each with one
`execute(...)` method. This is the article's five-dengue-subagent architecture
**re-cast for coding**:

- **`Planner`** — runs `make_plan` on the task and stores each step into memory as a
  `plan` fact. Produces the roadmap.
- **`CodeImplementer`** — the workhorse. Round 1 uses **architect/editor** to draft the
  target file; later rounds switch to the **reasoning** model fed the verbatim test
  failure. Each round: lint-gated write → run the compiled contract tests → record the
  pass/fail into memory → return success the moment tests are green, else loop. This is
  `code_with_tests` specialized to the contract, with an escalation to the stronger model
  on retry — a deliberate "spend more compute only when the cheap path failed" choice.
- **`Tester`** — runs `spec_verify` **independently** of the implementer. *Why a separate
  role re-runs the same tests:* it's a clean separation between "the builder thinks it
  passes" and "an independent step confirms it passes," matching the
  no-claim-without-evidence ethos.
- **`Reviewer`** — reads the file, scores it with `verifier_score`, red-teams it with
  `adversarial_probe`, and stores the review. Advisory, not gating.
- **`ReportWriter`** — recalls the run's facts from memory and uses `self_refine` to
  write a concise `REPORT.md`. The memory store is what lets the report be grounded in
  what *actually happened* rather than a hallucinated narrative.

### `class CodingAgent` + `agent_run(agent, max_iters)`
`CodingAgent` is the shared state container — task, target filename, contract, DAG,
memory, and the `routing` map (DAG node id → subagent instance). `agent_run` is the
**DAG-driven scheduler**: repeatedly pull `ready_nodes()`, dispatch the first to its
subagent, mark `done`/`failed` by the result, and **stop the whole run on the first
failure** (fail-fast — no point reviewing code whose tests don't pass). Each node runs in
a tracer `dag` span with a coloured success/fail event.

*Why a DAG instead of a fixed sequence?* The dependency graph (`plan → implement → test
→ review → report`) makes the ordering data, not code — you can re-wire the pipeline by
editing the seed list (Phase 8) without touching `agent_run`.

---

<a name="phase-8"></a>
## Phase 8 — How it all runs

The driver cell ties everything together on a deliberately simple, fully-deterministic
task (a Roman-numeral module, *not* dengue — to prove the engine is general):

1. **Healthcheck** the server/models (fail-fast).
2. Define `CODING_TASK` + four `CRITERIA` (including a brutal round-trip check over all
   1–3999) and persist the **definition of done** via `write_definition_of_done`.
3. **Seed the DAG**: `sg1 plan → sg2 implement → sg3 test → sg4 review → sg5 report`.
4. Build a fresh `BiTemporalMemory` and a `CodingAgent`, then `agent_run`.

The later cells re-verify the contract *independently*, inspect the produced files, dump
the per-label cost ledger via `tracer.summary()`, run a **structural census** of the
engine (`census` cell — counts tools/roles/patterns with no model calls), and render the
LangSmith-style tree with `render_trace()`. The self-test cells (Phase 10) exercise the
memory, the tolerant parsers, and the trim logic deterministically (no model calls), so
the plumbing is testable without burning GPU time.

---

<a name="themes"></a>
## Cross-cutting design themes

These recur deliberately and are the real "lessons" of the notebook:

1. **One model-call chokepoint (`chat_complete`).** Backend portability, uniform
   logging/tracing, and a single place to add retries later. Named labels turn it into a
   cost ledger.

2. **Two-tier model economics.** Cheap model for high-volume generation and routine
   work; expensive model reserved for *judging/architecting* — the steps where quality
   asymmetry actually pays. `asymmetric_solve` is the purest expression: parallel cheap
   generation, one strong ranking call.

3. **No claim without a runnable artifact.** Lint gate → real test execution →
   independent re-verification. The "definition of done" is *compiled into tests*, not
   self-graded. Test failures feed back **verbatim**, never paraphrased.

4. **Tolerant parsing everywhere.** `<think>` stripping, `{…}`-span JSON fallback, code-
   fence stripping, and a try/except-with-safe-default on *every* classifier. A parse
   failure degrades; it never halts the run.

5. **Graceful degradation as a habit.** No rich → indented logs. No pytest → plain
   python. Distill/consolidate time out → raw summary. The system always has a fallback.

6. **Durable, inspectable state.** SQLite DAG survives crashes; bi-temporal memory
   invalidates rather than deletes (you can audit what the agent believed and when);
   in-memory snapshots give undo without git.

7. **Compute-adaptive effort on two axes.** Difficulty sizes the *budget*; problem type
   selects the *strategy*. The type axis is near-inert for coding but is kept as the
   bridge to the non-coding domains this notebook leads toward.

8. **Context as a managed resource.** A coding loop grows by tool observations, so the
   bounded window distils dropped steps into durable facts and re-injects them with an
   explicit usage policy — with a writer+critic guard against losing load-bearing facts.

9. **Observability is first-class, not bolted on.** Two views (live stream + navigable
   tree) fed from the same instrumentation points, controllable down to a no-op via env
   vars so it costs nothing in production.

10. **Honest about its own limits.** The comments openly flag where v2 simplifies the
    article (no Docker, no ChromaDB, no git checkpointer, near-constant type axis) and
    where the safety is a speed bump rather than a wall (`tool_bash`). The intent is a
    *teaching* engine you can reason about end-to-end, not a hardened product.
