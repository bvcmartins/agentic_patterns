# code_assistant

## `building_claude_from_scratch.ipynb`

A runnable notebook reproduction of Fareed Khan's article
**"Building Claude from Scratch: 62 Components Behind Anthropic's Thinking
Engine"** (Level Up Coding, May 2026).

The notebook builds a Claude-like agentic *harness* on top of an
open-source model (DeepSeek via an OpenAI-compatible API) across 8 phases,
and applies it to one hard task: reproducing the national 75th-percentile
dengue forecast (1,405,191 cases) from Freitas et al. 2025 within 5%.

### Phases
1. **Cognitive Substrate** — thinking channel, compute-adaptive budgets,
   self-consistency / best-of-N / budget forcing.
2. **Reasoning Topology** — step-back, least-to-most, Tree of Thoughts,
   OODA subagents, single-threaded master loop, subagent output discipline.
3. **Tool-Grounded Execution** — plan-and-execute, ReAct,
   evaluator-optimizer, Reflexion, CRITIC, mixture-of-agents,
   verifier asymmetry.
4. **Production Reliability** — self-refine, verifier-guided search,
   external-feedback verification, adversarial probing, architect/editor
   split, linter-in-the-loop, cache-aware prompt ordering.
5. **Frontier-Only Patterns** — thought signatures, Goldilocks altitude,
   compute-optimal allocation, coverage curves, soul document,
   deliberative alignment, bi-temporal memory.
6. **Meta-Cognition & Stateful Orchestration** — problem-type
   classification, definition-of-done contract, SQLite-backed task DAG,
   selective rollback / replan.
7. **Grounding & the Trust Gate** — persistent sandboxed REPL, git
   checkpointing, executable pytest spec layer, four-tier memory,
   MCP-compatible tool registry.
8. **Composition** — five-subagent architecture, master loop, the
   end-to-end reproduction run and verdict.

### Running it
1. Set an OpenAI-compatible API key:
   ```bash
   export DEEPSEEK_API_KEY="sk-..."
   ```
   Swap `base_url` / `MODEL_*` in the foundation cell to target any other
   OpenAI-compatible backend (vLLM, Ollama, OpenRouter, Together, ...).
2. Provide the workspace data the later phases expect under
   `./seird_workspace/`: `paper.txt` and `data/cases.csv.gz` (DATASUS dengue
   surveillance data). Phases 1–6 (LLM-only) run without the dataset;
   Phases 7–8 also need Docker for the sandbox.

### Notes
- Each code cell is the real code from the article. Where the article showed
  results, they are reproduced in `Output (from the article)` blocks so you
  can follow along without spending API budget — live re-runs produce similar,
  not byte-identical, output.
- One copy/paste indentation slip in the article's first `think_then_answer`
  cell was corrected so the cell is valid Python.

Original code + theory:
https://github.com/FareedKhan-dev/building-claude-from-scratch
