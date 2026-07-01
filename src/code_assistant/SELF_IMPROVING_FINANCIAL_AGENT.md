# Self-Improving Financial Agent — Design

A design for a financial-QA agent that **improves itself** with no large pre-existing
evaluation set. The agent answers complex questions over balance-sheet / income-statement
tables (Text-to-SQL); an adversarial **Setter** manufactures the evaluation data it lacks; a
**Judge** turns failures into typed improvement instructions; a **gate runner** applies each
change only if it provably helps. The coding-agent half of the loop already exists as
`coding_agent_gemini.py` — this document specifies the rest and the contracts that bind them.

> Companion code: `financial_agent_trainer.py` (the trainer skeleton implementing the
> contracts below). Coding agent: `coding_agent_gemini.py`. Base design lineage:
> `claude_code_from_scratch_v3.ipynb`.

---

## 1. The core problem: how does the Judge know an answer is correct?

The tempting answer — **run the main agent 3× and take the most common output** — measures
**confidence, not correctness**. Self-consistency tells you the model is *stable*, not that it
is *right*; in a self-improvement loop it actively **amplifies systematic errors**, because a
consistently-wrong agent looks maximally "correct." Voting is kept only as a *non-determinism
signal*, never as a truth signal.

Correctness here comes from **ground truth by construction + execution accuracy (EX)**:

- The question is built *by querying the tables*, so the constructor **knows the gold answer
  because it built the query**. There is no labelling step.
- Correctness = the executed result set of the agent's SQL equals the executed result set of
  the gold SQL (Spider/BIRD-style execution accuracy), compared with numeric tolerance and
  set/þorder normalization — not string match.
- The referee is therefore **non-learned** (SQL engine + a unit-tested metrics library), which
  is the single fact that keeps the loop from the GAN failure mode (a learned discriminator
  that can drift, collude, or be fooled).

---

## 2. Architecture: asymmetric self-play (not "two agents grading each other")

```
  Setter (adversary)  ──constructs──►  (NL question, gold_sql, gold_value, category)
        │                                          │
        │                              ┌───────────┴────────────┐
        │                              ▼                        ▼
        │                    Production Solver           Oracle Solver
        │                    (agent under improvement)    (strong/expensive ref:
        │                              │                   reasoning model, gold
        │                              │                   schema, N tries)
        │                              ▼                        ▼
        │           Verifier (NON-learned: SQL exec + metrics lib)
        │                              │
        │                       FilterVerdict ──► fair-hard? ──► Example (eval datum)
        │                              │                              │
        └────── reward(candidate) ◄────┘                             ▼
                                                                  Judge
                                                                     │
                                                              ImprovementProposal
                                                                     │
                                                           GateRunner (scratch→test→
                                                           benchmark→commit/revert)
                                                                     │
                                                          CodingAgent / policy edit
```

Roles:

- **Setter** — constructs verifiable questions designed to expose the Solver's weaknesses.
  Not free-text generation: it picks a schema region → a metric/template → samples
  parameters → renders to NL, so **gold SQL and gold value fall out for free**.
- **Production Solver** — the Text-to-SQL agent we want to improve.
- **Oracle Solver** — a strong, expensive reference (reasoning model, gold-schema access, more
  attempts). Used only to certify a question is *fair* (answerable & well-posed).
- **Verifier** — non-learned: executes SQL and compares result sets; also recomputes gold via
  the metrics library for the `gold_suspect` check.
- **Judge** — classifies each failure (taxonomy §4) and emits a typed `ImprovementProposal`.
- **GateRunner** — applies one atomic change to a scratch copy, proves it, commits or reverts.

This is **AlphaGeometry / asymmetric-self-play** territory (a Setter proposes verifiable
problems, a Solver solves them), **not** a GAN.

---

## 3. Bootstrapping data from zero: the fair-hard certificate

Seed = the **schema + a unit-tested metrics library** (no dataset). A naked adversary trivially
"wins" by proposing impossible/ambiguous/absurd questions, so each candidate passes a
**three-way check** before it becomes data:

1. **Well-posed & solvable?**  `oracle_value == gold_value`. If the strong solver can't get it,
   the question is ill-posed/unfair → **discard**. *(This is the anti-degeneracy gate.)*
2. **Informative?**  `production_value != gold_value`. Already-solved questions teach nothing →
   keep as a **frozen regression case** (weight 0), don't train on them.
3. **Novel?**  not a near-duplicate (embedding/template/region dedup) and not over its category
   quota → otherwise **discard** (mode-collapse guard).

Plus the escape valve: if `gold_sql` executes but disagrees with an **independent metrics-lib
recompute**, it's a `gold_suspect` → routed to the generator/metrics maintainer, **never** to
the coding agent. A buggy gold must never train the agent toward a wrong answer.

A candidate passing all three is a **fair-hard Example**: provably answerable, currently failed,
new. **Every fair-hard Example is a labelled eval datum** → the adversarial loop *is* the
dataset generator. Solved examples freeze into a no-regression benchmark; today's fair-hard is
tomorrow's regression guard. A tiny frozen **real anchor set** (30–50 hand-checked questions)
rides alongside as the overfitting detector — self-play measures progress against the Setter's
distribution; only the anchor proves progress on reality.

### The Setter reward

Rewarded for **fair-hard, novel, on-quota** examples in a **difficulty band** — not for
maximizing failure:

```
reward(candidate) =
    if gold_suspect:          -1.0     # constructing broken golds is punished hard
    if not well_posed:        -0.5     # impossible/ambiguous: the trivial exploit, blocked
    if not novel:              0.0     # duplicates earn nothing
    else:
        difficulty     = 1 - abs(solve_rate - TARGET)     # TARGET≈0.5: band, not max
        coverage_bonus = β * deficit[category]             # under-sampled F-category → more reward
        depth_bonus    = γ * min(len(decomposition), D_cap) / D_cap
        reward = difficulty + coverage_bonus + depth_bonus
```

- `difficulty` pins the Setter to the **zone of proximal development** (≈50% solve), so it can't
  win with a wall of impossible questions even past the well-posed gate.
- `coverage_bonus` is driven by live deficits vs. the quota (§5), so the Setter can't camp one
  family.
- `depth_bonus` nudges toward multi-hop (F5), capped at `D_cap`.
- `solve_rate` from k repeat runs doubles as the `non_deterministic` signal for the Judge.

---

## 4. Failure taxonomy (the routing key)

Each non-`CORRECT` answer gets exactly one **primary** category. The category routes the fix to
the right edit surface — a wrong category sends a good instruction to the wrong layer.

| #  | Category | Looks like | Default edit tier |
|----|----------|-----------|-------------------|
| F1 | Schema/grounding | wrong table/column, hallucinated field, wrong join key | tools/retrieval → few-shot |
| F2 | Semantic/financial-logic | runs, but answers a different question (gross≠net, flow≠stock, sign, intercompany) | system prompt → few-shot |
| F3 | Aggregation/period | right columns, wrong grain (period rollup, fiscal boundary, TTM, FX) | system prompt → few-shot |
| F4 | SQL-correctness | fails to execute, NULL/div-by-zero/bad cast | core code (guardrail/repair) |
| F5 | Decomposition | multi-hop; dropped a constraint, stopped early | few-shot → core code (planner) |
| F6 | Calculation | right rows, wrong in-agent arithmetic (ratio, growth %, margin) | core code (**promote to verified tool**) |
| F7 | Format/extraction | right value, wrong unit/rounding/format | system prompt (output contract) — lowest priority |
| F8 | Abstention | unanswerable from tables, but fabricated a number | system prompt → few-shot |

Two cross-cutting flags (not categories): `gold_suspect` (→ generator path) and
`non_deterministic` (instability, evidence of confidence not correctness).

F6 and F2/F3 are the highest-value targets — where **"promote a recurring fix into a verified
tool"** pays off most.

---

## 5. Construction templates × taxonomy (the coverage quota)

Each template is a **parameterized constructor**: it builds `gold_sql` from sampled params so
the gold value is free, and is designed to *stress* a specific category. The quota is a target
distribution; `deficit[category] = target_share − observed_share` drives the reward's coverage
bonus. Quota is enforced on the **realized** category (from the verdict), not the intended one.

| Target | Template family | Construction | Quota |
|--------|----------------|-------------|-------|
| F1 | `near_miss_entity` | two similarly-named accounts; ask about one; gold filters the right one | 12% |
| F2 | `gross_vs_net`, `flow_vs_stock`, `sign_convention`, `intercompany` | build from a precisely-defined metric; gold uses the correct definition | 18% |
| F3 | `period_rollup`, `fiscal_boundary`, `ttm_vs_point`, `fx_normalize` | window straddling a fiscal boundary / multi-currency; gold normalizes | 18% |
| F4 | `null_trap`, `div_zero`, `type_cast` | inject a NULL/zero denominator into a normal ratio question | 8% |
| F5 | `multi_constraint`, `nested_compare` | compose 2–4 filters + a comparison; decomposition = the constraints | 20% |
| F6 | `ratio`, `growth_pct`, `margin`, `cagr` | gold computed by the **verified metric function**, not SQL arithmetic | 12% |
| F7 | `unit_round`, `presentation` | same value, demand a specific unit/precision/format | 4% |
| F8 | `unanswerable` | ask for a field/period **not present**; gold = `UNANSWERABLE` | 8% |

Notes:
- **F6 gold must come from the metrics library, never `gold_sql` arithmetic** — else
  "promote to verified tool" has no trustworthy target.
- **F8 gold is a label, not a value**: its filter asserts the oracle *also* abstains; an oracle
  that finds an answer ⇒ `gold_suspect` (the field exists).
- Quotas are starting priors with a **per-category floor**; the loop lets them drift toward the
  Solver's weakest categories so coverage never collapses.

---

## 6. The Judge→CodingAgent contract

Diagnosis (what's wrong + evidence) is separated from prescription (what to change) so the
Judge's reasoning is auditable and a downstream **policy**, not the Judge's prose, decides what
is allowed.

```python
Evidence(task_id, question, gold_sql, gold_value, agent_sql, agent_value,
         agent_trace_ref, ex_match, runs)
Diagnosis(category: Failure, gold_suspect, non_deterministic, root_cause, evidence: list)
Instruction(target: {system_prompt|few_shot|tools|core_code|generator},
            action:  {add|modify|remove|promote_to_tool},
            rationale, payload, scope, expected_effect, risk, test_to_add)
ImprovementProposal(proposal_id, diagnosis, instructions: list, supersedes)
```

Design choices:
- `target`/`action` are **enums** — the Judge can't invent an edit surface. `promote_to_tool`
  is first-class (the strongest move).
- `test_to_add` is **mandatory** for any category-fixing change: a frozen benchmark case that
  fails before and passes after. No falsifiable test ⇒ not "done". This blocks reward hacking.
- `scope` fights over-generalization (a multi-period fix must say so or it regresses
  single-period questions).
- **Atomic changes**: the loop applies **one `Instruction` per iteration**, gated independently;
  the `ImprovementProposal` carries the cluster, the loop linearizes it.
- `supersedes` gives an append-only revision chain with full provenance + rollback.

Mapping into the existing coding agent:
- `target ∈ {system_prompt, few_shot}` → `CodingAgent.add_instruction(payload)` (the
  LEARNED-GUIDANCE seam) + `save_policy`/`load_policy`.
- `target ∈ {tools, core_code}`, `action=promote_to_tool` →
  `CodingAgent.generate(task=<spec>, test_code=<test_to_add>)`; `test_to_add` *is* the
  definition of done.
- `target=generator` → metrics-library maintainer path; never touches the coding agent.

---

## 7. The gate (between "instruction received" and "change kept")

```
Instruction
   │  apply to a SCRATCH copy of the agent policy
   ├─ run test_to_add        → must pass        (proves the fix)
   ├─ run frozen benchmark   → no regression     (proves no collateral)
   ├─ Δ execution-accuracy   → must be ≥ 0       (proves net value)
   ├─ all pass → COMMIT  (append to policy, record provenance, freeze test_to_add into benchmark)
   └─ any fail → REVERT  (log as a rejected hypothesis, feed back to the Judge)
```

The rejected-hypothesis feedback closes the meta-loop: the Judge learns which *kinds* of
instructions don't work, not just the agent.

---

## 8. Build order

1. **Contracts** — the typed envelopes above (done first; everything else consumes them).
2. **Warehouse + Verifier** — SQL execution + execution-accuracy comparison (SQLite for offline
   tests, the real warehouse as a swap-in adapter).
3. **Metrics library** — unit-tested pure functions; the gold-by-construction anchor.
4. **Setter (templates)** — the construction engine + the fair-hard filter + reward.
5. **Solver** — Text-to-SQL production agent; **Oracle** — strong reference.
6. **Judge** — verdict → Diagnosis → ImprovementProposal.
7. **GateRunner** — scratch/test/benchmark/commit-or-revert, wired to `CodingAgent`.
8. **Trainer loop** — orchestration + coverage-quota tracking + the real anchor set.

Offline-first, mirroring `coding_agent_gemini.py`: every pure-logic path (contracts, verifier,
filter, reward, gate state machine) is runnable and self-tested **without a Gemini backend**;
only the Setter/Solver/Oracle/Judge generation steps need the cloud model, and they sit behind
role interfaces with deterministic offline stubs so the whole loop runs end-to-end on SQLite.
