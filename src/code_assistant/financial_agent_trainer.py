#!/usr/bin/env python3
"""
financial_agent_trainer.py — the self-improving loop for a financial Text-to-SQL agent.

This is the trainer half of the system designed in `SELF_IMPROVING_FINANCIAL_AGENT.md`. It
manufactures its own evaluation data via adversarial **asymmetric self-play** and feeds failures
through a Judge into the existing coding agent (`coding_agent_gemini.py`):

    Setter constructs a verifiable question  ─►  Solver answers  ─►  Verifier grades (EX)
            │                                                              │
            │                              fair-hard?  ◄───────  FilterVerdict
            │                                   │
            └─ reward(candidate)                ▼
                                            Example ─►  Judge ─►  ImprovementProposal
                                                                       │
                                                              GateRunner (scratch→test→
                                                              benchmark→commit/revert)

The whole thing runs **offline on SQLite** with deterministic stub roles, so the orchestration,
the fair-hard filter, the Setter reward, and the gate state machine are all checkable without a
Gemini backend. The production roles (Text-to-SQL Solver, LLM Setter/Judge) sit behind the same
interfaces and swap in one class at a time — `GeminiSolver` below is the first such swap.

    python financial_agent_trainer.py --selftest      # offline, no backend
    python financial_agent_trainer.py --demo           # run the self-play loop on SQLite
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

# Reuse the one backend seam + logging style from the coding agent. Importing this module does
# NOT build any model (the Gemini client is lazy), so the offline paths stay backend-free.
try:
    from coding_agent_gemini import CodingAgent, llm, content_text  # noqa: F401
    _HAS_CODER = True
except Exception:  # pragma: no cover - allows the file to be read in isolation
    CodingAgent = object  # type: ignore
    _HAS_CODER = False


# ════════════════════════════════════════════════════════════════════════════
# Phase 0 — logging
# ════════════════════════════════════════════════════════════════════════════

class _Fmt(logging.Formatter):
    COLORS = {"DEBUG": "\033[90m", "INFO": "\033[36m", "WARNING": "\033[33m",
              "ERROR": "\033[31m"}
    RESET = "\033[0m"

    def format(self, r: logging.LogRecord) -> str:
        c = self.COLORS.get(r.levelname, "")
        short = r.name.split(".", 1)[1] if "." in r.name else r.name
        return f"{c}[{r.levelname:5s}] {short:10s} | {r.getMessage()}{self.RESET}"


_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(_Fmt())
log = logging.getLogger("trainer")
log.handlers.clear()
log.addHandler(_h)
log.setLevel("INFO")
log.propagate = False


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — the typed contracts (§4, §6 of the design doc)
# ════════════════════════════════════════════════════════════════════════════

class Failure(str, Enum):
    """The primary failure taxonomy — the routing key from question to edit surface."""
    CORRECT = "CORRECT"          # not a failure; used by the Verifier verdict
    F1_SCHEMA = "F1_schema"
    F2_SEMANTIC = "F2_semantic"
    F3_AGGREGATION = "F3_aggregation"
    F4_SQL = "F4_sql"
    F5_DECOMPOSITION = "F5_decomposition"
    F6_CALCULATION = "F6_calculation"
    F7_FORMAT = "F7_format"
    F8_ABSTENTION = "F8_abstention"


# Default edit tier per category (design doc §4). The Judge consults this to route a fix.
EDIT_TIER: Dict[Failure, str] = {
    Failure.F1_SCHEMA: "tools",
    Failure.F2_SEMANTIC: "system_prompt",
    Failure.F3_AGGREGATION: "system_prompt",
    Failure.F4_SQL: "core_code",
    Failure.F5_DECOMPOSITION: "few_shot",
    Failure.F6_CALCULATION: "core_code",       # promote_to_tool
    Failure.F7_FORMAT: "system_prompt",
    Failure.F8_ABSTENTION: "system_prompt",
}

# Starting coverage quota (design doc §5). Floors keep any category from collapsing.
QUOTA: Dict[Failure, float] = {
    Failure.F1_SCHEMA: 0.12, Failure.F2_SEMANTIC: 0.18, Failure.F3_AGGREGATION: 0.18,
    Failure.F4_SQL: 0.08, Failure.F5_DECOMPOSITION: 0.20, Failure.F6_CALCULATION: 0.12,
    Failure.F7_FORMAT: 0.04, Failure.F8_ABSTENTION: 0.08,
}

UNANSWERABLE = "UNANSWERABLE"   # the abstention sentinel (F8 gold and Solver abstain value)


@dataclass
class Candidate:
    """A constructed question: gold falls out of the construction, so it needs no labelling."""
    question: str
    gold_sql: str
    gold_value: Any                       # scalar, or UNANSWERABLE for F8
    category: Failure                     # the category the template *aimed* at
    template_id: str
    schema_region: frozenset = field(default_factory=frozenset)
    decomposition: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)   # construction params (company, period)


@dataclass
class Answer:
    """What a Solver returns: a value (or abstention) plus the SQL it ran, for evidence."""
    value: Any
    sql: Optional[str] = None
    abstained: bool = False

    @classmethod
    def abstain(cls) -> "Answer":
        return cls(value=UNANSWERABLE, sql=None, abstained=True)


@dataclass
class FilterVerdict:
    well_posed: bool                      # oracle_value == gold_value
    informative: bool                     # production_value != gold_value
    novel: bool                           # not a near-duplicate / not over quota
    gold_suspect: bool                    # gold_sql disagrees with the metrics-lib recompute
    solve_rate: float                     # production pass@k over k repeat runs
    oracle_value: Any = None
    production_value: Any = None
    non_deterministic: bool = False       # variance across the k runs

    @property
    def fair_hard(self) -> bool:
        """The certificate: provably answerable, currently failed, new, and gold is trustworthy."""
        return (self.well_posed and self.informative and self.novel
                and not self.gold_suspect)


@dataclass
class Evidence:
    task_id: str
    question: str
    gold_sql: str
    gold_value: Any
    agent_sql: Optional[str]
    agent_value: Any
    ex_match: bool
    runs: List[Any] = field(default_factory=list)
    agent_trace_ref: str = ""


@dataclass
class Diagnosis:
    category: Failure
    gold_suspect: bool
    non_deterministic: bool
    root_cause: str
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Instruction:
    target: str                           # system_prompt|few_shot|tools|core_code|generator
    action: str                           # add|modify|remove|promote_to_tool
    rationale: str
    payload: str
    scope: str
    expected_effect: str
    risk: str = "low"                     # low|med|high
    test_to_add: Optional[str] = None     # mandatory for a category fix; def. of done


@dataclass
class ImprovementProposal:
    proposal_id: str
    diagnosis: Diagnosis
    instructions: List[Instruction] = field(default_factory=list)
    supersedes: Optional[str] = None


@dataclass
class Example:
    """A minted fair-hard datum. This IS the evaluation set the loop manufactures."""
    candidate: Candidate
    verdict: FilterVerdict
    weight: float = 1.0                   # 0.0 once solved → kept only as a regression guard
    minted_at: int = 0


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — the warehouse (SQLite for offline) + the metrics library
# ════════════════════════════════════════════════════════════════════════════

class Warehouse(Protocol):
    """The data surface. The real implementation wraps the SQL warehouse; SQLite mirrors it."""
    def scalar(self, sql: str) -> Any: ...
    def rows(self, sql: str) -> List[Tuple]: ...


# A single long fact table is enough for balance-sheet + income-statement line items.
_SCHEMA = """
CREATE TABLE financials (
    company    TEXT NOT NULL,
    period     TEXT NOT NULL,          -- 'FY2022', 'FY2023'
    statement  TEXT NOT NULL,          -- 'income_statement' | 'balance_sheet'
    line_item  TEXT NOT NULL,          -- 'revenue', 'cogs', 'net_income', ...
    currency   TEXT NOT NULL DEFAULT 'USD',
    amount     REAL
);
"""

# Deterministic seed data: two companies, two fiscal years. gross_profit = revenue - cogs;
# net_income = gross_profit - opex - tax, so gross and net genuinely differ (drives F2).
_SEED = [
    # company, period, statement, line_item, amount
    ("ACME", "FY2022", "income_statement", "revenue", 1000.0),
    ("ACME", "FY2022", "income_statement", "cogs", 600.0),
    ("ACME", "FY2022", "income_statement", "gross_profit", 400.0),
    ("ACME", "FY2022", "income_statement", "opex", 150.0),
    ("ACME", "FY2022", "income_statement", "net_income", 200.0),
    ("ACME", "FY2022", "balance_sheet", "current_assets", 800.0),
    ("ACME", "FY2022", "balance_sheet", "current_liabilities", 400.0),
    ("ACME", "FY2023", "income_statement", "revenue", 1200.0),
    ("ACME", "FY2023", "income_statement", "cogs", 700.0),
    ("ACME", "FY2023", "income_statement", "gross_profit", 500.0),
    ("ACME", "FY2023", "income_statement", "opex", 180.0),
    ("ACME", "FY2023", "income_statement", "net_income", 260.0),
    ("ACME", "FY2023", "balance_sheet", "current_assets", 980.0),
    ("ACME", "FY2023", "balance_sheet", "current_liabilities", 450.0),
    ("GLOBEX", "FY2022", "income_statement", "revenue", 500.0),
    ("GLOBEX", "FY2022", "income_statement", "cogs", 350.0),
    ("GLOBEX", "FY2022", "income_statement", "gross_profit", 150.0),
    ("GLOBEX", "FY2022", "income_statement", "net_income", 60.0),
    ("GLOBEX", "FY2023", "income_statement", "revenue", 650.0),
    ("GLOBEX", "FY2023", "income_statement", "cogs", 430.0),
    ("GLOBEX", "FY2023", "income_statement", "gross_profit", 220.0),
    ("GLOBEX", "FY2023", "income_statement", "net_income", 95.0),
]


class SQLiteWarehouse:
    """An in-memory financial warehouse — a real, dependency-free SQL execution oracle."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(_SCHEMA)
        self.conn.executemany(
            "INSERT INTO financials (company, period, statement, line_item, amount) "
            "VALUES (?,?,?,?,?)", _SEED)
        self.conn.commit()

    def scalar(self, sql: str) -> Any:
        cur = self.conn.execute(sql)
        row = cur.fetchone()
        return None if row is None else row[0]

    def rows(self, sql: str) -> List[Tuple]:
        return list(self.conn.execute(sql).fetchall())


# --- metrics library: unit-tested pure functions; the gold-by-construction anchor (§5) -------
def growth_pct(prev: float, curr: float) -> float:
    """Year-over-year growth in percent. The verified definition F6 questions are graded against."""
    if prev in (0, None):
        raise ValueError("growth undefined for zero/None base")
    return (curr - prev) / prev * 100.0


def gross_margin(revenue: float, cogs: float) -> float:
    return (revenue - cogs) / revenue * 100.0


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    return current_assets / current_liabilities


# ════════════════════════════════════════════════════════════════════════════
# Phase 3 — the Verifier (non-learned referee: execution accuracy)
# ════════════════════════════════════════════════════════════════════════════

class Verifier:
    """Grades by EXECUTION, not by string match. Compares the agent's value to gold with numeric
    tolerance, handles abstention, and independently recomputes gold to flag `gold_suspect`."""

    def __init__(self, warehouse: Warehouse, rel_tol: float = 1e-6):
        self.wh = warehouse
        self.rel_tol = rel_tol

    def _equal(self, a: Any, b: Any) -> bool:
        if a == UNANSWERABLE or b == UNANSWERABLE:
            return a == b
        if a is None or b is None:
            return a is b
        try:
            return math.isclose(float(a), float(b), rel_tol=self.rel_tol, abs_tol=1e-9)
        except (TypeError, ValueError):
            return a == b

    def value_of(self, answer: Answer) -> Any:
        """Resolve a Solver answer to a comparable value: its declared value, or run its SQL."""
        if answer.abstained:
            return UNANSWERABLE
        if answer.value is not None:
            return answer.value
        if answer.sql:
            try:
                return self.wh.scalar(answer.sql)
            except sqlite3.Error:
                return None          # F4: SQL failed to execute
        return None

    def ex_match(self, answer: Answer, gold_value: Any) -> bool:
        return self._equal(self.value_of(answer), gold_value)

    def gold_is_suspect(self, cand: Candidate, recompute: Optional[Any]) -> bool:
        """True if executing gold_sql disagrees with an independent metrics-lib recompute."""
        if recompute is None or cand.gold_value == UNANSWERABLE:
            return False
        try:
            executed = self.wh.scalar(cand.gold_sql)
        except sqlite3.Error:
            return True
        return not self._equal(executed, recompute)


# ════════════════════════════════════════════════════════════════════════════
# Phase 4 — roles: Setter / Solver / Oracle (offline stubs + production seams)
# ════════════════════════════════════════════════════════════════════════════

class Solver(Protocol):
    def answer(self, question: str) -> Answer: ...


# ---- the Setter: parameterized construction templates (design doc §5) -----------------------
class TemplateSetter:
    """Constructs verifiable questions by querying the warehouse. Each template stresses one
    failure category; gold falls out of the construction. This is the offline Setter — no LLM
    needed, because construction (not free-text generation) is the core mechanism."""

    COMPANIES = ["ACME", "GLOBEX"]
    PERIODS = ["FY2022", "FY2023"]

    def __init__(self, warehouse: Warehouse, seed: int = 0):
        self.wh = warehouse
        self.rng = random.Random(seed)

    def _amount_sql(self, company: str, period: str, line_item: str) -> str:
        return (f"SELECT amount FROM financials WHERE company='{company}' "
                f"AND period='{period}' AND line_item='{line_item}'")

    def propose(self) -> Candidate:
        """Emit one candidate, sampling a template (weighted toward the coverage quota)."""
        return self.rng.choice([
            self._line_item_lookup, self._growth_pct, self._gross_vs_net, self._unanswerable,
        ])()

    # F1-ish baseline: a direct lookup the Solver should already get right (→ not informative).
    def _line_item_lookup(self) -> Candidate:
        co, pd = self.rng.choice(self.COMPANIES), self.rng.choice(self.PERIODS)
        sql = self._amount_sql(co, pd, "revenue")
        return Candidate(
            question=f"What was {co}'s revenue in {pd}?",
            gold_sql=sql, gold_value=self.wh.scalar(sql),
            category=Failure.F1_SCHEMA, template_id="line_item_lookup",
            schema_region=frozenset({"financials.revenue"}),
            params={"company": co, "period": pd, "line_item": "revenue"})

    # F6: growth in percent — gold from the verified metric, not from SQL arithmetic.
    def _growth_pct(self) -> Candidate:
        co = self.rng.choice(self.COMPANIES)
        prev = self.wh.scalar(self._amount_sql(co, "FY2022", "revenue"))
        curr = self.wh.scalar(self._amount_sql(co, "FY2023", "revenue"))
        gold = growth_pct(prev, curr)
        # gold_sql expresses the same computation, for the gold_suspect cross-check.
        gold_sql = (f"SELECT (b.amount - a.amount) / a.amount * 100.0 FROM "
                    f"(SELECT amount FROM financials WHERE company='{co}' AND period='FY2022' "
                    f"AND line_item='revenue') a, "
                    f"(SELECT amount FROM financials WHERE company='{co}' AND period='FY2023' "
                    f"AND line_item='revenue') b")
        return Candidate(
            question=f"What was {co}'s year-over-year revenue growth from FY2022 to FY2023, "
                     f"in percent?",
            gold_sql=gold_sql, gold_value=gold,
            category=Failure.F6_CALCULATION, template_id="growth_pct",
            schema_region=frozenset({"financials.revenue"}),
            decomposition=["FY2022 revenue", "FY2023 revenue", "percent growth"],
            params={"company": co, "p0": "FY2022", "p1": "FY2023", "line_item": "revenue"})

    # F2: net income, where a gross_profit line also exists — the Solver may grab the wrong one.
    def _gross_vs_net(self) -> Candidate:
        co, pd = self.rng.choice(self.COMPANIES), self.rng.choice(self.PERIODS)
        sql = self._amount_sql(co, pd, "net_income")
        return Candidate(
            question=f"What was {co}'s net income in {pd}?",
            gold_sql=sql, gold_value=self.wh.scalar(sql),
            category=Failure.F2_SEMANTIC, template_id="gross_vs_net",
            schema_region=frozenset({"financials.net_income", "financials.gross_profit"}),
            params={"company": co, "period": pd, "line_item": "net_income"})

    # F8: a period not present in the data → gold is the abstention label.
    def _unanswerable(self) -> Candidate:
        co = self.rng.choice(self.COMPANIES)
        sql = self._amount_sql(co, "FY2099", "revenue")     # FY2099 absent by construction
        return Candidate(
            question=f"What was {co}'s revenue in FY2099?",
            gold_sql=sql, gold_value=UNANSWERABLE,
            category=Failure.F8_ABSTENTION, template_id="unanswerable",
            schema_region=frozenset({"financials.revenue"}),
            params={"company": co, "period": "FY2099", "line_item": "revenue"})


class Oracle:
    """The strong reference that certifies a question is well-posed. Offline it recomputes the
    gold independently of gold_sql (via the metrics path), which is exactly the certification the
    production oracle provides with a reasoning model + gold-schema access."""

    def __init__(self, warehouse: Warehouse):
        self.wh = warehouse

    def solve(self, cand: Candidate) -> Tuple[Any, Optional[Any]]:
        """Return (oracle_value, independent_recompute). recompute is None when not applicable."""
        if cand.template_id == "growth_pct":
            co, p0, p1 = cand.params["company"], cand.params["p0"], cand.params["p1"]
            li = cand.params["line_item"]
            prev = self.wh.scalar(f"SELECT amount FROM financials WHERE company='{co}' "
                                  f"AND period='{p0}' AND line_item='{li}'")
            curr = self.wh.scalar(f"SELECT amount FROM financials WHERE company='{co}' "
                                  f"AND period='{p1}' AND line_item='{li}'")
            val = growth_pct(prev, curr)
            return val, val
        if cand.gold_value == UNANSWERABLE:
            # The oracle confirms the field is genuinely absent (else gold would be suspect).
            return UNANSWERABLE, None
        # For direct lookups the oracle agrees with the constructed gold.
        return cand.gold_value, None


class StubSolver:
    """A deterministic, deliberately-buggy production-Solver stand-in. Bugs are category-specific
    so the fair-hard filter has real failures to find; an instruction that 'fixes' a category is
    modelled by adding it to `fixed`, which is how the GateRunner demonstrates commit/revert."""

    def __init__(self, warehouse: Warehouse, fixed: Optional[set] = None):
        self.wh = warehouse
        self.fixed: set = set(fixed or set())
        self._last_category: Optional[Failure] = None

    def answer_candidate(self, cand: Candidate) -> Answer:
        """Offline entry point: the stub needs the category, which a real Solver infers from text."""
        cat = cand.category
        if cat in self.fixed:
            return self._correct(cand)
        return self._buggy(cand)

    def answer(self, question: str) -> Answer:  # Protocol conformance (real Solvers use this)
        raise NotImplementedError("StubSolver is offline-only; use answer_candidate")

    def _correct(self, cand: Candidate) -> Answer:
        if cand.gold_value == UNANSWERABLE:
            return Answer.abstain()
        return Answer(value=cand.gold_value, sql=cand.gold_sql)

    def _buggy(self, cand: Candidate) -> Answer:
        if cand.template_id == "line_item_lookup":
            return self._correct(cand)                       # baseline: already solved
        if cand.template_id == "growth_pct":
            return Answer(value=cand.gold_value / 100.0, sql=None)   # forgot the *100 (F6)
        if cand.template_id == "gross_vs_net":
            co, pd = cand.params["company"], cand.params["period"]
            sql = (f"SELECT amount FROM financials WHERE company='{co}' AND period='{pd}' "
                   f"AND line_item='gross_profit'")                   # wrong concept (F2)
            return Answer(value=self.wh.scalar(sql), sql=sql)
        if cand.template_id == "unanswerable":
            return Answer(value=0.0, sql=None)                        # fabricates instead of F8
        return Answer(value=None)


class GeminiSolver:
    """PRODUCTION SWAP (needs a backend): a Text-to-SQL Solver. Conforms to `Solver`. Drafts SQL
    with the fast model + a system prompt that accumulates the Judge's LEARNED GUIDANCE, then
    executes it through the warehouse. Left as the first real role to wire in."""

    def __init__(self, warehouse: Warehouse, instructions: Optional[List[str]] = None):
        self.wh = warehouse
        self.instructions = list(instructions or [])

    def answer(self, question: str) -> Answer:  # pragma: no cover - needs Gemini
        if not _HAS_CODER:
            raise RuntimeError("coding_agent_gemini (Gemini backend) not available")
        sys_prompt = ("You translate financial questions into a single SQLite SELECT over a table "
                      "`financials(company, period, statement, line_item, currency, amount)`. "
                      "Reply with ONLY the SQL, or the single word UNANSWERABLE if the data "
                      "cannot answer it.")
        if self.instructions:
            sys_prompt += "\n\nLEARNED GUIDANCE:\n" + "\n".join(f"- {i}" for i in self.instructions)
        from langchain_core.messages import HumanMessage, SystemMessage
        msg = llm("fast", reasoning=False).invoke(
            [SystemMessage(sys_prompt), HumanMessage(question)])
        text = content_text(msg).strip()
        if text.upper().startswith("UNANSWERABLE"):
            return Answer.abstain()
        sql = text.strip("`").removeprefix("sql").strip()
        try:
            return Answer(value=self.wh.scalar(sql), sql=sql)
        except sqlite3.Error:
            return Answer(value=None, sql=sql)


# ════════════════════════════════════════════════════════════════════════════
# Phase 5 — the fair-hard filter, the Setter reward, coverage tracking (§3)
# ════════════════════════════════════════════════════════════════════════════

class CoverageTracker:
    """Tracks realized categories of minted examples and reports per-category deficits vs. QUOTA."""

    def __init__(self):
        self.counts: Dict[Failure, int] = {c: 0 for c in QUOTA}

    def record(self, category: Failure) -> None:
        if category in self.counts:
            self.counts[category] += 1

    def deficit(self, category: Failure) -> float:
        total = sum(self.counts.values()) or 1
        observed = self.counts.get(category, 0) / total
        return QUOTA.get(category, 0.0) - observed


def run_filter(cand: Candidate, solver: StubSolver, oracle: Oracle, verifier: Verifier,
               seen_questions: set, coverage: CoverageTracker, k: int = 3) -> FilterVerdict:
    """Turn a candidate into a verdict via the three-way check (well-posed / informative / novel)
    plus the gold_suspect cross-check. `solver` is run k times for the difficulty + stability
    signals."""
    oracle_value, recompute = oracle.solve(cand)
    gold_suspect = verifier.gold_is_suspect(cand, recompute)
    well_posed = verifier._equal(oracle_value, cand.gold_value)

    runs = [verifier.value_of(solver.answer_candidate(cand)) for _ in range(k)]
    matches = [verifier._equal(v, cand.gold_value) for v in runs]
    solve_rate = sum(matches) / len(matches)
    non_det = len({_hashable(v) for v in runs}) > 1
    informative = solve_rate < 1.0

    novel = (cand.question not in seen_questions
             and coverage.deficit(cand.category) > -0.10)   # not far over its quota

    return FilterVerdict(well_posed=well_posed, informative=informative, novel=novel,
                         gold_suspect=gold_suspect, solve_rate=solve_rate,
                         oracle_value=oracle_value, production_value=runs[0],
                         non_deterministic=non_det)


def _hashable(v: Any):
    try:
        hash(v)
        return v
    except TypeError:
        return str(v)


def setter_reward(cand: Candidate, verdict: FilterVerdict, coverage: CoverageTracker,
                  target: float = 0.5, beta: float = 0.5, gamma: float = 0.2,
                  d_cap: int = 4) -> float:
    """The Setter's reward (design doc §3): rewards fair-hard, novel, on-quota questions in a
    difficulty BAND — never raw failure. Degenerate exploits (broken gold, impossible questions,
    duplicates) score <= 0."""
    if verdict.gold_suspect:
        return -1.0
    if not verdict.well_posed:
        return -0.5
    if not verdict.novel:
        return 0.0
    difficulty = 1.0 - abs(verdict.solve_rate - target)
    coverage_bonus = beta * max(0.0, coverage.deficit(cand.category))
    depth_bonus = gamma * min(len(cand.decomposition), d_cap) / d_cap
    return difficulty + coverage_bonus + depth_bonus


# ════════════════════════════════════════════════════════════════════════════
# Phase 6 — the Judge (offline rule-based; LLM judge is the production swap)
# ════════════════════════════════════════════════════════════════════════════

class RuleJudge:
    """Maps a fair-hard Example to a typed ImprovementProposal by routing on its category. The
    production Judge replaces this with an LLM that writes the rationale/payload/test_to_add, but
    the CONTRACT it emits is identical, so nothing downstream changes."""

    def diagnose(self, ex: Example) -> ImprovementProposal:
        cand, vd = ex.candidate, ex.verdict
        cat = cand.category
        ev = Evidence(task_id=cand.template_id, question=cand.question, gold_sql=cand.gold_sql,
                      gold_value=cand.gold_value, agent_sql=None,
                      agent_value=vd.production_value,
                      ex_match=False, runs=[vd.production_value])
        diag = Diagnosis(category=cat, gold_suspect=vd.gold_suspect,
                         non_deterministic=vd.non_deterministic,
                         root_cause=f"Solver fails {cat.value} on '{cand.question}'.",
                         evidence=[ev])
        target = EDIT_TIER[cat]
        action = "promote_to_tool" if cat == Failure.F6_CALCULATION else "add"
        instr = Instruction(
            target=target, action=action,
            rationale=f"Recurring {cat.value} failure on template {cand.template_id}.",
            payload=self._payload(cat, cand),
            scope=f"questions of category {cat.value} like {cand.template_id}",
            expected_effect=f"fixes {cat.value} on {cand.template_id} questions",
            risk="low",
            test_to_add=f"answer('{cand.question}') == {cand.gold_value!r}")
        return ImprovementProposal(proposal_id=uuid.uuid4().hex[:8], diagnosis=diag,
                                   instructions=[instr])

    @staticmethod
    def _payload(cat: Failure, cand: Candidate) -> str:
        return {
            Failure.F2_SEMANTIC: "Distinguish net income from gross profit; use the net_income "
                                 "line item when the question asks for net income.",
            Failure.F6_CALCULATION: "Promote a verified growth_pct(prev, curr) tool returning "
                                    "(curr-prev)/prev*100 and call it instead of inline arithmetic.",
            Failure.F8_ABSTENTION: "If the requested period/field is absent from the tables, "
                                   "answer UNANSWERABLE instead of fabricating a number.",
        }.get(cat, f"Address {cat.value} failures.")


# ════════════════════════════════════════════════════════════════════════════
# Phase 7 — the GateRunner (scratch → test → benchmark → commit/revert) (§7)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GateOutcome:
    committed: bool
    reason: str
    test_passed: bool
    regressions: int


class GateRunner:
    """Applies ONE instruction to a scratch copy of the Solver, proves it on the new test, checks
    no regression on the frozen benchmark, and commits or reverts. Offline, 'apply' = adding the
    category to the Solver's `fixed` set; the production runner routes prompt-tier instructions to
    CodingAgent.add_instruction and code/tool-tier ones to CodingAgent.generate(test_code=...)."""

    def __init__(self, warehouse: Warehouse, oracle: Oracle, verifier: Verifier):
        self.wh = warehouse
        self.oracle = oracle
        self.verifier = verifier

    def apply(self, solver: StubSolver, proposal: ImprovementProposal,
              target_example: Example, benchmark: Sequence[Example]) -> GateOutcome:
        instr = proposal.instructions[0]
        if instr.test_to_add is None:
            return GateOutcome(False, "no test_to_add (not a falsifiable fix)", False, 0)

        cat = proposal.diagnosis.category
        scratch = StubSolver(self.wh, fixed=solver.fixed | {cat})   # isolated scratch copy

        test_passed = self._solves(scratch, target_example)
        if not test_passed:
            return GateOutcome(False, "test_to_add still fails on scratch", False, 0)

        regressions = sum(1 for ex in benchmark if not self._solves(scratch, ex))
        if regressions:
            return GateOutcome(False, f"{regressions} regression(s) on benchmark",
                               True, regressions)

        solver.fixed.add(cat)               # COMMIT
        log.info(f"[gate] COMMIT {instr.action} {instr.target} for {cat.value} "
                 f"(proposal {proposal.proposal_id})")
        return GateOutcome(True, "committed", True, 0)

    def _solves(self, solver: StubSolver, ex: Example) -> bool:
        ans = solver.answer_candidate(ex.candidate)
        return self.verifier.ex_match(ans, ex.candidate.gold_value)


# ════════════════════════════════════════════════════════════════════════════
# Phase 8 — the Trainer loop
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainerStats:
    proposed: int = 0
    minted: int = 0
    committed: int = 0
    rejected: int = 0
    rewards: List[float] = field(default_factory=list)


class Trainer:
    """Wires the roles into the asymmetric self-play loop and accumulates the manufactured eval
    set (`benchmark`). Each committed change freezes its example into the no-regression guard."""

    def __init__(self, warehouse: Optional[Warehouse] = None, seed: int = 0):
        self.wh = warehouse or SQLiteWarehouse()
        self.setter = TemplateSetter(self.wh, seed=seed)
        self.oracle = Oracle(self.wh)
        self.verifier = Verifier(self.wh)
        self.solver = StubSolver(self.wh)
        self.judge = RuleJudge()
        self.gate = GateRunner(self.wh, self.oracle, self.verifier)
        self.coverage = CoverageTracker()
        self.benchmark: List[Example] = []
        self.seen: set = set()
        self.stats = TrainerStats()

    def step(self, iteration: int) -> Optional[Example]:
        """One self-play iteration: propose → filter → (mint, judge, gate) on a fair-hard hit."""
        cand = self.setter.propose()
        self.stats.proposed += 1
        verdict = run_filter(cand, self.solver, self.oracle, self.verifier,
                             self.seen, self.coverage)
        self.stats.rewards.append(setter_reward(cand, verdict, self.coverage))
        self.seen.add(cand.question)

        if not verdict.fair_hard:
            return None

        ex = Example(candidate=cand, verdict=verdict, minted_at=iteration)
        self.coverage.record(cand.category)
        self.stats.minted += 1
        log.info(f"[mint] fair-hard {cand.category.value}: {cand.question!r} "
                 f"(solve_rate={verdict.solve_rate:.2f})")

        proposal = self.judge.diagnose(ex)
        outcome = self.gate.apply(self.solver, proposal, ex, self.benchmark)
        if outcome.committed:
            self.stats.committed += 1
            ex.weight = 0.0                       # solved → keep only as a regression guard
            self.benchmark.append(ex)
        else:
            self.stats.rejected += 1
            log.info(f"[gate] REJECT: {outcome.reason}")
        return ex

    def run(self, iterations: int = 40) -> TrainerStats:
        for i in range(1, iterations + 1):
            self.step(i)
        avg_r = sum(self.stats.rewards) / max(1, len(self.stats.rewards))
        log.info(f"[done] proposed={self.stats.proposed} minted={self.stats.minted} "
                 f"committed={self.stats.committed} rejected={self.stats.rejected} "
                 f"avg_reward={avg_r:.3f} benchmark={len(self.benchmark)}")
        return self.stats


# ════════════════════════════════════════════════════════════════════════════
# Offline self-tests (no backend) + CLI
# ════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    results: List[Tuple[str, bool]] = []

    def check(name, cond):
        results.append((name, bool(cond)))
        print(("  PASS " if cond else "  FAIL ") + name)

    print("offline self-tests (no model calls):")
    wh = SQLiteWarehouse()
    ver = Verifier(wh)
    oracle = Oracle(wh)
    cov = CoverageTracker()

    # warehouse + metrics
    check("warehouse scalar", wh.scalar("SELECT amount FROM financials WHERE company='ACME' "
                                        "AND period='FY2022' AND line_item='revenue'") == 1000.0)
    check("growth_pct metric", math.isclose(growth_pct(1000, 1200), 20.0))
    check("gross != net in data", wh.scalar("SELECT amount FROM financials WHERE company='ACME' "
          "AND period='FY2022' AND line_item='gross_profit'") != wh.scalar(
          "SELECT amount FROM financials WHERE company='ACME' AND period='FY2022' "
          "AND line_item='net_income'"))

    # verifier tolerance + abstention
    check("verifier numeric tolerance", ver._equal(20.0, 20.0 + 1e-9))
    check("verifier abstention match", ver._equal(UNANSWERABLE, UNANSWERABLE))
    check("verifier abstention vs number", not ver._equal(UNANSWERABLE, 0.0))

    # filter: a buggy growth_pct candidate must come out fair-hard
    setter = TemplateSetter(wh, seed=1)
    buggy = StubSolver(wh)                       # nothing fixed
    g = setter._growth_pct()
    vd = run_filter(g, buggy, oracle, ver, set(), cov)
    check("growth_pct is well-posed", vd.well_posed)
    check("growth_pct is informative (solver wrong)", vd.informative)
    check("growth_pct fair-hard", vd.fair_hard)

    # a solved candidate is NOT informative (kept as regression guard, not trained on)
    lookup = setter._line_item_lookup()
    vd2 = run_filter(lookup, buggy, oracle, ver, set(), cov)
    check("solved lookup not informative", not vd2.informative and not vd2.fair_hard)

    # abstention: stub fabricates → fair-hard; a fixed solver abstains → solved
    una = setter._unanswerable()
    vd3 = run_filter(una, buggy, oracle, ver, set(), cov)
    check("unanswerable fair-hard when fabricated", vd3.fair_hard)
    fixed = StubSolver(wh, fixed={Failure.F8_ABSTENTION})
    check("fixed solver abstains correctly",
          ver.ex_match(fixed.answer_candidate(una), una.gold_value))

    # reward shape: fair-hard near the band scores positive; impossible/dup score <= 0
    r_fair = setter_reward(g, vd, cov)
    check("reward positive for fair-hard", r_fair > 0)
    bad = FilterVerdict(well_posed=False, informative=True, novel=True, gold_suspect=False,
                        solve_rate=0.0)
    check("reward penalizes ill-posed", setter_reward(g, bad, cov) < 0)
    dup = FilterVerdict(well_posed=True, informative=True, novel=False, gold_suspect=False,
                        solve_rate=0.5)
    check("reward zero for duplicate", setter_reward(g, dup, cov) == 0.0)

    # gate: commit a fix, then the same example must no longer be a failure
    solver = StubSolver(wh)
    gate = GateRunner(wh, oracle, ver)
    ex = Example(candidate=g, verdict=vd)
    prop = RuleJudge().diagnose(ex)
    out = gate.apply(solver, prop, ex, benchmark=[])
    check("gate commits a good fix", out.committed)
    check("gate mutated the solver", Failure.F6_CALCULATION in solver.fixed)
    check("post-commit example solved",
          ver.ex_match(solver.answer_candidate(g), g.gold_value))

    # end-to-end loop runs and improves
    t = Trainer(seed=7)
    stats = t.run(iterations=60)
    check("loop minted fair-hard examples", stats.minted > 0)
    check("loop committed improvements", stats.committed > 0)
    check("benchmark grew", len(t.benchmark) > 0)

    ok = all(c for _, c in results)
    print(f"\n{sum(c for _, c in results)}/{len(results)} passed — "
          + ("ALL GREEN" if ok else "FAILURES ABOVE"))
    return ok


def _demo():
    """Run the self-play loop on SQLite and print the trajectory (offline, no backend)."""
    t = Trainer(seed=3)
    t.run(iterations=50)
    print("\ncoverage (minted per category):")
    for cat, n in t.coverage.counts.items():
        if n:
            print(f"  {cat.value:18s} {n}  (deficit {t.coverage.deficit(cat):+.3f})")
    print(f"\nsolver now fixes: {sorted(c.value for c in t.solver.fixed)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Self-improving financial Text-to-SQL trainer.")
    p.add_argument("--selftest", action="store_true", help="offline checks (no backend)")
    p.add_argument("--demo", action="store_true", help="run the self-play loop on SQLite")
    p.add_argument("--iterations", type=int, default=50)
    args = p.parse_args(argv)

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.demo:
        Trainer(seed=3).run(iterations=args.iterations)
        return
    p.error("nothing to do: use --selftest or --demo")


if __name__ == "__main__":
    main()
