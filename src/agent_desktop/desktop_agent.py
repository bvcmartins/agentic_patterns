#!/usr/bin/env python3
"""
desktop_agent.py — a general-assistance agent (Claude-Code-style) on LangGraph + Gemini 3.0.

This is a single-file port of `code_assistant/claude_code_from_scratch_v3.ipynb`, with two
deliberate changes:

  1. BACKEND. v3 talked to a local Ollama (qwen3) through ChatOllama. This runs on **Google
     Cloud** against **Gemini 3.0** — Vertex AI by default (Application-Default Credentials,
     no API key), or the Gemini Developer API via `LLM_BACKEND=genai`. The model factory is
     the *only* swappable seam; every graph below is backend-agnostic.

  2. PROBLEM SHAPE. v3 was a *coding* agent: its centre of gravity is a five-subagent team
     that drives an implementer<->tester loop until a pytest contract goes green — i.e. it
     assumes the task is **convergent** (one verifiable right answer). A general desktop
     assistant is not so lucky: most requests are divergent (designs, recommendations),
     exploratory (research, open questions) or structural (multi-step builds). So the default
     entry point here is a **bounded-window ReAct assistant** with the full toolset, and the
     adaptive router keeps v3's *non-convergent* strategies (asymmetric_solve / wide_pass /
     decompose) instead of forcing self-consistency voting on everything. The convergent
     machinery (spec layer, code-with-tests, the team graph) is kept as an *opt-in* capability,
     not the spine.

------------------------------------------------------------------------------------------
RUNNING IT (in a Google Cloud environment — NOT designed for this laptop)
------------------------------------------------------------------------------------------
  pip install -U langgraph langchain-core langchain-google-vertexai pydantic rich
  #   (or, for the Developer API backend:  langchain-google-genai)

  # Vertex AI (default backend): authenticate once with ADC, then point at a project.
  gcloud auth application-default login          # or run on a VM/Cloud Run with a SA
  export GOOGLE_CLOUD_PROJECT=my-project
  export GOOGLE_CLOUD_LOCATION=us-central1       # any region that serves Gemini 3.0

  # Gemini Developer API backend instead of Vertex:
  #   export LLM_BACKEND=genai ; export GOOGLE_API_KEY=...

  python desktop_agent.py "summarise the files in ./ and propose a refactor"   # one-shot
  python desktop_agent.py -i                                                   # REPL
  python desktop_agent.py --think "should we use Postgres or DynamoDB here?"   # reason-only
  python desktop_agent.py --health                                            # probe backend
  python desktop_agent.py --team --file solution.py --task "write fizzbuzz(n)" # convergent team

Every knob is an env var (see CONFIG). Model defaults are Gemini 3.0; override the per-role
model IDs to mix in a cheaper flash for the routers/summariser.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
import py_compile
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict
import glob as _glob

from pydantic import BaseModel, Field

# --- LangChain / LangGraph ---------------------------------------------------
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                      SystemMessage, ToolMessage)
from langchain_core.tools import tool, StructuredTool
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


# ════════════════════════════════════════════════════════════════════════════
# Phase 0 — logging, config, the model factory (the one backend seam)
# ════════════════════════════════════════════════════════════════════════════

AGENT_LOG_LEVEL = os.environ.get("AGENT_LOG_LEVEL", "INFO").upper()


class _Fmt(logging.Formatter):
    COLORS = {"DEBUG": "\033[90m", "INFO": "\033[36m", "WARNING": "\033[33m",
              "ERROR": "\033[31m", "CRITICAL": "\033[41m"}
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        short = record.name.split(".", 1)[1] if "." in record.name else record.name
        return f"{color}[{record.levelname:5s}] {short:16s} | {record.getMessage()}{self.RESET}"


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_Fmt())
log = logging.getLogger("agentd")
log.handlers.clear()
log.addHandler(_handler)
log.setLevel(AGENT_LOG_LEVEL)
log.propagate = False

log_llm = logging.getLogger("agentd.llm")
log_tool = logging.getLogger("agentd.tool")
log_graph = logging.getLogger("agentd.graph")
log_sub = logging.getLogger("agentd.subagent")

# --- Backend / Gemini config -------------------------------------------------
# "vertex" (default) -> ChatVertexAI on Google Cloud via ADC; "genai" -> Developer API key.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "vertex").lower()
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
GCP_LOCATION = (os.environ.get("GOOGLE_CLOUD_LOCATION")
                or os.environ.get("VERTEXAI_LOCATION") or "us-central1")

# Default everything to Gemini 3.0. Override per-role to slot a cheaper flash into the
# routers/summariser without touching the reasoning model.
DEFAULT_MODEL = os.environ.get("AGENTD_MODEL", "gemini-3-pro-preview")
MODELS = {
    "reasoning":  os.environ.get("AGENTD_REASONING_MODEL",  DEFAULT_MODEL),
    "fast":       os.environ.get("AGENTD_FAST_MODEL",       DEFAULT_MODEL),
    "summarizer": os.environ.get("AGENTD_SUMMARIZER_MODEL", DEFAULT_MODEL),
}

# --- Sandbox workspace -------------------------------------------------------
WORKSPACE = Path(os.environ.get("AGENTD_WORKSPACE",
                                str(Path.cwd() / "desktop_workspace"))).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
AGENT_CODE_DIR = WORKSPACE / "agent_code"
AGENT_CODE_DIR.mkdir(exist_ok=True)
DB_PATH = WORKSPACE / "dag.db"

# --- Long-term memory backend ------------------------------------------------
# Where the bi-temporal *semantic* memory (facts/recall) lives. This is NOT the LangGraph
# checkpointer (turn-by-turn graph state) — see RagManagedMemory's docstring for why those two
# want different stores. "memory" = in-process (default); "rag" = Vertex AI RAG Engine
# (RagManagedDb), a managed vector store that gives real semantic recall.
MEMORY_BACKEND = os.environ.get("AGENTD_MEMORY", "memory").lower()
RAG_PROJECT = os.environ.get("AGENTD_RAG_PROJECT") or GCP_PROJECT
RAG_LOCATION = os.environ.get("AGENTD_RAG_LOCATION") or GCP_LOCATION
RAG_CORPUS_DISPLAY = os.environ.get("AGENTD_RAG_CORPUS", "agent-memory")
# Optional: reuse an existing corpus directly by full resource name
# (projects/.../locations/.../ragCorpora/123). If unset we find-or-create by display name.
RAG_CORPUS_NAME = os.environ.get("AGENTD_RAG_CORPUS_NAME") or None
RAG_EMBEDDING_MODEL = os.environ.get(
    "AGENTD_RAG_EMBEDDING", "publishers/google/models/text-embedding-005")

# --- Limits ------------------------------------------------------------------
MAX_TOOL_OUTPUT = 12_000
BASH_TIMEOUT_S = 60
TEST_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 600
MAX_ITERATIONS = 20

BASH_BLOCKLIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "> /dev/", ":(){ :|:& };:"]
_HAS_PYTEST = importlib.util.find_spec("pytest") is not None


def _build_model(model: str, thinking: Optional[bool], temperature: float,
                 max_tokens: Optional[int]):
    """Construct one chat model. The ONLY place a Gemini client is built.

    `thinking` maps to Gemini's thinking budget: True -> dynamic (-1, thoughts returned),
    False -> off (0, best for structured/JSON routing), None -> backend default. Params are
    passed defensively (Gemini SDK versions vary), falling back to a plain client.
    """
    if LLM_BACKEND == "genai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kw: Dict[str, Any] = dict(model=model, temperature=temperature,
                                  timeout=REQUEST_TIMEOUT_S)
        if max_tokens:
            kw["max_output_tokens"] = max_tokens
        if thinking is not None:
            kw["thinking_budget"] = -1 if thinking else 0
            kw["include_thoughts"] = bool(thinking)
        try:
            return ChatGoogleGenerativeAI(**kw)
        except TypeError:
            for k in ("thinking_budget", "include_thoughts", "timeout"):
                kw.pop(k, None)
            return ChatGoogleGenerativeAI(**kw)

    # default: Vertex AI on Google Cloud (ADC; project+location).
    from langchain_google_vertexai import ChatVertexAI
    kw = dict(model=model, temperature=temperature, project=GCP_PROJECT,
              location=GCP_LOCATION, max_retries=2)
    if max_tokens:
        kw["max_output_tokens"] = max_tokens
    if thinking is not None:
        kw["thinking_budget"] = -1 if thinking else 0
    try:
        return ChatVertexAI(**kw)
    except TypeError:
        kw.pop("thinking_budget", None)
        return ChatVertexAI(**kw)


@lru_cache(maxsize=32)
def _client(model: str, thinking: Optional[bool], temperature: float,
            num_predict: Optional[int]):
    return _build_model(model, thinking, temperature, num_predict)


def llm(role: str = "fast", *, reasoning: Optional[bool] = True,
        temperature: float = 0.2, max_tokens: Optional[int] = None):
    """Get a chat model by ROLE ('reasoning' | 'fast' | 'summarizer') or an explicit model id.

    `reasoning` is kept as the public knob name (so the v3 graphs read identically); it drives
    Gemini's thinking budget under the hood.
    """
    model = MODELS.get(role, role)
    return _client(model, reasoning, temperature, max_tokens)


def backend_healthcheck() -> bool:
    """Cheap probe: build a client and do a 1-token round-trip so a bad ADC/project fails fast."""
    log_llm.info(f"backend={LLM_BACKEND} project={GCP_PROJECT} location={GCP_LOCATION}")
    log_llm.info(f"models={MODELS}")
    if LLM_BACKEND == "vertex" and not GCP_PROJECT:
        log_llm.error("GOOGLE_CLOUD_PROJECT is unset — Vertex AI needs a project.")
        return False
    try:
        m = llm("fast", reasoning=False, temperature=0.0, max_tokens=8)
        r = m.invoke([HumanMessage("reply with the single word: ok")])
        log_llm.info(f"healthcheck OK — got {content_text(r)[:40]!r}")
        return True
    except Exception as e:
        log_llm.error(f"healthcheck FAILED: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# Phase 0.5 — observability: a LangChain callback handler + stream pretty-printer
# ════════════════════════════════════════════════════════════════════════════

TRACE_LEVEL = os.environ.get("AGENT_TRACE", "full").lower()
TRACE_PREVIEW = int(os.environ.get("AGENT_TRACE_PREVIEW", "1600"))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    _HAS_RICH = True
    _console = Console(highlight=False, soft_wrap=True)
except Exception:
    _HAS_RICH = False
    _console = None


def _clip(s, n=None):
    s = "" if s is None else str(s)
    n = n or TRACE_PREVIEW
    return s if len(s) <= n else s[:n] + f"\n... [+{len(s) - n} chars]"


def content_text(msg) -> str:
    """Flatten a message's content to plain answer text, dropping Gemini 'thought' blocks.

    Gemini may return content as a list of typed parts; this keeps only the visible answer.
    """
    c = msg.content if isinstance(msg, BaseMessage) else msg
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                if b.get("type") in ("thinking", "reasoning") or b.get("thought"):
                    continue
                out.append(b.get("text", "") or "")
        return "".join(out)
    return str(c or "")


def thinking_of(msg: BaseMessage) -> str:
    """Gemini's reasoning channel, wherever the integration surfaces it."""
    ak = getattr(msg, "additional_kwargs", {}) or {}
    t = ak.get("reasoning_content") or ak.get("thinking") or ak.get("thought") or ""
    if t:
        return t
    c = getattr(msg, "content", None)
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict) and (b.get("type") in ("thinking", "reasoning")
                                        or b.get("thought")):
                parts.append(b.get("text") or b.get("thinking") or "")
        return "\n".join(p for p in parts if p)
    return ""


class RichTracer(BaseCallbackHandler):
    """Narrates every model/tool call. Works at any graph depth because LangChain drives it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens = 0
        self.tool_calls = 0
        self._starts: Dict[Any, float] = {}

    @property
    def on(self):
        return TRACE_LEVEL != "off"

    @property
    def full(self):
        return TRACE_LEVEL == "full"

    def _emit(self, renderable, plain):
        if not self.on:
            return
        if _HAS_RICH and renderable is not None:
            _console.print(renderable)
        else:
            log_llm.info(plain)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kw):
        with self._lock:
            self.calls += 1
            n = self.calls
        self._starts[run_id] = time.time()
        if not self.on:
            return
        last = messages[0][-1] if messages and messages[0] else None
        tail = content_text(last) if last else ""
        self._emit(Text(f">> #{n} model call", style="bold cyan") if _HAS_RICH else None,
                   f">> #{n} model call")
        if self.full and tail:
            self._emit(Panel(Text(_clip(tail)), title=f"#{n} prompt tail",
                             title_align="left", border_style="cyan") if _HAS_RICH else None,
                       f"   prompt: {_clip(tail, 200)}")

    def on_llm_end(self, response, *, run_id=None, **kw):
        dt = time.time() - self._starts.pop(run_id, time.time())
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = content_text(msg) if msg else getattr(gen, "text", "")
            think = thinking_of(msg) if msg else ""
            usage = (getattr(msg, "usage_metadata", None) or {}) if msg else {}
            tok = usage.get("output_tokens", 0) or 0
            tcs = getattr(msg, "tool_calls", []) if msg else []
        except Exception:
            text, think, tok, tcs = "", "", 0, []
        with self._lock:
            self.tokens += tok
        if not self.on:
            return
        if self.full and think:
            self._emit(Panel(Text(_clip(think), style="italic"), title="thinking",
                             title_align="left", border_style="grey50") if _HAS_RICH else None,
                       f"[think] {_clip(think, 300)}")
        if text:
            self._emit(Panel(Text(_clip(text)), title=f"answer ({dt:.1f}s, {tok} tok)",
                             title_align="left", border_style="green") if _HAS_RICH else None,
                       f"[answer] {_clip(text, 300)}")
        for tc in tcs or []:
            nm = tc.get("name", "?")
            self._emit(Text(f"-> wants tool: {nm}("
                            f"{_clip(json.dumps(tc.get('args', {}), default=str), 160)})",
                            style="bold yellow") if _HAS_RICH else None, f"-> tool {nm}")

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kw):
        with self._lock:
            self.tool_calls += 1
        name = (serialized or {}).get("name", "tool")
        self._emit(Panel(Text(_clip(input_str, 800)), title=f"tool: {name} (args)",
                         title_align="left", border_style="yellow") if _HAS_RICH else None,
                   f"[tool] {name}({_clip(input_str, 120)})")

    def on_tool_end(self, output, *, run_id=None, **kw):
        text = str(getattr(output, "content", output))
        err = text.lstrip().lower().startswith(("error", "[error", "reverted", "traceback"))
        self._emit(Panel(Text(_clip(text)), title="tool result", title_align="left",
                         border_style="red" if err else "green") if _HAS_RICH else None,
                   ("[fail] " if err else "[ok] ") + _clip(text, 200))

    def event(self, title, body="", style="bold blue"):
        if not self.on:
            return
        if _HAS_RICH:
            t = Text(title, style=style)
            if body:
                t.append("\n" + _clip(body))
            _console.print(Panel(t, border_style="blue", title_align="left"))
        else:
            log.info(f"[event] {title} -- {_clip(body, 200)}")

    def summary(self):
        if _HAS_RICH:
            t = Table(title="Trace summary")
            t.add_column("metric")
            t.add_column("value", justify="right")
            t.add_row("model calls", str(self.calls))
            t.add_row("output tokens", str(self.tokens))
            t.add_row("tool calls", str(self.tool_calls))
            _console.print(t)
        else:
            log.info(f"TRACE: {self.calls} calls, {self.tokens} tok, {self.tool_calls} tools")


tracer = RichTracer()
CB = {"callbacks": [tracer]}


def run_config(label: str = "run", **extra) -> dict:
    cfg = {"callbacks": [tracer], "configurable": {"thread_id": f"{label}-{uuid.uuid4().hex[:6]}"}}
    cfg.update(extra)
    return cfg


def stream_run(app, inputs, config=None, *, mode: str = "updates"):
    """Run a graph and print each node update as it streams. Returns the final node update."""
    config = config or run_config("stream")
    final = None
    for chunk in app.stream(inputs, config=config, stream_mode=mode):
        for node, update in chunk.items():
            msgs = (update or {}).get("messages") if isinstance(update, dict) else None
            tail = ""
            if msgs:
                last = msgs[-1]
                tail = ("[+think] " if thinking_of(last) else "") + _clip(content_text(last), 200)
            log_graph.info(f"· node «{node}» -> {tail or list((update or {}).keys())}")
            final = update
    return final


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — cognitive substrate: thinking, structured routing, test-time compute
#
# Generalised away from "senior software engineer / verifiable code only" toward a broad
# assistant. The router keeps v3's NON-convergent strategies, since most desktop requests
# do not have a single checkable answer.
# ════════════════════════════════════════════════════════════════════════════

GENERAL_SYSTEM_PROMPT = (
    "You are a capable, careful general assistant — like Claude Code, but not limited to code. "
    "You help with engineering, analysis, writing, research and planning. You think before you "
    "act, and you name your assumptions before committing to them.\n\n"
    "PRINCIPLES:\n"
    "1. Prefer evidence over assertion: when a claim is checkable, check it (run code, read the "
    "file, test it) rather than guessing.\n"
    "2. Many tasks are open-ended. When a request is divergent (designs, recommendations) or "
    "exploratory (research, trade-offs), give a well-reasoned answer and surface the genuine "
    "alternatives and their trade-offs — do not pretend there is one right answer.\n"
    "3. If you do not know something and cannot find out with your tools, say so plainly.\n"
    "4. Use tools to investigate and to act; keep changes reversible where you can.\n"
    "5. Be concise and direct. Lead with the answer, then the reasoning."
)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def split_think(msg) -> tuple[str, str]:
    """Return (thinking, answer) for an AIMessage or a raw string."""
    if isinstance(msg, BaseMessage):
        think = thinking_of(msg)
        content = content_text(msg)
        if think:
            return think, strip_think(content)
        m = _THINK_RE.search(content)
        return (m.group(1).strip() if m else ""), strip_think(content)
    text = msg or ""
    m = _THINK_RE.search(text)
    return (m.group(1).strip() if m else ""), strip_think(text)


@dataclass
class ThoughtfulResponse:
    thinking: str
    answer: str
    output_tokens: int


def think_then_answer(query: str, role: str = "fast", temperature: float = 0.3,
                      max_tokens: int = 2048,
                      system: str = GENERAL_SYSTEM_PROMPT) -> ThoughtfulResponse:
    """One free-text call with the thinking channel separated out."""
    model = llm(role, reasoning=True, temperature=temperature, max_tokens=max_tokens)
    msg = model.invoke([SystemMessage(system), HumanMessage(query)], config=CB)
    thinking, answer = split_think(msg)
    usage = getattr(msg, "usage_metadata", None) or {}
    return ThoughtfulResponse(thinking=thinking, answer=answer,
                              output_tokens=usage.get("output_tokens", 0))


# --- structured routing ------------------------------------------------------
THINKING_BUDGETS = {"trivial": 256, "easy": 512, "medium": 1500, "hard": 3000, "extreme": 6000}
PROBLEM_TYPES = ["convergent", "divergent", "exploratory", "structural"]
TYPE_STRATEGY = {"convergent": "self_consistency", "divergent": "asymmetric_solve",
                 "exploratory": "wide_pass", "structural": "decompose"}


class Difficulty(BaseModel):
    difficulty: Literal["trivial", "easy", "medium", "hard", "extreme"]


class ProblemKind(BaseModel):
    type: Literal["convergent", "divergent", "exploratory", "structural"]
    reason: str = Field(description="one sentence")


def estimate_difficulty(query: str) -> str:
    model = llm("fast", reasoning=False, temperature=0.0).with_structured_output(Difficulty)
    try:
        return model.invoke([SystemMessage("Classify the difficulty of the task."),
                             HumanMessage(query)], config=CB).difficulty
    except Exception:
        return "medium"


def classify_problem(query: str) -> dict:
    sys_p = ("Classify the KIND of problem (not its difficulty).\n"
             "convergent  = one correct/defensible answer (a fact, a calculation, a fix).\n"
             "divergent   = many valid answers (designs, strategies, recommendations).\n"
             "exploratory = open-ended / under-specified (research, implications).\n"
             "structural  = needs decomposition into parts before answering (multi-step builds).")
    model = llm("fast", reasoning=False, temperature=0.0).with_structured_output(ProblemKind)
    try:
        r = model.invoke([SystemMessage(sys_p), HumanMessage(query)], config=CB)
        return {"type": r.type, "reason": r.reason}
    except Exception:
        # For a general assistant, when in doubt assume the problem is OPEN, not convergent.
        return {"type": "exploratory", "reason": "fallback (classifier failed)"}


# --- test-time compute -------------------------------------------------------
class Verdict(BaseModel):
    score: int = Field(ge=1, le=10)
    reason: str


class Ranking(BaseModel):
    best_index: int
    reason: str


def self_consistency(query: str, k: int = 3, role: str = "fast") -> dict:
    """For CONVERGENT problems only: sample k times, take the modal answer."""
    model = llm(role, reasoning=True, temperature=0.7, max_tokens=1500)
    batch = [[SystemMessage(GENERAL_SYSTEM_PROMPT), HumanMessage(query)] for _ in range(k)]
    outs = model.batch(batch, config=CB)
    samples = [strip_think(content_text(m)) for m in outs]
    keys = [s[:60].lower() for s in samples]
    winner_key, votes = Counter(keys).most_common(1)[0]
    winner = next(s for s in samples if s[:60].lower() == winner_key)
    tracer.event(f"self-consistency: {votes}/{k} agree ({votes / k:.0%})", winner[:200])
    return {"winner": winner, "votes": votes, "k": k, "agreement": votes / k, "all_samples": samples}


def verifier_score(question: str, candidate: str, role: str = "reasoning") -> dict:
    model = llm(role, reasoning=False, temperature=0.0).with_structured_output(Verdict)
    try:
        v = model.invoke([SystemMessage("You are a careful verifier. Score correctness and "
                                        "soundness, not style, 1-10."),
                          HumanMessage(f"QUESTION:\n{question}\n\nCANDIDATE:\n{candidate}")],
                         config=CB)
        out = {"score": v.score, "reason": v.reason}
    except Exception as e:
        out = {"score": 0, "reason": f"verifier error: {e}"}
    tracer.event(f"verifier score: {out['score']}/10", out["reason"])
    return out


def asymmetric_solve(query: str, n_candidates: int = 3) -> dict:
    """For DIVERGENT problems: generate diverse candidates, then a stronger model ranks them."""
    gen = llm("fast", reasoning=True, temperature=0.7, max_tokens=1500)
    cands = [strip_think(content_text(m)) for m in
             gen.batch([[HumanMessage(query)] for _ in range(n_candidates)], config=CB)]
    ranker = llm("reasoning", reasoning=False).with_structured_output(Ranking)
    listing = "\n\n".join(f"CANDIDATE {i}:\n{c}" for i, c in enumerate(cands))
    try:
        r = ranker.invoke([HumanMessage("Pick the single best candidate.\n\n" + listing)], config=CB)
        idx = max(0, min(r.best_index, n_candidates - 1))
        tracer.event(f"asymmetry: picked #{idx} of {n_candidates}", r.reason)
        return {"winner": cands[idx], "winner_reason": r.reason, "candidates": cands}
    except Exception:
        return {"winner": cands[0], "winner_reason": "fallback", "candidates": cands}


def adaptive_think(query: str, route: bool = True) -> dict:
    """Route difficulty->thinking budget and problem-kind->strategy, then answer.

    The strategy map intentionally favours non-convergent handling; self_consistency is used
    ONLY when the router is confident the task is convergent.
    """
    difficulty = estimate_difficulty(query)
    budget = THINKING_BUDGETS.get(difficulty, 1500)
    problem = classify_problem(query) if route else {"type": "exploratory", "reason": "routing off"}
    strategy = TYPE_STRATEGY.get(problem["type"], "wide_pass") if route else "single_pass"
    log.info(f"[adaptive] difficulty={difficulty} type={problem['type']} -> budget {budget}, {strategy}")
    tracer.event(f"adaptive route: {difficulty} / {problem['type']} -> {strategy}",
                 f"budget={budget}; {problem['reason']}")
    if strategy == "self_consistency":
        r = self_consistency(query, k=3)
        answer, extra = r["winner"], {"agreement": r["agreement"]}
    elif strategy == "asymmetric_solve":
        r = asymmetric_solve(query, 3)
        answer, extra = r["winner"], {"winner_reason": r["winner_reason"]}
    elif strategy == "decompose":
        r = think_then_answer("Break this into parts, solve each, then assemble:\n\n" + query,
                              max_tokens=budget)
        answer, extra = r.answer, {"tokens": r.output_tokens}
    elif strategy == "wide_pass":
        r = think_then_answer(query, temperature=0.7, max_tokens=max(budget, 3000))
        answer, extra = r.answer, {"tokens": r.output_tokens}
    else:
        r = think_then_answer(query, max_tokens=budget)
        answer, extra = r.answer, {"tokens": r.output_tokens}
    return {"difficulty": difficulty, "type": problem["type"], "budget": budget,
            "strategy": strategy, "answer": answer, **extra}


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — tools: @tool-decorated, sandboxed to the workspace
# ════════════════════════════════════════════════════════════════════════════

SNAPSHOTS: Dict[str, Optional[str]] = {}


def _safe_path(path: str) -> Path:
    p = (WORKSPACE / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    try:
        p.relative_to(WORKSPACE)
    except ValueError:
        raise ValueError(f"path escapes WORKSPACE: {p}")
    return p


def _truncate(s: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    return s if len(s) <= limit else s[:limit] + f"\n... [truncated {len(s) - limit} chars]"


@tool
def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read a text file (relative to the workspace). Optional 1-indexed line range."""
    try:
        lines = _safe_path(path).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        i0 = max(0, (start_line or 1) - 1)
        i1 = end_line if end_line is not None else len(lines)
        return _truncate("".join(f"{i0 + 1 + i:5d}\t{ln}" for i, ln in enumerate(lines[i0:i1])) or "(empty)")
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (snapshotted first; use revert_file to undo)."""
    try:
        p = _safe_path(path)
        SNAPSHOTS[str(p)] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        action = "updated" if SNAPSHOTS[str(p)] is not None else "created"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        log_tool.info(f"[write] {action} {p}")
        return f"{action}: {p} (snapshot saved -- use revert_file to undo)"
    except Exception as e:
        return f"Error: {e}"


@tool
def revert_file(path: str) -> str:
    """Undo the last write to a file from its snapshot."""
    try:
        p = _safe_path(path)
        key = str(p)
        if key not in SNAPSHOTS:
            return f"Error: no snapshot for {p}"
        prev = SNAPSHOTS.pop(key)
        if prev is None:
            p.unlink(missing_ok=True)
            return f"reverted: deleted {p} (was new)"
        p.write_text(prev, encoding="utf-8")
        return f"reverted: {p}"
    except Exception as e:
        return f"Error: {e}"


@tool
def grep(pattern: str, path: str = ".", recursive: bool = True) -> str:
    """Regex search across files under a path (relative to the workspace)."""
    try:
        p = _safe_path(path)
        flags = ["-rn"] if recursive else ["-n"]
        proc = subprocess.run(["grep", *flags, pattern, str(p)],
                              capture_output=True, text=True, timeout=30)
        return _truncate((proc.stdout + proc.stderr).strip() or "(no matches)", 8000)
    except Exception as e:
        return f"Error: {e}"


@tool
def glob_files(pattern: str) -> str:
    """Find files matching a glob pattern, e.g. '**/*.py' (scoped to the workspace)."""
    full = str(WORKSPACE / pattern) if not os.path.isabs(pattern) else pattern
    matches = [m for m in sorted(_glob.glob(full, recursive=True))[:200]
               if Path(m).resolve().is_relative_to(WORKSPACE)]
    return "\n".join(matches) if matches else "(no matches)"


@tool
def bash(command: str) -> str:
    """Run a shell command in the workspace (blocklisted destructive commands are refused)."""
    log_tool.info(f"[bash] {command[:100]}")
    for bad in BASH_BLOCKLIST:
        if bad in command:
            return f"Error: blocked by safety policy (matched {bad!r})"
    try:
        p = subprocess.run(command, shell=True, cwd=str(WORKSPACE),
                           capture_output=True, text=True, timeout=BASH_TIMEOUT_S)
        return _truncate((p.stdout + p.stderr).strip() or "(no output)")
    except subprocess.TimeoutExpired:
        return f"Error: timeout after {BASH_TIMEOUT_S}s"
    except Exception as e:
        return f"Error: {e}"


# --- coding-specific tools (opt-in capability for the convergent path) -------
def lint_python(code: str) -> dict:
    """Lightweight static gate: must compile; flag bare excepts."""
    errors = []
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        py_compile.compile(tmp_path, doraise=True)
    except py_compile.PyCompileError as e:
        errors.append(f"SyntaxError: {e}")
    finally:
        os.unlink(tmp_path)
    try:
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                errors.append(f"Style: bare `except:` at line {node.lineno}")
    except SyntaxError:
        pass
    return {"passed": len(errors) == 0, "errors": errors}


def _run_tests(test_code: str, timeout: int = TEST_TIMEOUT_S) -> dict:
    """Write a test module and run it (pytest if available, else plain python)."""
    f = WORKSPACE / "_spec_test.py"
    f.write_text(test_code, encoding="utf-8")
    cmd = ([sys.executable, "-m", "pytest", "-q", str(f)] if _HAS_PYTEST else [sys.executable, str(f)])
    try:
        p = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"all_passed": False, "passed": 0, "failed": 0,
                "stdout": f"timeout after {timeout}s", "exit_code": -1}
    out = p.stdout + p.stderr
    mp, mf = re.search(r"(\d+) passed", out), re.search(r"(\d+) failed", out)
    return {"all_passed": p.returncode == 0,
            "passed": int(mp.group(1)) if mp else (0 if p.returncode else 1),
            "failed": int(mf.group(1)) if mf else (1 if p.returncode else 0),
            "stdout": _truncate(out, 4000), "exit_code": p.returncode}


@tool
def write_code(filename: str, content: str) -> str:
    """Write a bare *.py file under agent_code/ -- ONLY persists if it lints clean."""
    if not filename.endswith(".py") or "/" in filename or ".." in filename:
        return "ERROR: invalid filename (must be a bare *.py name)"
    res = lint_python(content)
    if not res["passed"]:
        return "REVERTED: linter rejected. errors:\n  " + "\n  ".join(res["errors"])
    path = AGENT_CODE_DIR / filename
    path.write_text(content, encoding="utf-8")
    log_tool.info(f"[write_code] wrote {path} ({len(content)} bytes, lint OK)")
    return f"WROTE {len(content)} bytes to {filename} (lint passed)"


@tool
def run_python(code: str) -> str:
    """Execute a Python snippet in the workspace (agent_code on sys.path); returns stdout/stderr."""
    f = WORKSPACE / "_run.py"
    f.write_text("import sys\nsys.path.insert(0, %r)\n" % str(AGENT_CODE_DIR) + code, encoding="utf-8")
    try:
        p = subprocess.run([sys.executable, str(f)], cwd=str(WORKSPACE),
                           capture_output=True, text=True, timeout=BASH_TIMEOUT_S)
        return _truncate(f"exit={p.returncode}\n" + (p.stdout + p.stderr).strip(), 8000)
    except subprocess.TimeoutExpired:
        return f"timeout after {BASH_TIMEOUT_S}s"


@tool
def run_tests(test_code: str) -> str:
    """Write a test module and run it; returns a pass/fail summary with output."""
    v = _run_tests(test_code)
    return f"all_passed={v['all_passed']} passed={v['passed']} failed={v['failed']}\n" + v["stdout"]


# The desktop assistant's toolset: file/shell first, coding tools available when needed.
TOOLS_BASE = [read_file, write_file, revert_file, grep, glob_files, bash,
              write_code, run_python, run_tests]
TOOLS_BY_NAME = {t.name: t for t in TOOLS_BASE}


# ════════════════════════════════════════════════════════════════════════════
# Phase 3 — the tool loop as a graph + the agent-as-tool subagent pattern
# ════════════════════════════════════════════════════════════════════════════

def build_agent_graph(tools, system: str = GENERAL_SYSTEM_PROMPT, role: str = "fast",
                      reasoning: bool = True, checkpointer=None):
    """START -> agent -> (tools_condition) -> tools -> agent -> ... -> END."""
    model = llm(role, reasoning=reasoning).bind_tools(tools)

    def agent_node(state: MessagesState):
        msgs = list(state["messages"])
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(system)] + msgs
        return {"messages": [model.invoke(msgs, config=CB)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or InMemorySaver())


SUBAGENT_SYSTEM = (
    f"You are a focused subagent working in a sandbox at {WORKSPACE}. You have a single "
    "subtask. Use tools to investigate or act. When confident, reply with a concise, "
    "self-contained summary and STOP calling tools -- that final message is the ONLY thing "
    "the parent agent sees. Do not ask clarifying questions; make a reasonable assumption "
    "and proceed."
)


def spawn_subagent(prompt: str, tools=TOOLS_BASE, system: str = SUBAGENT_SYSTEM,
                   role: str = "fast") -> str:
    sub_id = uuid.uuid4().hex[:6]
    log_sub.info(f"[sub:{sub_id}] spawn -- {prompt[:80]!r}")
    graph = build_agent_graph(tools, system=system, role=role)
    out = graph.invoke({"messages": [HumanMessage(prompt)]},
                       config=run_config(f"sub:{sub_id}", recursion_limit=2 * MAX_ITERATIONS))
    for m in reversed(out["messages"]):
        if isinstance(m, AIMessage) and content_text(m).strip():
            return strip_think(content_text(m))
    return "(subagent produced no output)"


def make_subagent_tool(name: str, description: str, system: str,
                       tools=TOOLS_BASE, role: str = "fast") -> StructuredTool:
    def _run(prompt: str) -> str:
        return spawn_subagent(prompt, tools=tools, system=system, role=role)
    return StructuredTool.from_function(_run, name=name, description=description)


# A general "delegate" subagent the lead assistant can call to scope a focused subtask.
delegate_tool = make_subagent_tool(
    "delegate",
    "Delegate a focused, self-contained subtask to a fresh subagent with the same tools. "
    "Pass a complete instruction; you get back only its final summary. Use for investigations "
    "or chunks of work you want isolated from the main thread.",
    SUBAGENT_SYSTEM)


# ════════════════════════════════════════════════════════════════════════════
# Phase 4 — the hardening stack as small graphs (kept from v3, used by the team)
# ════════════════════════════════════════════════════════════════════════════

class Section(BaseModel):
    section: str
    intent: str
    key_constraints: List[str] = Field(default_factory=list)


class ArchitectPlan(BaseModel):
    plan: List[Section]


class _AEState(TypedDict):
    task: str
    plan: dict
    output: str


def _architect_node(state: _AEState):
    model = llm("reasoning", reasoning=True, temperature=0.2).with_structured_output(ArchitectPlan)
    try:
        plan = model.invoke([SystemMessage("You are a senior architect. Produce a STRUCTURED PLAN "
                                           "the editor will implement -- not the final output."),
                             HumanMessage(f"TASK:\n{state['task']}")], config=CB)
        pd = plan.model_dump()
    except Exception:
        pd = {"plan": []}
    tracer.event(f"architect plan: {len(pd.get('plan', []))} section(s)",
                 ", ".join(s["section"] for s in pd.get("plan", [])))
    return {"plan": pd}


def _editor_node(state: _AEState):
    model = llm("fast", reasoning=False, temperature=0.3, max_tokens=3072)
    msg = model.invoke([SystemMessage("You are an editor. Execute the architect's plan precisely. "
                                      "Do NOT redesign. Output the final result only."),
                        HumanMessage(f"TASK:\n{state['task']}\n\nPLAN:\n"
                                     f"{json.dumps(state['plan'], indent=2)}\n\n"
                                     "Produce the final output now.")], config=CB)
    return {"output": strip_think(content_text(msg))}


_ae = StateGraph(_AEState)
_ae.add_node("architect", _architect_node)
_ae.add_node("editor", _editor_node)
_ae.add_edge(START, "architect")
_ae.add_edge("architect", "editor")
_ae.add_edge("editor", END)
architect_editor_app = _ae.compile()


def architect_editor_solve(task: str) -> dict:
    out = architect_editor_app.invoke({"task": task}, config=CB)
    return {"plan": out.get("plan", {}), "output": out.get("output", "")}


class _RefineState(TypedDict):
    query: str
    current: str
    critique: str
    iteration: int
    max_iter: int
    history: List[dict]


def _gen_node(state: _RefineState):
    r = think_then_answer(state["query"], max_tokens=1500)
    return {"current": r.answer, "iteration": 0, "history": [{"iteration": 0, "output": r.answer}]}


def _critique_node(state: _RefineState):
    model = llm("fast", reasoning=True, temperature=0.3, max_tokens=600)
    msg = model.invoke([HumanMessage(state["query"]), AIMessage(state["current"]),
                        HumanMessage("Critique your output as a strict reviewer. List 2-5 specific "
                                     "issues. If it is already excellent, say so.")], config=CB)
    return {"critique": strip_think(content_text(msg))}


def _refine_node(state: _RefineState):
    model = llm("fast", reasoning=True, temperature=0.3, max_tokens=1500)
    msg = model.invoke([HumanMessage(state["query"]),
                        HumanMessage(f"Previous output:\n{state['current']}\n\n"
                                     f"Critique:\n{state['critique']}\n\n"
                                     "Produce a refined version addressing every point.")], config=CB)
    cur = strip_think(content_text(msg))
    it = state["iteration"] + 1
    return {"current": cur, "iteration": it,
            "history": state["history"] + [{"iteration": it, "critique": state["critique"],
                                            "output": cur}]}


def _refine_route(state: _RefineState) -> Literal["critique", "__end__"]:
    return "critique" if state["iteration"] < state["max_iter"] else END


_rf = StateGraph(_RefineState)
_rf.add_node("generate", _gen_node)
_rf.add_node("critique", _critique_node)
_rf.add_node("refine", _refine_node)
_rf.add_edge(START, "generate")
_rf.add_edge("generate", "critique")
_rf.add_edge("critique", "refine")
_rf.add_conditional_edges("refine", _refine_route, {"critique": "critique", END: END})
self_refine_app = _rf.compile()


def self_refine(query: str, iterations: int = 2) -> dict:
    out = self_refine_app.invoke({"query": query, "max_iter": iterations}, config=CB)
    return {"final": out["current"], "history": out["history"], "iterations_run": out["iteration"]}


def strip_code_fences(text: str) -> str:
    text = strip_think(text)
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if inner.lstrip().startswith("python"):
                inner = inner.split("python", 1)[1]
            return inner.strip()
    return text.strip()


class Attack(BaseModel):
    category: str
    scenario: str
    why_it_breaks: str
    severity: Literal["critical", "major", "minor"]


class AttackList(BaseModel):
    attacks: List[Attack]


def adversarial_probe(target_description: str, candidate_output: str, n_max: int = 4) -> list:
    model = llm("reasoning", reasoning=True, temperature=0.4).with_structured_output(AttackList)
    try:
        res = model.invoke([SystemMessage("You are a hostile adversary. Find ways to BREAK the "
                                          "candidate: edge cases, bad assumptions, concrete "
                                          "counterexamples."),
                            HumanMessage(f"TARGET:\n{target_description}\n\nCANDIDATE:\n"
                                         f"{candidate_output}\n\nFind up to {n_max} ways to break "
                                         "this.")], config=CB)
        attacks = [a.model_dump() for a in res.attacks][:n_max]
    except Exception:
        attacks = []
    tracer.event(f"adversary found {len(attacks)} attack(s)",
                 "\n".join(f"[{a['severity']}] {a['scenario']}" for a in attacks))
    return attacks


# ════════════════════════════════════════════════════════════════════════════
# Phase 5 — planning + durable state (structured Plan, TaskDAG, bi-temporal memory, spec)
# ════════════════════════════════════════════════════════════════════════════

class PlanStep(BaseModel):
    step_id: str
    description: str
    depends_on: List[str] = Field(default_factory=list)
    expected_artifact: str = ""


class Plan(BaseModel):
    goal: str
    steps: List[PlanStep]


def make_plan(goal: str, role: str = "reasoning") -> Plan:
    model = llm(role, reasoning=True, temperature=0.0, max_tokens=2000).with_structured_output(Plan)
    try:
        plan = model.invoke([SystemMessage("Produce a step-by-step, dependency-ordered plan."),
                             HumanMessage(goal)], config=CB)
        tracer.event(f"plan: {len(plan.steps)} step(s)",
                     "\n".join(f"{s.step_id} <- {s.depends_on}: {s.description[:60]}"
                               for s in plan.steps))
        return plan
    except Exception:
        return Plan(goal=goal, steps=[])


class TaskDAG:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(str(db_path), isolation_level=None)
        self.conn.execute("CREATE TABLE IF NOT EXISTS nodes ("
                          "node_id TEXT PRIMARY KEY, title TEXT, status TEXT, "
                          "attempts INTEGER DEFAULT 0, depends_on TEXT)")

    def add_node(self, node_id, title, depends_on=None):
        self.conn.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?)",
                          (node_id, title, "pending", 0, json.dumps(depends_on or [])))

    def all_nodes(self):
        return list(self.conn.execute("SELECT node_id, title, status, attempts FROM nodes"))

    def ready_nodes(self):
        done = {r[0] for r in self.conn.execute("SELECT node_id FROM nodes WHERE status='done'")}
        out = []
        for nid, title, deps in self.conn.execute(
                "SELECT node_id, title, depends_on FROM nodes WHERE status='pending'"):
            if all(d in done for d in json.loads(deps)):
                out.append((nid, title))
        return out

    def set_status(self, node_id, status):
        self.conn.execute("UPDATE nodes SET status=?, attempts=attempts+1 WHERE node_id=?",
                          (status, node_id))


class BiTemporalMemory:
    """Facts with validity intervals; superseded facts are invalidated, not deleted."""

    def __init__(self):
        self.records: List[dict] = []

    def store(self, fact, kind="observation", source="agent", thread_id=None):
        rec_id = uuid.uuid4().hex[:8]
        self.records.append({"id": rec_id, "fact": fact, "kind": kind, "source": source,
                             "thread_id": thread_id, "valid_from": time.time(), "valid_to": None})
        return rec_id

    def invalidate(self, fact_id, reason):
        for r in self.records:
            if r["id"] == fact_id and r["valid_to"] is None:
                r["valid_to"] = time.time()
                r["invalidated_reason"] = reason

    def query_valid(self, kind=None, thread_id=None):
        return [r for r in self.records
                if r["valid_to"] is None and (kind is None or r["kind"] == kind)
                and (thread_id is None or r.get("thread_id") == thread_id)]

    def recall(self, query, k=3, thread_id=None):
        q = set(re.findall(r"\w+", query.lower()))
        scored = []
        for r in self.query_valid(thread_id=thread_id):
            overlap = len(q & set(re.findall(r"\w+", r["fact"].lower())))
            if overlap:
                scored.append((overlap, r["fact"]))
        scored.sort(reverse=True)
        return [f for _, f in scored[:k]]


class RagManagedMemory:
    """Long-term semantic memory backed by Vertex AI RAG Engine on RagManagedDb — same
    interface as BiTemporalMemory, so the context hook and assist() are unchanged.

    WHY RAG ENGINE / RagManagedDb FITS THIS LAYER (and not the checkpointer): semantic facts are
    append-only and are recalled by *meaning*, which is exactly what a managed vector store gives
    you — embeddings + ANN search with zero infra to run. RagManagedDb is Vertex AI RAG Engine's
    built-in vector database (no Vector Search index or Cloud SQL to provision). The turn-by-turn
    LangGraph CHECKPOINTER is the opposite access pattern: a hot transactional read-modify-write
    keyed by thread_id, once per superstep. That belongs on Firestore / Cloud SQL / Memorystore,
    NOT a vector store. So: facts -> RAG Engine; checkpoints -> a transactional store.

    Each fact is one RagFile in the corpus. We prepend a machine-readable tag line
    ("[mem] id=… kind=… thread=… source=…") to the text and set display_name="mem-<id>", so we
    can scope recall to a thread and resolve a fact id back to its RagFile for invalidation.

    NOTE on bi-temporality: RAG Engine has no soft-delete, so invalidate() *deletes* the RagFile
    rather than stamping valid_to. The bi-temporal "invalidate, don't delete" guarantee of the
    in-process store is relaxed to a hard delete here — call it out if audit history matters.
    """

    _TAG_RE = re.compile(r"^\[mem\] id=(\S+) kind=(\S+) thread=(\S+) source=(\S+)\n(.*)$", re.S)

    def __init__(self, project=None, location=None, corpus_display=RAG_CORPUS_DISPLAY,
                 corpus_name=RAG_CORPUS_NAME, embedding_model=RAG_EMBEDDING_MODEL):
        import vertexai
        from vertexai import rag
        self._rag = rag
        self.project = project or RAG_PROJECT
        self.location = location or RAG_LOCATION
        vertexai.init(project=self.project, location=self.location)
        self.corpus = self._find_or_create_corpus(corpus_display, corpus_name, embedding_model)
        log_tool.info(f"[ragmem] using corpus {self.corpus.name}")

    def _find_or_create_corpus(self, display, name, embedding_model):
        rag = self._rag
        if name:  # reuse an existing corpus by full resource name
            return rag.get_corpus(name=name)
        for c in rag.list_corpora():
            if c.display_name == display:
                return c
        backend = rag.RagVectorDbConfig(
            vector_db=rag.RagManagedDb(),
            rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model=embedding_model)))
        return rag.create_corpus(display_name=display, backend_config=backend)

    def _tagged(self, fact, rec_id, kind, source, thread_id):
        return (f"[mem] id={rec_id} kind={kind} thread={thread_id or '-'} "
                f"source={source}\n{fact}")

    def store(self, fact, kind="observation", source="agent", thread_id=None):
        rag = self._rag
        rec_id = uuid.uuid4().hex[:8]
        text = self._tagged(fact, rec_id, kind, source, thread_id)
        # RAG Engine ingests files; write the fact to a temp file and upload it as one RagFile.
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(text)
            path = fh.name
        try:
            rag.upload_file(corpus_name=self.corpus.name, path=path,
                            display_name=f"mem-{rec_id}",
                            description=f"{kind}|{thread_id or '-'}")
        except Exception as e:
            log_tool.warning(f"[ragmem] upload failed: {e}")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return rec_id

    def invalidate(self, fact_id, reason):
        # RAG Engine has no soft-delete — resolve the RagFile by display_name and delete it.
        rag = self._rag
        for f in rag.list_files(corpus_name=self.corpus.name):
            if getattr(f, "display_name", "") == f"mem-{fact_id}":
                rag.delete_file(name=f.name)
                return

    def _parse(self, text):
        m = self._TAG_RE.match(text or "")
        if not m:
            return {"id": None, "kind": None, "thread_id": None, "source": None, "fact": text}
        rid, kind, thread, source, fact = m.groups()
        return {"id": rid, "kind": kind, "source": source,
                "thread_id": None if thread == "-" else thread, "fact": fact}

    def query_valid(self, kind=None, thread_id=None, limit=200):
        # No bulk text read-back from RAG Engine; recover facts via a broad retrieval and filter.
        rows = self._retrieve("", top_k=limit)
        return [r for r in rows
                if (kind is None or r["kind"] == kind)
                and (thread_id is None or r["thread_id"] == thread_id)]

    def _retrieve(self, query, top_k):
        rag = self._rag
        cfg = rag.RagRetrievalConfig(top_k=top_k)
        resp = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=self.corpus.name)],
            text=query or "memory", rag_retrieval_config=cfg)
        return [self._parse(c.text) for c in resp.contexts.contexts]

    def recall(self, query, k=3, thread_id=None):
        # Real semantic recall: vector search over the corpus. Over-fetch so we can drop facts
        # from other threads before truncating to k.
        rows = self._retrieve(query, top_k=max(k * 5, 10))
        out = [r["fact"] for r in rows if thread_id is None or r["thread_id"] == thread_id]
        return out[:k]


def write_definition_of_done(criteria: List[dict], import_line: str = "") -> dict:
    contract = {"passing_criteria": criteria, "import_line": import_line}
    (AGENT_CODE_DIR / "DEFINITION_OF_DONE.json").write_text(json.dumps(contract, indent=2))
    return contract


def compile_test_suite(criteria: List[dict], import_line: str = "") -> str:
    lines = ["import sys", f"sys.path.insert(0, {str(AGENT_CODE_DIR)!r})"]
    if import_line:
        lines.append(import_line)
    lines.append("")
    for c in criteria:
        lines.append(f"def test_{c['name']}():")
        lines.append(f"    assert {c['check']}")
        lines.append("")
    return "\n".join(lines)


def spec_verify(contract: dict) -> dict:
    suite = compile_test_suite(contract["passing_criteria"], contract.get("import_line", ""))
    return _run_tests(suite)


# ════════════════════════════════════════════════════════════════════════════
# Phase 6 — context engineering: a pre_model_hook that bounds the model's view
# ════════════════════════════════════════════════════════════════════════════

def make_context_hook(max_recent: int = 6, memory: Optional[BiTemporalMemory] = None):
    def hook(state: MessagesState) -> dict:
        msgs = list(state["messages"])
        if len(msgs) <= max_recent + 2:
            return {}
        head = msgs[:1]
        anchor = [m for m in msgs[1:3] if isinstance(m, HumanMessage)][:1]
        recent = msgs[-max_recent:]
        dropped = msgs[len(head) + len(anchor):-max_recent]
        recall = ""
        if memory is not None:
            anchor_text = content_text(anchor[0]) if anchor else ""
            facts = memory.recall(anchor_text, k=4)
            if facts:
                recall = "\n".join(f"- {f}" for f in facts)
        note = SystemMessage(
            f"[context note] {len(dropped)} earlier step(s) were elided to bound the window."
            + (f"\n<durable_memory>\n{recall}\n</durable_memory>" if recall else ""))
        tracer.event(f"context trim: {len(dropped)} step(s) elided",
                     f"window now {len(head) + len(anchor) + 1 + len(recent)} msgs")
        return {"llm_input_messages": head + anchor + [note] + recent}
    return hook


def build_managed_agent(tools=None, system: str = GENERAL_SYSTEM_PROMPT, role: str = "fast",
                        max_recent: int = 8, memory: Optional[BiTemporalMemory] = None,
                        checkpointer=None):
    """create_react_agent + the context hook = a bounded-window general assistant.

    This is the DEFAULT desktop agent: open-ended, durable, tool-using — the right shape for
    requests that are not guaranteed convergent.
    """
    if tools is None:
        tools = TOOLS_BASE + [delegate_tool]
    return create_react_agent(
        llm(role, reasoning=True), tools, prompt=system,
        pre_model_hook=make_context_hook(max_recent, memory),
        checkpointer=checkpointer or InMemorySaver())


# ════════════════════════════════════════════════════════════════════════════
# Phase 7 — the convergent coding team (opt-in): planner -> implementer <-> tester -> review
# ════════════════════════════════════════════════════════════════════════════

class TeamState(TypedDict):
    task: str
    target_filename: str
    contract: dict
    plan: List[dict]
    test_result: dict
    review: dict
    report: str
    attempts: int
    max_attempts: int
    notes: List[str]


def _note(state, msg):
    return state.get("notes", []) + [msg]


def planner_node(state: TeamState):
    plan = make_plan(state["task"])
    return {"plan": [s.model_dump() for s in plan.steps],
            "notes": _note(state, f"planner: {len(plan.steps)} step(s)")}


def implementer_node(state: TeamState):
    fb = ""
    if state.get("test_result") and not state["test_result"].get("all_passed", True):
        fb = f"\n\nYour previous attempt failed its tests:\n{state['test_result'].get('stdout', '')[:600]}"
    task = (f"{state['task']}{fb}\n\nWrite the COMPLETE contents of {state['target_filename']}. "
            "Output ONLY raw Python source.")
    code = strip_code_fences(architect_editor_solve(task)["output"])
    msg = write_code.invoke({"filename": state["target_filename"], "content": code})
    attempts = state.get("attempts", 0) + 1
    return {"attempts": attempts, "notes": _note(state, f"implementer attempt {attempts}: {msg[:60]}")}


def tester_node(state: TeamState):
    v = spec_verify(state["contract"])
    status = "passed" if v["all_passed"] else "failed"
    return {"test_result": v, "notes": _note(state, f"tester: {status} ({v['passed']}p/{v['failed']}f)")}


def reviewer_node(state: TeamState):
    try:
        code = (AGENT_CODE_DIR / state["target_filename"]).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"review": {"error": "nothing to review"}}
    score = verifier_score(state["task"], code)
    attacks = adversarial_probe(state["task"], code, n_max=3)
    return {"review": {"score": score["score"], "reason": score["reason"], "n_attacks": len(attacks)},
            "notes": _note(state, f"reviewer: {score['score']}/10, {len(attacks)} attack(s)")}


def report_node(state: TeamState):
    facts = "\n".join(f"- {n}" for n in state.get("notes", []))
    draft = self_refine(f"Write a concise REPORT.md (<200 words) for this task.\n\nTASK:\n"
                        f"{state['task']}\n\nWHAT HAPPENED:\n{facts}", iterations=1)
    (AGENT_CODE_DIR / "REPORT.md").write_text(draft["final"], encoding="utf-8")
    return {"report": draft["final"], "notes": _note(state, "report_writer: wrote REPORT.md")}


def tester_route(state: TeamState) -> Literal["implementer", "reviewer"]:
    if state["test_result"]["all_passed"]:
        return "reviewer"
    return "implementer" if state["attempts"] < state["max_attempts"] else "reviewer"


def build_team_graph(checkpointer=None):
    g = StateGraph(TeamState)
    g.add_node("planner", planner_node)
    g.add_node("implementer", implementer_node)
    g.add_node("tester", tester_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("report_writer", report_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "implementer")
    g.add_edge("implementer", "tester")
    g.add_conditional_edges("tester", tester_route,
                            {"implementer": "implementer", "reviewer": "reviewer"})
    g.add_edge("reviewer", "report_writer")
    g.add_edge("report_writer", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def run_team(task: str, target_filename: str, contract: dict, max_attempts: int = 2,
             stream: bool = True) -> dict:
    app = build_team_graph()
    inputs = {"task": task, "target_filename": target_filename, "contract": contract,
              "attempts": 0, "max_attempts": max_attempts, "test_result": {}, "notes": []}
    cfg = run_config("team", recursion_limit=50)
    if stream:
        stream_run(app, inputs, cfg)
        return app.get_state(cfg).values
    return app.invoke(inputs, config=cfg)


# ════════════════════════════════════════════════════════════════════════════
# Top-level entry points + CLI
# ════════════════════════════════════════════════════════════════════════════

# One durable assistant + memory for the process (the REPL shares a thread).
_ASSISTANT = None
_THREAD_ID = f"desktop-{uuid.uuid4().hex[:6]}"


def build_memory():
    """Construct the long-term memory store from AGENTD_MEMORY ('memory' | 'rag')."""
    if MEMORY_BACKEND in ("rag", "ragmanageddb", "vertex"):
        try:
            m = RagManagedMemory()
            log.info(f"long-term memory: Vertex AI RAG Engine ({m.corpus.name})")
            return m
        except Exception as e:
            log.error(f"RAG memory unavailable ({e}); falling back to in-process memory.")
    log.info("long-term memory: in-process (BiTemporalMemory)")
    return BiTemporalMemory()


_MEMORY = build_memory()


def remember(fact: str, *, kind: str = "observation", thread_id: Optional[str] = None,
             source: str = "agent") -> None:
    """Persist one fact to the long-term store. Never raises — memory must not break a turn."""
    try:
        _MEMORY.store(fact, kind=kind, source=source, thread_id=thread_id or _THREAD_ID)
    except Exception as e:
        log_tool.warning(f"[memory] store failed: {e}")


def get_assistant():
    global _ASSISTANT
    if _ASSISTANT is None:
        _ASSISTANT = build_managed_agent(memory=_MEMORY)
    return _ASSISTANT


def assist(query: str, *, thread_id: Optional[str] = None, stream: bool = True) -> str:
    """Run the default bounded-window general assistant on one request; return its final answer.

    Long-term memory is written for the whole interaction: the request before the run, the
    answer after. The context hook (make_context_hook) then recalls from this same store on the
    next turn, so memory feeds back into the model's view.
    """
    app = get_assistant()
    tid = thread_id or _THREAD_ID
    remember(f"User request: {query}", kind="user", thread_id=tid)
    cfg = {"callbacks": [tracer],
           "configurable": {"thread_id": tid},
           "recursion_limit": 2 * MAX_ITERATIONS}
    inputs = {"messages": [HumanMessage(query)]}
    if stream:
        stream_run(app, inputs, cfg)
        msgs = app.get_state(cfg).values.get("messages", [])
    else:
        msgs = app.invoke(inputs, config=cfg)["messages"]
    answer = "(no answer)"
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and content_text(m).strip():
            answer = strip_think(content_text(m))
            break
    remember(f"Assistant answer: {answer[:600]}", kind="assistant", thread_id=tid)
    return answer


def repl():
    print(f"desktop_agent REPL — backend={LLM_BACKEND}, model={MODELS['fast']}, "
          f"workspace={WORKSPACE}")
    print("Type your request; 'exit' or Ctrl-D to quit. The conversation is durable within "
          "this session.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        ans = assist(q)
        print(f"\nassistant> {ans}\n")
    tracer.summary()


def main(argv=None):
    p = argparse.ArgumentParser(description="General-assistance agent on LangGraph + Gemini 3.0.")
    p.add_argument("query", nargs="*", help="one-shot request for the assistant")
    p.add_argument("-i", "--interactive", action="store_true", help="interactive REPL")
    p.add_argument("--think", action="store_true",
                   help="reason-only (adaptive router, no tools) instead of the tool loop")
    p.add_argument("--health", action="store_true", help="probe the Gemini backend and exit")
    p.add_argument("--no-stream", action="store_true", help="do not stream node updates")
    # opt-in convergent coding team:
    p.add_argument("--team", action="store_true", help="run the convergent coding team instead")
    p.add_argument("--file", help="(team) target *.py filename")
    p.add_argument("--task", help="(team) task description")
    args = p.parse_args(argv)

    if args.health:
        ok = backend_healthcheck()
        sys.exit(0 if ok else 1)

    if args.team:
        if not (args.file and args.task):
            p.error("--team requires --file and --task")
        # A trivially-checkable contract by default; refine per task as needed.
        contract = write_definition_of_done(
            [{"name": "module_imports", "check": "True"}],
            import_line=f"import {Path(args.file).stem}")
        remember(f"Coding-team task ({args.file}): {args.task}", kind="user")
        state = run_team(args.task, args.file, contract, max_attempts=3,
                         stream=not args.no_stream)
        remember(f"Coding-team result for {args.file}: attempts={state.get('attempts')} "
                 f"review={state.get('review')}", kind="assistant")
        print("\n=== TEAM RESULT ===")
        print("attempts:", state.get("attempts"), "| review:", state.get("review"))
        print((AGENT_CODE_DIR / "REPORT.md").read_text() if (AGENT_CODE_DIR / "REPORT.md").exists()
              else "(no report)")
        return

    if args.interactive:
        repl()
        return

    query = " ".join(args.query).strip()
    if not query:
        p.error("provide a request, or use -i for the REPL / --health to probe the backend")

    if args.think:
        remember(f"User request: {query}", kind="user")
        r = adaptive_think(query)
        remember(f"Assistant answer ({r['strategy']}): {r['answer'][:600]}", kind="assistant")
        print(f"\n[{r['difficulty']}/{r['type']} -> {r['strategy']}]\n")
        print(r["answer"])
    else:
        ans = assist(query, stream=not args.no_stream)
        print(f"\nassistant> {ans}")
    tracer.summary()


if __name__ == "__main__":
    main()
