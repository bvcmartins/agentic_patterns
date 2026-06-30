#!/usr/bin/env python3
"""
coding_agent_gemini.py — a callable coding agent (generate + fix) on LangGraph + Gemini.

This is the *coding agent* slice of `code_assistant/claude_code_from_scratch_v3.ipynb`,
re-pointed at Google's Gemini (the same backend seam as `agent_desktop/desktop_agent.py`)
and reshaped into a **library** rather than a CLI assistant. It exists to be driven by an
outer harness:

    judge generates a task  ->  CodingAgent solves it  ->  judge assesses the result
                                        ^                            |
                                        |   judge.add_instruction()  |
                                        +----------- self-improvement loop ----------+

So the design centre of gravity is a clean programmatic API, not a REPL:

  * `CodingAgent.generate(task, test_code=...)` — write code for a task, optionally iterating
    a generate -> verify loop against a pytest contract, feeding the VERBATIM test failure
    back into the next attempt (v3's `code_with_tests`).
  * `CodingAgent.fix(task, code, feedback, test_code=...)` — repair existing code given
    feedback (failing tests, a judge critique, a stack trace) and re-verify.
  * `CodingAgent.add_instruction(text)` — the **self-improvement seam**. The judge calls this
    with guidance ("you keep forgetting empty-input edge cases"); every subsequent generate/fix
    renders the accumulated guidance into the system prompt. `save_policy`/`load_policy` persist
    those learned instructions across process runs, so the loop survives separate invocations.

Only the model factory (`_build_model`) is backend-specific; everything above it is
backend-agnostic LangChain/LangGraph, so swapping Gemini for anything else is one function.

------------------------------------------------------------------------------------------
RUNNING IT  (the Gemini backend runs on Google Cloud — not this laptop)
------------------------------------------------------------------------------------------
  pip install -U langgraph langchain-core langchain-google-vertexai pydantic rich
  #   (or, for the Developer API backend:  langchain-google-genai)

  # Vertex AI (default backend): ADC + a project.
  gcloud auth application-default login
  export GOOGLE_CLOUD_PROJECT=my-project
  export GOOGLE_CLOUD_LOCATION=us-central1

  # Gemini Developer API backend instead of Vertex:
  #   export LLM_BACKEND=genai ; export GOOGLE_API_KEY=...

  python coding_agent_gemini.py --health                      # probe the backend
  python coding_agent_gemini.py --selftest                    # offline checks, no backend
  python coding_agent_gemini.py --task "write inc(n) returning n+1" \
                                --test "from solution import inc" --check "inc(1) == 2"
  python coding_agent_gemini.py --demo-loop                   # tiny self-improvement demo

Programmatic use:
  from coding_agent_gemini import CodingAgent
  agent = CodingAgent()
  res = agent.generate("Write is_prime(n).", test_code=..., filename="solution.py")
  if not res.passed:
      agent.add_instruction("Always handle n < 2 explicitly.")
      res = agent.fix(res.task, res.code, res.test_output, test_code=..., filename="solution.py")
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
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
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
        return f"{color}[{record.levelname:5s}] {short:14s} | {record.getMessage()}{self.RESET}"


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_Fmt())
log = logging.getLogger("coder")
log.handlers.clear()
log.addHandler(_handler)
log.setLevel(AGENT_LOG_LEVEL)
log.propagate = False

log_llm = logging.getLogger("coder.llm")
log_tool = logging.getLogger("coder.tool")
log_graph = logging.getLogger("coder.graph")

# --- Backend / Gemini config -------------------------------------------------
# "vertex" (default) -> ChatVertexAI on Google Cloud via ADC; "genai" -> Developer API key.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "vertex").lower()
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
GCP_LOCATION = (os.environ.get("GOOGLE_CLOUD_LOCATION")
                or os.environ.get("VERTEXAI_LOCATION") or "us-central1")

# Two-tier economics carried from v3: a cheap model for high-volume generation, a strong
# model reserved for judging/architecting. Override any role independently by env var.
DEFAULT_MODEL = os.environ.get("CODER_MODEL", "gemini-3-pro-preview")
MODELS = {
    "reasoning": os.environ.get("CODER_REASONING_MODEL", DEFAULT_MODEL),
    "fast":      os.environ.get("CODER_FAST_MODEL", DEFAULT_MODEL),
}

# --- Sandbox workspace -------------------------------------------------------
WORKSPACE = Path(os.environ.get("CODER_WORKSPACE",
                                str(Path.cwd() / "coder_workspace"))).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
AGENT_CODE_DIR = WORKSPACE / "agent_code"
AGENT_CODE_DIR.mkdir(exist_ok=True)
# Where learned self-improvement instructions persist between runs (see CodingAgent).
DEFAULT_POLICY_PATH = WORKSPACE / "learned_policy.json"

# --- Limits ------------------------------------------------------------------
MAX_TOOL_OUTPUT = 12_000
BASH_TIMEOUT_S = 60
TEST_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 600
MAX_ITERATIONS = 20
MAX_INSTRUCTIONS = 40  # cap the learned-guidance block so the prompt can't grow unbounded

BASH_BLOCKLIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "> /dev/", ":(){ :|:& };:"]
_HAS_PYTEST = importlib.util.find_spec("pytest") is not None


def _build_model(model: str, thinking: Optional[bool], temperature: float,
                 max_tokens: Optional[int]):
    """Construct one chat model. The ONLY place a Gemini client is built.

    `thinking` maps to Gemini's thinking budget: True -> dynamic (-1, thoughts returned),
    False -> off (0, best for structured/JSON routing), None -> backend default. Args are
    passed defensively (SDK versions vary), falling back to a plain client on TypeError.
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
    """Get a chat model by ROLE ('reasoning' | 'fast') or an explicit model id.

    `reasoning` is the public knob name (so the v3 graphs read identically); it drives
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
# Phase 0.5 — observability: a LangChain callback handler (off by default for a library)
# ════════════════════════════════════════════════════════════════════════════

# A library driven by a harness should be quiet by default; set AGENT_TRACE=full to watch.
TRACE_LEVEL = os.environ.get("AGENT_TRACE", "off").lower()
TRACE_PREVIEW = int(os.environ.get("AGENT_TRACE_PREVIEW", "1200"))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
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
    """Flatten a message's content to plain answer text, dropping Gemini 'thought' blocks."""
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
    """Narrates model/tool calls when AGENT_TRACE != off. Counters are always live."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens = 0
        self.tool_calls = 0
        self._starts: Dict[Any, float] = {}

    @property
    def on(self):
        return TRACE_LEVEL != "off"

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
        self._starts[run_id] = time.time()

    def on_llm_end(self, response, *, run_id=None, **kw):
        dt = time.time() - self._starts.pop(run_id, time.time())
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = content_text(msg) if msg else getattr(gen, "text", "")
            usage = (getattr(msg, "usage_metadata", None) or {}) if msg else {}
            tok = usage.get("output_tokens", 0) or 0
        except Exception:
            text, tok = "", 0
        with self._lock:
            self.tokens += tok
        if text:
            self._emit(Panel(Text(_clip(text)), title=f"answer ({dt:.1f}s, {tok} tok)",
                             title_align="left", border_style="green") if _HAS_RICH else None,
                       f"[answer] {_clip(text, 240)}")

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kw):
        with self._lock:
            self.tool_calls += 1

    def event(self, title, body=""):
        if not self.on:
            return
        if _HAS_RICH:
            t = Text(title, style="bold blue")
            if body:
                t.append("\n" + _clip(body))
            _console.print(Panel(t, border_style="blue", title_align="left"))
        else:
            log.info(f"[event] {title} -- {_clip(body, 200)}")

    def summary(self) -> dict:
        s = {"model_calls": self.calls, "output_tokens": self.tokens, "tool_calls": self.tool_calls}
        log.info(f"TRACE: {s}")
        return s


tracer = RichTracer()
CB = {"callbacks": [tracer]}


def run_config(label: str = "run", **extra) -> dict:
    cfg = {"callbacks": [tracer], "configurable": {"thread_id": f"{label}-{uuid.uuid4().hex[:6]}"}}
    cfg.update(extra)
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — the coding epistemics + free-text helpers
# ════════════════════════════════════════════════════════════════════════════

# v3's STRONG_SYSTEM_PROMPT: a *coding* agent's rules of engagement (not the general one).
CODER_SYSTEM_PROMPT = (
    "You are a careful, senior software engineer. You write correct, minimal, idiomatic "
    "Python and you never claim behaviour you have not verified.\n\n"
    "RULES OF ENGAGEMENT:\n"
    "1. Never claim code works without a runnable artifact — prefer code that can be executed "
    "and tested over prose about what it would do.\n"
    "2. Defer every question of behaviour to execution: if a test or the linter fails, that "
    "signal is correct until you have proven otherwise.\n"
    "3. The spec / the tests are the source of truth. Make the tests pass; do not argue with "
    "them.\n"
    "4. Say 'I don't know' or surface an assumption explicitly rather than guessing silently.\n"
    "5. Output COMPLETE, self-contained source — no placeholders, no '...', no TODOs. Handle "
    "the edge cases (empty inputs, bounds, None) the task implies."
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


def strip_code_fences(text: str) -> str:
    """Pull raw source out of a ```python ... ``` fence if the model wrapped its answer."""
    text = strip_think(text)
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if inner.lstrip().startswith("python"):
                inner = inner.split("python", 1)[1]
            return inner.strip()
    return text.strip()


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — tools & quality gates (lint, test runner, sandboxed file/shell ops)
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


def lint_python(code: str) -> dict:
    """Lightweight static gate: must compile; flag bare excepts. Fast, dependency-free."""
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
    cmd = ([sys.executable, "-m", "pytest", "-q", str(f)] if _HAS_PYTEST
           else [sys.executable, str(f)])
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


def write_code_to_disk(filename: str, content: str) -> str:
    """Persist a bare *.py file under agent_code/ ONLY if it lints clean. Returns a status line.

    This is the central reliability mechanism: broken code never reaches disk, so a downstream
    test run never fails for a trivial syntax reason. (Plain function so the agent loop and the
    @tool wrapper share one body.)
    """
    if not filename.endswith(".py") or "/" in filename or ".." in filename:
        return "ERROR: invalid filename (must be a bare *.py name)"
    res = lint_python(content)
    if not res["passed"]:
        return "REVERTED: linter rejected. errors:\n  " + "\n  ".join(res["errors"])
    path = AGENT_CODE_DIR / filename
    path.write_text(content, encoding="utf-8")
    log_tool.info(f"[write_code] wrote {path} ({len(content)} bytes, lint OK)")
    return f"WROTE {len(content)} bytes to {filename} (lint passed)"


# --- @tool-decorated versions (only needed for the optional agentic generation mode) ---------
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
        return f"{action}: {p} (snapshot saved -- use revert_file to undo)"
    except Exception as e:
        return f"Error: {e}"


@tool
def bash(command: str) -> str:
    """Run a shell command in the workspace (blocklisted destructive commands are refused)."""
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


@tool
def write_code(filename: str, content: str) -> str:
    """Write a bare *.py file under agent_code/ -- ONLY persists if it lints clean."""
    return write_code_to_disk(filename, content)


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


TOOLS_BASE = [read_file, write_file, bash, write_code, run_python, run_tests]


# ════════════════════════════════════════════════════════════════════════════
# Phase 3 — the optional agentic tool loop (for harder, multi-step tasks)
# ════════════════════════════════════════════════════════════════════════════

def build_agent_graph(tools, system: str, role: str = "fast", reasoning: bool = True,
                      checkpointer=None):
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


# ════════════════════════════════════════════════════════════════════════════
# Phase 4 — generation: architect -> editor (separate deliberation from transcription)
# ════════════════════════════════════════════════════════════════════════════

class Section(BaseModel):
    section: str
    intent: str
    key_constraints: List[str] = Field(default_factory=list)


class ArchitectPlan(BaseModel):
    plan: List[Section]


class _AEState(TypedDict):
    task: str
    system: str
    plan: dict
    output: str


def _architect_node(state: _AEState):
    model = llm("reasoning", reasoning=True, temperature=0.2).with_structured_output(ArchitectPlan)
    try:
        plan = model.invoke([SystemMessage(state["system"] + "\n\nYou are now ARCHITECT: produce a "
                                           "STRUCTURED PLAN the editor will implement — not code."),
                             HumanMessage(f"TASK:\n{state['task']}")], config=CB)
        pd = plan.model_dump()
    except Exception:
        pd = {"plan": []}
    tracer.event(f"architect plan: {len(pd.get('plan', []))} section(s)",
                 ", ".join(s["section"] for s in pd.get("plan", [])))
    return {"plan": pd}


def _editor_node(state: _AEState):
    # reasoning=False: the architect already deliberated; the editor just transcribes (fast, and
    # thinking won't eat the token budget and truncate the generated code).
    model = llm("fast", reasoning=False, temperature=0.2, max_tokens=4096)
    msg = model.invoke([SystemMessage(state["system"] + "\n\nYou are now EDITOR: execute the "
                                      "architect's plan precisely. Do NOT redesign. Output ONLY raw "
                                      "Python source — no markdown fence, no commentary."),
                        HumanMessage(f"TASK:\n{state['task']}\n\nPLAN:\n"
                                     f"{json.dumps(state['plan'], indent=2)}\n\n"
                                     "Produce the final source now.")], config=CB)
    return {"output": strip_code_fences(content_text(msg))}


_ae = StateGraph(_AEState)
_ae.add_node("architect", _architect_node)
_ae.add_node("editor", _editor_node)
_ae.add_edge(START, "architect")
_ae.add_edge("architect", "editor")
_ae.add_edge("editor", END)
architect_editor_app = _ae.compile()


def architect_editor_solve(task: str, system: str = CODER_SYSTEM_PROMPT) -> dict:
    out = architect_editor_app.invoke({"task": task, "system": system}, config=CB)
    return {"plan": out.get("plan", {}), "output": out.get("output", "")}


# ════════════════════════════════════════════════════════════════════════════
# Phase 5 — the spec layer: a definition-of-done compiled into runnable tests
# ════════════════════════════════════════════════════════════════════════════

def compile_test_suite(criteria: List[dict], import_line: str = "") -> str:
    """Turn [{'name','check'}, ...] into a real pytest module. Each check becomes an assert."""
    lines = ["import sys", f"sys.path.insert(0, {str(AGENT_CODE_DIR)!r})"]
    if import_line:
        lines.append(import_line)
    lines.append("")
    for c in criteria:
        lines.append(f"def test_{c['name']}():")
        lines.append(f"    assert {c['check']}")
        lines.append("")
    return "\n".join(lines)


def spec_verify(criteria: List[dict], import_line: str = "") -> dict:
    """Compile a contract to a pytest module and run it against the agent's code."""
    return _run_tests(compile_test_suite(criteria, import_line))


# ════════════════════════════════════════════════════════════════════════════
# Phase 6 — the CodingAgent: the callable API + the self-improvement seam
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CodeResult:
    """The structured return the judge consumes."""
    task: str
    code: str
    filename: str
    passed: Optional[bool]          # None when no tests were supplied (judge grades externally)
    test_output: str = ""
    rounds_used: int = 0
    file_path: Optional[str] = None
    history: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"task": self.task, "code": self.code, "filename": self.filename,
                "passed": self.passed, "test_output": self.test_output,
                "rounds_used": self.rounds_used, "file_path": self.file_path,
                "history": self.history}


class CodingAgent:
    """A callable coding agent: generate code, fix code, and absorb judge feedback.

    The self-improvement loop lives in `instructions`: a list of learned directives the judge
    appends via `add_instruction`. Every generate/fix renders them into the system prompt under
    a LEARNED GUIDANCE header, so feedback from past tasks changes future behaviour. This is the
    *durable policy* of the agent; per-task `feedback` (a failing test, one critique) is separate
    and ephemeral.

    Parameters
    ----------
    role : "fast" | "reasoning"   which MODELS tier the editor/agent runs on.
    instructions : initial learned directives (e.g. loaded from a previous run's policy).
    mode : "direct" (architect->editor->verify loop, deterministic) or
           "agentic" (a tool-using ReAct loop that can read/run/test on its own — for harder,
           multi-step tasks).
    policy_path : where save_policy/load_policy persist `instructions` as JSON.
    """

    def __init__(self, role: str = "fast", instructions: Optional[List[str]] = None,
                 mode: Literal["direct", "agentic"] = "direct",
                 policy_path: Optional[os.PathLike] = None):
        self.role = role
        self.mode = mode
        self.instructions: List[str] = list(instructions or [])
        self.policy_path = Path(policy_path) if policy_path else DEFAULT_POLICY_PATH

    # -- self-improvement seam ------------------------------------------------
    def add_instruction(self, instruction: str) -> None:
        """Judge entry point: record a learned directive. De-duplicates and caps the list."""
        instruction = (instruction or "").strip()
        if not instruction or instruction in self.instructions:
            return
        self.instructions.append(instruction)
        if len(self.instructions) > MAX_INSTRUCTIONS:
            self.instructions = self.instructions[-MAX_INSTRUCTIONS:]
        log.info(f"[improve] +1 instruction (now {len(self.instructions)})")

    def set_instructions(self, instructions: List[str]) -> None:
        self.instructions = list(instructions or [])

    def save_policy(self, path: Optional[os.PathLike] = None) -> Path:
        p = Path(path) if path else self.policy_path
        p.write_text(json.dumps({"instructions": self.instructions}, indent=2), encoding="utf-8")
        return p

    def load_policy(self, path: Optional[os.PathLike] = None) -> "CodingAgent":
        p = Path(path) if path else self.policy_path
        if p.exists():
            self.instructions = json.loads(p.read_text(encoding="utf-8")).get("instructions", [])
            log.info(f"[improve] loaded {len(self.instructions)} learned instruction(s)")
        return self

    def system_prompt(self) -> str:
        """The base coding epistemics + the accumulated learned guidance from the judge."""
        if not self.instructions:
            return CODER_SYSTEM_PROMPT
        block = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.instructions))
        return (CODER_SYSTEM_PROMPT + "\n\nLEARNED GUIDANCE (from review of your past work — "
                "follow these unless the task contradicts them):\n" + block)

    # -- the public capabilities ---------------------------------------------
    def generate(self, task: str, *, test_code: Optional[str] = None,
                 criteria: Optional[List[dict]] = None, import_line: str = "",
                 filename: str = "solution.py", max_rounds: int = 3) -> CodeResult:
        """Write code for `task`. If a contract (test_code or criteria+import_line) is given,
        iterate generate -> verify, feeding the VERBATIM test failure back, until green or
        `max_rounds` is spent."""
        test_code = self._resolve_tests(test_code, criteria, import_line)
        return self._solve_loop(task, filename, test_code, max_rounds, seed_code=None,
                                seed_feedback=None)

    def fix(self, task: str, code: str, feedback: str, *, test_code: Optional[str] = None,
            criteria: Optional[List[dict]] = None, import_line: str = "",
            filename: str = "solution.py", max_rounds: int = 3) -> CodeResult:
        """Repair `code` for `task` given `feedback` (failing tests, a judge critique, a trace).
        Re-verifies against the contract if one is supplied."""
        test_code = self._resolve_tests(test_code, criteria, import_line)
        return self._solve_loop(task, filename, test_code, max_rounds, seed_code=code,
                                seed_feedback=feedback)

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _resolve_tests(test_code, criteria, import_line) -> Optional[str]:
        if test_code:
            return test_code
        if criteria:
            return compile_test_suite(criteria, import_line)
        return None

    def _generate_once(self, task: str, feedback: Optional[str], prior_code: Optional[str]) -> str:
        """One code draft. `feedback`/`prior_code`, when present, steer a repair."""
        sys_prompt = self.system_prompt()
        parts = [task]
        if prior_code:
            parts.append(f"\n\nPREVIOUS CODE:\n```python\n{prior_code}\n```")
        if feedback:
            parts.append(f"\n\nFEEDBACK TO ADDRESS (verbatim — treat as ground truth):\n{feedback[:1500]}")
        full_task = "".join(parts)

        if self.mode == "agentic":
            return self._generate_agentic(full_task, sys_prompt)
        return strip_code_fences(architect_editor_solve(full_task, system=sys_prompt)["output"])

    def _generate_agentic(self, task: str, sys_prompt: str) -> str:
        """Tool-using draft: let the agent read/run/test, then return its final code block."""
        graph = build_agent_graph(TOOLS_BASE, system=sys_prompt, role=self.role)
        prompt = (task + "\n\nUse your tools to draft and check the code. When done, reply with "
                  "ONLY the final, complete Python source in a ```python fence.")
        out = graph.invoke({"messages": [HumanMessage(prompt)]},
                           config=run_config("gen", recursion_limit=2 * MAX_ITERATIONS))
        for m in reversed(out["messages"]):
            if isinstance(m, AIMessage) and content_text(m).strip():
                return strip_code_fences(content_text(m))
        return ""

    def _solve_loop(self, task: str, filename: str, test_code: Optional[str], max_rounds: int,
                    seed_code: Optional[str], seed_feedback: Optional[str]) -> CodeResult:
        history: List[dict] = []
        code = seed_code or ""
        feedback = seed_feedback
        prior = seed_code
        last_test: dict = {}

        # No contract: a single draft, lint-gated, returned for external grading.
        rounds = max_rounds if test_code else 1

        for rnd in range(1, rounds + 1):
            code = self._generate_once(task, feedback, prior)
            lint = lint_python(code)
            if not lint["passed"]:
                feedback = "Linter rejected the previous code:\n" + "\n".join(lint["errors"])
                prior = code
                history.append({"round": rnd, "stage": "lint", "passed": False,
                                "errors": lint["errors"]})
                continue

            write_code_to_disk(filename, code)

            if not test_code:
                history.append({"round": rnd, "stage": "lint", "passed": True})
                return CodeResult(task=task, code=code, filename=filename, passed=None,
                                  rounds_used=rnd, history=history,
                                  file_path=str(AGENT_CODE_DIR / filename))

            last_test = _run_tests(test_code)
            history.append({"round": rnd, "stage": "test", "passed": last_test["all_passed"],
                            "summary": f"{last_test['passed']}p/{last_test['failed']}f"})
            tracer.event(f"round {rnd}: tests {'PASS' if last_test['all_passed'] else 'FAIL'}",
                         last_test["stdout"][:300])
            if last_test["all_passed"]:
                break
            # Feed the VERBATIM test stdout back — the ground-truth error is the best next signal.
            feedback = "The tests FAILED. Fix the code so they pass.\n\n" + last_test["stdout"]
            prior = code

        return CodeResult(task=task, code=code, filename=filename,
                          passed=bool(last_test.get("all_passed")) if test_code else None,
                          test_output=last_test.get("stdout", ""), rounds_used=len(history),
                          history=history, file_path=str(AGENT_CODE_DIR / filename))


# ════════════════════════════════════════════════════════════════════════════
# Offline self-tests (no backend) + CLI
# ════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    """Exercise every backend-free path so the plumbing is checkable without Gemini."""
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        print(("  PASS " if cond else "  FAIL ") + name)

    print("offline self-tests (no model calls):")
    # lint gate
    check("lint accepts good code", lint_python("def f():\n    return 1\n")["passed"])
    check("lint rejects syntax error", not lint_python("def f(:\n")["passed"])
    check("lint flags bare except",
          any("bare" in e for e in lint_python("try:\n    x=1\nexcept:\n    pass\n")["errors"]))
    # fence stripping
    check("strip fence", strip_code_fences("```python\nx=1\n```") == "x=1")
    check("strip think", strip_think("<think>a</think>b") == "b")
    # write-code gate
    check("write_code rejects bad name", write_code_to_disk("a/b.py", "x=1").startswith("ERROR"))
    check("write_code rejects unlintable", write_code_to_disk("bad.py", "def(:").startswith("REVERTED"))
    check("write_code persists good", write_code_to_disk("good.py", "y=2\n").startswith("WROTE"))
    # spec layer compiles + runs against real code
    write_code_to_disk("solution.py", "def inc(n):\n    return n + 1\n")
    v = spec_verify([{"name": "inc", "check": "inc(1) == 2"}], import_line="from solution import inc")
    check("spec_verify green on correct code", v["all_passed"])
    v2 = spec_verify([{"name": "wrong", "check": "inc(1) == 99"}], import_line="from solution import inc")
    check("spec_verify red on wrong assertion", not v2["all_passed"])
    # self-improvement seam: instructions render into the prompt, dedupe, persist
    a = CodingAgent(instructions=["always handle empty input"])
    a.add_instruction("always handle empty input")          # dup, ignored
    a.add_instruction("validate types")
    check("instruction dedupe", len(a.instructions) == 2)
    check("instructions in prompt", "validate types" in a.system_prompt())
    p = a.save_policy(WORKSPACE / "_selftest_policy.json")
    b = CodingAgent().load_policy(p)
    check("policy round-trips", b.instructions == a.instructions)
    # graph compiles
    check("architect/editor graph compiled", architect_editor_app is not None)

    ok = all(c for _, c in results)
    print(f"\n{sum(c for _, c in results)}/{len(results)} passed — "
          + ("ALL GREEN" if ok else "FAILURES ABOVE"))
    return ok


def _demo_loop():
    """A tiny end-to-end self-improvement demo (NEEDS a live Gemini backend).

    Shows the intended harness shape: solve -> (judge would assess) -> add_instruction -> solve.
    Here the 'judge' is hard-coded for illustration.
    """
    agent = CodingAgent()
    task = "Write is_prime(n) returning True iff n is a prime number."
    tests = compile_test_suite(
        [{"name": "two", "check": "is_prime(2)"},
         {"name": "neg", "check": "not is_prime(-3)"},
         {"name": "one", "check": "not is_prime(1)"},
         {"name": "nine", "check": "not is_prime(9)"}],
        import_line="from solution import is_prime")
    r1 = agent.generate(task, test_code=tests, filename="solution.py")
    print(f"\nround 1: passed={r1.passed} rounds={r1.rounds_used}")
    if not r1.passed:
        # The judge would derive this from the failure; hard-coded here.
        agent.add_instruction("Treat n < 2 as non-prime before any loop.")
        r2 = agent.fix(task, r1.code, r1.test_output, test_code=tests, filename="solution.py")
        print(f"round 2 (after improvement): passed={r2.passed} rounds={r2.rounds_used}")
    tracer.summary()


def main(argv=None):
    p = argparse.ArgumentParser(description="Callable coding agent (generate/fix) on Gemini.")
    p.add_argument("--health", action="store_true", help="probe the Gemini backend and exit")
    p.add_argument("--selftest", action="store_true", help="run offline checks (no backend)")
    p.add_argument("--demo-loop", action="store_true", help="tiny self-improvement demo (needs backend)")
    p.add_argument("--task", help="generate code for this task")
    p.add_argument("--test", help="(with --task/--check) an import line for the contract, "
                                  "e.g. 'from solution import inc'")
    p.add_argument("--check", action="append", default=[],
                   help="(with --task) a boolean check, repeatable, e.g. 'inc(1) == 2'")
    p.add_argument("--file", default="solution.py", help="target *.py filename")
    p.add_argument("--mode", choices=["direct", "agentic"], default="direct")
    p.add_argument("--max-rounds", type=int, default=3)
    args = p.parse_args(argv)

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.health:
        sys.exit(0 if backend_healthcheck() else 1)
    if args.demo_loop:
        _demo_loop()
        return
    if args.task:
        criteria = [{"name": f"c{i}", "check": c} for i, c in enumerate(args.check)]
        agent = CodingAgent(mode=args.mode)
        res = agent.generate(args.task, criteria=criteria or None, import_line=args.test or "",
                             filename=args.file, max_rounds=args.max_rounds)
        print(f"\n=== RESULT ===\npassed={res.passed} rounds={res.rounds_used} "
              f"file={res.file_path}\n")
        print(res.code)
        if res.test_output:
            print("\n--- test output ---\n" + res.test_output)
        tracer.summary()
        return
    p.error("nothing to do: use --selftest, --health, --demo-loop, or --task")


if __name__ == "__main__":
    main()
