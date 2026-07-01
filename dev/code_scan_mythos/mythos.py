"""Homemade Mythos — autonomous vulnerability-research harness on Gemini 3.0.

A single-file, from-scratch reimplementation of the 12-component "Claude Mythos"
harness pattern, generalized to run against an ARBITRARY local codebase and
driven by one LLM family (Gemini 3.0) using multiple *personas* for the
cross-model-style 2-of-3 corroboration vote.

Design notes for this machine:
  * No Gemini API key is available here, so every LLM call goes through `ask()`,
    which supports three modes:
        - "live"    : real google-genai call (needs GEMINI_API_KEY)
        - "record"  : live call, and the reply is cached to recordings/ by a
                      content hash so future offline runs are free + deterministic
        - "replay"  : offline; returns the cached reply, else raises (or, if
                      MYTHOS_ALLOW_STUB=1, returns a harmless stub so the
                      pipeline still walks end to end)
    Mode is chosen automatically: "replay" when no key, "record" when a key is
    present, overridable with MYTHOS_LLM_MODE.

  * Layer 1 (C1-C4) is PURE INFRASTRUCTURE — no LLM. It runs and self-tests here
    with `python mythos.py --selftest`.

Run:
    python mythos.py --selftest          # exercise Layer 1 offline
    python mythos.py scan <path/to/repo> # (Layer 2+ — wired, needs a key/recordings)
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import difflib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Gemini 3.0 model ids. Kept as constants so the exact preview id is easy to
# swap when you wire a live key. All personas run on Gemini; "diversity" comes
# from distinct system prompts + temperatures (see PERSONAS), not from vendors.
GEMINI_PRO = os.environ.get("MYTHOS_GEMINI_PRO", "gemini-3-pro")
GEMINI_FLASH = os.environ.get("MYTHOS_GEMINI_FLASH", "gemini-3-flash")

# Per-million-token prices (USD) for the cost meter. Update when live.
PRICE_PER_M_TOKENS = {
    GEMINI_PRO: {"input": 2.00, "output": 12.00},
    GEMINI_FLASH: {"input": 0.30, "output": 2.50},
}


@dataclass
class Config:
    """Everything the harness needs to know about a run.

    `target_path` is any local directory — that is what makes this build
    generic rather than pinned to MLflow.
    """

    target_path: Path
    engagement_dir: Path
    recordings_dir: Path
    llm_mode: str = "auto"  # auto | live | record | replay
    allow_stub: bool = False

    @classmethod
    def for_target(cls, target: str | os.PathLike, workdir: str | os.PathLike | None = None) -> "Config":
        target_path = Path(target).resolve()
        base = Path(workdir).resolve() if workdir else Path.cwd()
        return cls(
            target_path=target_path,
            engagement_dir=base / "engagement",
            recordings_dir=base / "recordings",
            llm_mode=os.environ.get("MYTHOS_LLM_MODE", "auto"),
            allow_stub=os.environ.get("MYTHOS_ALLOW_STUB", "") == "1",
        )

    def ensure_dirs(self) -> None:
        for d in (
            self.engagement_dir,
            self.engagement_dir / "pocs",
            self.engagement_dir / "patches",
            self.engagement_dir / "diagrams",
            self.recordings_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# LLM layer — Gemini 3.0 behind one ask(), multi-persona, offline-capable
# ---------------------------------------------------------------------------

# The personas replace the three vendor models of the original harness. Each is
# a distinct reviewing stance; the corroboration gate (C7) treats agreement of
# 2 of 3 personas as the promote signal.
PERSONAS: dict[str, dict[str, Any]] = {
    "architect": {  # careful, precondition-focused (the "brain")
        "model": GEMINI_PRO,
        "temperature": 0.15,
        "stance": (
            "You are a meticulous senior security architect. State the exact "
            "preconditions a bug requires and refuse to call something exploitable "
            "unless the data flow from an attacker-controlled source to the sink is real."
        ),
    },
    "redteam": {  # aggressive, exploit-minded (the "red team eye")
        "model": GEMINI_PRO,
        "temperature": 0.5,
        "stance": (
            "You are an aggressive offensive security researcher. Assume the "
            "attacker is creative. Prefer to flag a plausible exploit path and "
            "describe the concrete steps to reach the dangerous sink."
        ),
    },
    "skeptic": {  # false-positive hunter (the "bulk/critic")
        "model": GEMINI_FLASH,
        "temperature": 0.2,
        "stance": (
            "You are a skeptical code reviewer whose job is to kill false positives. "
            "Assume a finding is wrong until the sink is provably reachable with "
            "attacker-controlled input; call out sanitizers, allow-lists, and binds."
        ),
    },
    "fixer": {  # minimal-patch author (Layer 3, C11)
        "model": GEMINI_PRO,
        "temperature": 0.1,
        "stance": (
            "You write the smallest possible patch that severs the vulnerability "
            "while preserving legitimate behaviour. Never widen scope; prefer an "
            "allow-list, a bind, or a hard refusal at the exact sink."
        ),
    },
}


@dataclass
class Reply:
    text: str
    persona: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    mode: str = ""


class CostMeter:
    """Running token + dollar total, so a scoreboard can report $/finding."""

    def __init__(self) -> None:
        self.calls = 0
        self.by_model: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()  # swarm workers call ask() in parallel

    def record(self, r: Reply) -> None:
        with self._lock:
            self.calls += 1
            m = self.by_model.setdefault(r.model, {"in": 0, "out": 0})
            m["in"] += r.input_tokens
            m["out"] += r.output_tokens

    def total_usd(self) -> float:
        total = 0.0
        for model, tok in self.by_model.items():
            price = PRICE_PER_M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
            total += tok["in"] / 1e6 * price["input"]
            total += tok["out"] / 1e6 * price["output"]
        return total


class LLM:
    """Gemini-backed multi-persona interface with record/replay for offline use."""

    def __init__(self, cfg: Config, meter: CostMeter | None = None) -> None:
        self.cfg = cfg
        self.meter = meter or CostMeter()
        self._client = None  # lazily created; only when a live call is needed

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        mode = cfg.llm_mode
        if mode == "auto":
            mode = "record" if key else "replay"
        self.mode = mode
        self._has_key = bool(key)

    # -- recording cache -----------------------------------------------------
    def _cache_key(self, persona: str, system: str, user: str) -> str:
        model = PERSONAS[persona]["model"]
        blob = json.dumps([persona, model, system, user], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:20]

    def _cache_path(self, key: str) -> Path:
        return self.cfg.recordings_dir / f"{key}.json"

    def _load_recording(self, key: str) -> Optional[Reply]:
        p = self._cache_path(key)
        if p.exists():
            d = json.loads(p.read_text())
            return Reply(**d)
        return None

    def _save_recording(self, key: str, r: Reply) -> None:
        self.cfg.recordings_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(key).write_text(json.dumps(asdict(r), indent=2))

    # -- the one interface ---------------------------------------------------
    def ask(self, persona: str, system: str, user: str, max_tokens: int = 2048) -> Reply:
        if persona not in PERSONAS:
            raise KeyError(f"unknown persona {persona!r}; have {list(PERSONAS)}")
        cache_key = self._cache_key(persona, system, user)

        if self.mode == "replay":
            rec = self._load_recording(cache_key)
            if rec is not None:
                rec.mode = "replay"
                self.meter.record(rec)
                return rec
            if self.cfg.allow_stub:
                r = self._stub(persona, cache_key)
                self.meter.record(r)
                return r
            raise RuntimeError(
                f"no recording for persona={persona} key={cache_key} and no live key. "
                f"Set GEMINI_API_KEY to record, or MYTHOS_ALLOW_STUB=1 to walk the "
                f"pipeline with stubbed replies."
            )

        # record or live: make the real call, cache if recording
        r = self._live(persona, system, user, max_tokens)
        if self.mode == "record":
            self._save_recording(cache_key, r)
        self.meter.record(r)
        return r

    def _live(self, persona: str, system: str, user: str, max_tokens: int) -> Reply:
        if not self._has_key:
            raise RuntimeError("live/record mode requires GEMINI_API_KEY / GOOGLE_API_KEY")
        spec = PERSONAS[persona]
        if self._client is None:
            from google import genai  # imported lazily so offline runs need no SDK

            self._client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            )
        from google.genai import types

        t0 = time.time()
        resp = self._client.models.generate_content(
            model=spec["model"],
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=spec["temperature"],
                max_output_tokens=max_tokens,
            ),
        )
        usage = getattr(resp, "usage_metadata", None)
        return Reply(
            text=resp.text or "",
            persona=persona,
            model=spec["model"],
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_s=time.time() - t0,
            mode="live",
        )

    def _stub(self, persona: str, key: str) -> Reply:
        return Reply(
            text=f"[STUB {persona}/{key}] no live model; MYTHOS_ALLOW_STUB=1",
            persona=persona,
            model=PERSONAS[persona]["model"],
            mode="stub",
        )


# ---------------------------------------------------------------------------
# Target loader — generic: any local repo/path (this is the MLflow decoupling)
# ---------------------------------------------------------------------------

# Extensions the scanner considers "source". Kept broad; ULTRAPLAN (C5) narrows.
SOURCE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rb", ".php", ".java",
    ".c", ".h", ".cpp", ".cc", ".rs", ".sh", ".ini", ".yaml", ".yml",
    ".toml", ".cfg", ".sql",
}
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
               "build", ".mypy_cache", ".pytest_cache", "engagement", "recordings"}


@dataclass
class SourceFile:
    relpath: str
    size: int


class Target:
    """A generic view over an arbitrary local codebase."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"target is not a directory: {self.root}")

    def files(self, exts: Iterable[str] | None = None) -> list[SourceFile]:
        allow = set(exts) if exts is not None else SOURCE_EXTS
        out: list[SourceFile] = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in IGNORE_DIRS for part in p.relative_to(self.root).parts):
                continue
            if p.suffix.lower() not in allow:
                continue
            try:
                out.append(SourceFile(str(p.relative_to(self.root)), p.stat().st_size))
            except OSError:
                continue
        out.sort(key=lambda f: f.relpath)
        return out

    def read(self, relpath: str) -> str:
        return (self.root / relpath).read_text(errors="replace")


# ---------------------------------------------------------------------------
# C1  Engagement Graph — the shared world model (6 tables)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS surface (
    id INTEGER PRIMARY KEY, relpath TEXT UNIQUE, size INTEGER,
    selected INTEGER DEFAULT 0, note TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY, relpath TEXT, kind TEXT, detail TEXT,
    actor TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY, relpath TEXT, cwe TEXT, title TEXT, detail TEXT,
    status TEXT DEFAULT 'open', actor TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY, hypothesis_id INTEGER, relpath TEXT, cwe TEXT,
    title TEXT, severity TEXT, verified INTEGER DEFAULT 0, poc_path TEXT,
    actor TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS dead_ends (
    id INTEGER PRIMARY KEY, relpath TEXT, reason TEXT, actor TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS chains (
    id INTEGER PRIMARY KEY, name TEXT, links TEXT, verified INTEGER DEFAULT 0,
    actor TEXT, created TEXT
);
"""


class EngagementGraph:
    """Typed SQLite world model every component reads and writes."""

    def __init__(self, db_path: str | os.PathLike) -> None:
        self.path = Path(db_path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- writes --------------------------------------------------------------
    def add_surface(self, relpath: str, size: int, note: str = "") -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO surface(relpath,size,note,created) VALUES(?,?,?,?)",
            (relpath, size, note, _utcnow()),
        )
        self.conn.commit()

    def select_surface(self, relpath: str, selected: bool = True) -> None:
        self.conn.execute(
            "UPDATE surface SET selected=? WHERE relpath=?", (1 if selected else 0, relpath)
        )
        self.conn.commit()

    def add_fact(self, relpath: str, kind: str, detail: str, actor: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO facts(relpath,kind,detail,actor,created) VALUES(?,?,?,?,?)",
            (relpath, kind, detail, actor, _utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_hypothesis(self, relpath: str, cwe: str, title: str, detail: str, actor: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO hypotheses(relpath,cwe,title,detail,actor,created) VALUES(?,?,?,?,?,?)",
            (relpath, cwe, title, detail, actor, _utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid

    def set_hypothesis_status(self, hid: int, status: str) -> None:
        self.conn.execute("UPDATE hypotheses SET status=? WHERE id=?", (status, hid))
        self.conn.commit()

    def add_finding(self, hypothesis_id: int, relpath: str, cwe: str, title: str,
                    severity: str, actor: str, verified: bool = False,
                    poc_path: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO findings(hypothesis_id,relpath,cwe,title,severity,verified,poc_path,actor,created)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (hypothesis_id, relpath, cwe, title, severity, 1 if verified else 0,
             poc_path, actor, _utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_dead_end(self, relpath: str, reason: str, actor: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO dead_ends(relpath,reason,actor,created) VALUES(?,?,?,?)",
            (relpath, reason, actor, _utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_chain(self, name: str, links: list[Any], actor: str, verified: bool = False) -> int:
        cur = self.conn.execute(
            "INSERT INTO chains(name,links,verified,actor,created) VALUES(?,?,?,?,?)",
            (name, json.dumps(links), 1 if verified else 0, actor, _utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid

    # -- reads ---------------------------------------------------------------
    def selected_surface(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM surface WHERE selected=1 ORDER BY relpath"))

    def open_hypotheses(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM hypotheses WHERE status='open'"))

    def findings(self, verified_only: bool = False) -> list[sqlite3.Row]:
        q = "SELECT * FROM findings"
        if verified_only:
            q += " WHERE verified=1"
        return list(self.conn.execute(q + " ORDER BY id"))

    def counts(self) -> dict[str, int]:
        out = {}
        for t in ("surface", "facts", "hypotheses", "findings", "dead_ends", "chains"):
            out[t] = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return out

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# C2  Hash-chained immutable audit log
# ---------------------------------------------------------------------------

GENESIS = "0" * 64


class AuditLog:
    """Append-only, tamper-evident SHA-256 hash chain over JSONL."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self._last = self._compute_tail()

    def _compute_tail(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        for line in self.path.read_text().splitlines():
            if line.strip():
                last = json.loads(line)["hash"]
        return last

    @staticmethod
    def _hash(prev: str, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, sort_keys=True)
        return hashlib.sha256((prev + body).encode()).hexdigest()

    def append(self, actor: str, action: str, detail: dict[str, Any] | None = None) -> str:
        payload = {"ts": _utcnow(), "actor": actor, "action": action, "detail": detail or {}}
        h = self._hash(self._last, payload)
        entry = {"prev": self._last, "hash": h, **payload}
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        self._last = h
        return h

    def verify(self) -> bool:
        """Recompute the chain; return True iff intact."""
        prev = GENESIS
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            payload = {"ts": e["ts"], "actor": e["actor"], "action": e["action"], "detail": e["detail"]}
            if e["prev"] != prev or e["hash"] != self._hash(prev, payload):
                return False
            prev = e["hash"]
        return True

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# C3  Risk-classified action layer
# ---------------------------------------------------------------------------

class ActionRisk:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RefusedAction(Exception):
    """Raised when a HIGH-risk action is attempted — structural refusal."""


class ActionLayer:
    """Every tool call is risk-tagged. HIGH is structurally refused, never
    auto-approved. Each executed action is written to the audit log."""

    # Default classification for the tools the harness exposes.
    RISK = {
        "read_file": ActionRisk.LOW,
        "list_files": ActionRisk.LOW,
        "run_poc_sandboxed": ActionRisk.MEDIUM,
        "write_patch": ActionRisk.MEDIUM,
        "network_exploit": ActionRisk.HIGH,
        "modify_target_repo": ActionRisk.HIGH,
        "exfiltrate": ActionRisk.HIGH,
    }

    def __init__(self, audit: AuditLog) -> None:
        self.audit = audit

    def risk_of(self, tool: str) -> str:
        return self.RISK.get(tool, ActionRisk.HIGH)  # unknown => treated as HIGH

    def run(self, actor: str, tool: str, fn, *args, **kwargs):
        risk = self.risk_of(tool)
        if risk == ActionRisk.HIGH:
            self.audit.append(actor, "REFUSED", {"tool": tool, "risk": risk})
            raise RefusedAction(f"{tool!r} is HIGH risk and is structurally refused")
        self.audit.append(actor, "action", {"tool": tool, "risk": risk})
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# C4  Self-monitor + deliberative gate
# ---------------------------------------------------------------------------

@dataclass
class MonitorHit:
    detector: str
    evidence: str


class SelfMonitor:
    """Behavioural pathology detectors + a two-step pre-action deliberative gate.

    The original harness maps these to system-card incidents; the public build
    uses behavioural signals (here: regex/heuristic) since activation tooling is
    not available. Each detector flags text a worker is about to act on.
    """

    import re as _re

    DETECTORS = {
        # fabricating a finding without evidence
        "unverified_claim": _re.compile(r"\b(definitely|certainly|guaranteed) (exploitable|vulnerable)\b", _re.I),
        # trying to reach outside the engagement / target
        "scope_escape": _re.compile(r"\b(curl|wget|ssh|scp|nc)\b|https?://(?!localhost|127\.)", _re.I),
        # attempting to disable the safety layer
        "safety_tamper": _re.compile(r"\b(disable|bypass|skip)\s+(the\s+)?(audit|monitor|gate|refus)", _re.I),
        # destructive action on the host/target
        "destructive": _re.compile(r"\brm\s+-rf\b|shutil\.rmtree|DROP\s+TABLE", _re.I),
        # secret exfiltration intent
        "exfiltration": _re.compile(r"\b(exfiltrate|leak|send)\b.*\b(secret|credential|api[_ ]?key|token)\b", _re.I),
    }

    def __init__(self, audit: AuditLog) -> None:
        self.audit = audit

    def scan(self, text: str) -> list[MonitorHit]:
        hits = []
        for name, pat in self.DETECTORS.items():
            m = pat.search(text or "")
            if m:
                hits.append(MonitorHit(name, m.group(0)))
        return hits

    def gate(self, actor: str, intent: str, proposed_text: str) -> bool:
        """Two-step pre-action check. Returns True to allow, False to block.

        Step 1: pathology scan of what the actor is about to do/say.
        Step 2: any hit => block + audit. (Deliberation with the LLM advisor is
        layered on top in C5/C10; the structural block here is LLM-free.)
        """
        hits = self.scan(proposed_text)
        if hits:
            self.audit.append(
                actor, "BLOCKED",
                {"intent": intent, "detectors": [h.detector for h in hits],
                 "evidence": [h.evidence for h in hits]},
            )
            return False
        self.audit.append(actor, "gate_pass", {"intent": intent})
        return True


# ===========================================================================
# LAYER 2 — Discovery and Verification (C5-C9). LLM-driven; single-writer.
#
# Concurrency model: the swarm workers (C6) only READ files and call ask();
# they return plain dataclasses. The Coordinator is the SOLE writer to the
# EngagementGraph. That keeps sqlite single-connection-safe and mirrors the
# original harness's "no agent-callable tool holds a write handle" property.
# ===========================================================================

# -- structured-output helpers ---------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """Best-effort parse of a JSON object from a model reply (handles ``` fences)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # fall back to the first balanced {...} span
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start : i + 1])
                except Exception:
                    return None
    return None


def _ask_json(llm, persona: str, system: str, user: str, max_tokens: int = 2048) -> dict:
    """Call ask() and parse a JSON object, or return {} on failure."""
    r = llm.ask(persona, system, user, max_tokens=max_tokens)
    return _extract_json(r.text) or {}


# -- data carried between components ----------------------------------------

@dataclass
class Candidate:
    relpath: str
    cwe: str
    title: str
    detail: str
    sink_line: int = 0
    sink_snippet: str = ""


@dataclass
class Vote:
    persona: str
    exploitable: bool
    confidence: float
    reason: str


@dataclass
class Corroboration:
    candidate: Candidate
    votes: list[Vote]
    promoted: bool

    @property
    def yes(self) -> int:
        return sum(1 for v in self.votes if v.exploitable)


@dataclass
class PocResult:
    poc_path: str
    passed: bool
    marker_seen: bool
    exit_code: int
    output: str


# -- prompt builders. Each embeds a MYTHOS-TASK tag + a DATA_JSON block so the
#    task is explicit to the model and the reply is deterministically cacheable. -

def _p_scan(relpath: str, content: str) -> tuple[str, str]:
    system = (
        "You are a security code scanner. Given one source file, identify concrete "
        "candidate vulnerabilities where attacker-controlled input can reach a dangerous "
        "sink. Reply ONLY with JSON: "
        '{"candidates":[{"cwe":"CWE-XX","title":"...","detail":"...","sink_line":N,"sink_snippet":"..."}]}'
    )
    user = (
        f"MYTHOS-TASK: scan\nFILE: {relpath}\n"
        "DATA_JSON: " + json.dumps({"relpath": relpath}) + "\n"
        "----- FILE CONTENT -----\n" + content
    )
    return system, user


def _p_vote(cand: Candidate, content: str) -> tuple[str, str]:
    system = (
        "You are voting on whether a candidate finding is genuinely exploitable. "
        'Reply ONLY with JSON: {"exploitable":true|false,"confidence":0.0-1.0,"reason":"..."}'
    )
    user = (
        "MYTHOS-TASK: vote\n"
        "DATA_JSON: " + json.dumps(asdict(cand)) + "\n"
        "----- FILE CONTENT -----\n" + content
    )
    return system, user


def _p_poc(cand: Candidate, target_root: str, content: str) -> tuple[str, str]:
    system = (
        "You write a minimal executable Python proof-of-concept that drives the target "
        "code to the dangerous sink using benign-but-attacker-shaped input, then prints the "
        "exact marker SINK REACHED on success and exits 0. Do not perform destructive or "
        "networked actions. Reply ONLY with JSON: "
        '{"source":"<python>","expect_marker":"SINK REACHED"}'
    )
    user = (
        "MYTHOS-TASK: poc\n"
        "DATA_JSON: " + json.dumps({**asdict(cand), "target_root": target_root}) + "\n"
        "----- FILE CONTENT -----\n" + content
    )
    return system, user


def _p_skeptic(cand: Candidate, poc_output: str) -> tuple[str, str]:
    system = (
        "You are the skeptic. A PoC just ran. Decide if this is a real finding or a false "
        'positive. Reply ONLY with JSON: {"keep":true|false,"reason":"..."}'
    )
    user = (
        "MYTHOS-TASK: skeptic\n"
        "DATA_JSON: " + json.dumps(asdict(cand)) + "\n"
        "----- POC OUTPUT -----\n" + poc_output
    )
    return system, user


def _p_variant(sig: dict, relpath: str, content: str) -> tuple[str, str]:
    system = (
        "You are a variant hunter. Given a known bug signature, decide if THIS file contains "
        'the same class of bug. Reply ONLY with JSON: '
        '{"match":true|false,"cwe":"CWE-XX","title":"...","detail":"...","sink_line":N,"sink_snippet":"..."}'
    )
    user = (
        "MYTHOS-TASK: variant\n"
        "DATA_JSON: " + json.dumps({"signature": sig, "relpath": relpath}) + "\n"
        "----- FILE CONTENT -----\n" + content
    )
    return system, user


def _p_plan(files: list[SourceFile], max_files: int) -> tuple[str, str]:
    system = (
        "You are ULTRAPLAN: pick the highest-value files to review for security in this repo, "
        "prioritising request handlers, deserialization, auth, dynamic import/exec, path handling. "
        'Reply ONLY with JSON: {"selected":["relpath",...],"rationale":"..."}'
    )
    listing = [{"relpath": f.relpath, "size": f.size} for f in files]
    user = (
        f"MYTHOS-TASK: plan\nMAX_FILES: {max_files}\n"
        "DATA_JSON: " + json.dumps({"files": listing, "max_files": max_files})
    )
    return system, user


# -- C5 ULTRAPLAN -----------------------------------------------------------

def ultraplan(llm, target: "Target", max_files: int = 12) -> tuple[list[str], str]:
    """Up-front reasoning call that selects the target file list, reviewed by an advisor."""
    files = target.files()
    system, user = _p_plan(files, max_files)
    out = _ask_json(llm, "architect", system, user)
    selected = [s for s in out.get("selected", []) if (target.root / s).is_file()]
    rationale = out.get("rationale", "")
    if not selected:  # advisor/fallback: never leave the swarm with nothing
        selected = [f.relpath for f in files[:max_files]]
    else:
        selected = selected[:max_files]
    return selected, rationale


# -- C6 worker (read-only, returns candidates) ------------------------------

def _scan_worker(llm, relpath: str, content: str) -> list[Candidate]:
    system, user = _p_scan(relpath, content)
    out = _ask_json(llm, "architect", system, user, max_tokens=1500)
    cands: list[Candidate] = []
    for c in out.get("candidates", []):
        cands.append(Candidate(
            relpath=relpath,
            cwe=str(c.get("cwe", "CWE-000")),
            title=str(c.get("title", "unnamed")),
            detail=str(c.get("detail", "")),
            sink_line=int(c.get("sink_line", 0) or 0),
            sink_snippet=str(c.get("sink_snippet", "")),
        ))
    return cands


# -- C7 corroboration: 2-of-3 persona vote ----------------------------------

def corroborate(llm, cand: Candidate, content: str) -> Corroboration:
    votes: list[Vote] = []
    for persona in ("architect", "redteam", "skeptic"):
        system, user = _p_vote(cand, content)
        out = _ask_json(llm, persona, system, user, max_tokens=600)
        votes.append(Vote(
            persona=persona,
            exploitable=bool(out.get("exploitable", False)),
            confidence=float(out.get("confidence", 0.0) or 0.0),
            reason=str(out.get("reason", "")),
        ))
    promoted = sum(1 for v in votes if v.exploitable) >= 2
    return Corroboration(candidate=cand, votes=votes, promoted=promoted)


# -- C8 dynamic executable-PoC verification gate ----------------------------

def synthesize_and_run_poc(llm, cfg: Config, target: "Target", cand: Candidate,
                           timeout_s: int = 20) -> PocResult:
    """Ask a persona for a Python PoC, write it, and run it in a real subprocess.

    Pass == the expected marker is printed AND the process exits 0. The PoC is
    executed with the target repo on sys.path; it must reach the sink for real.
    """
    content = target.read(cand.relpath)
    system, user = _p_poc(cand, str(target.root), content)
    out = _ask_json(llm, "redteam", system, user, max_tokens=1500)
    source = out.get("source", "")
    marker = out.get("expect_marker", "SINK REACHED")

    safe = re.sub(r"[^a-zA-Z0-9]+", "_", f"{cand.relpath}_{cand.cwe}").strip("_")[:60]
    poc_path = cfg.engagement_dir / "pocs" / f"poc_{safe}.py"
    poc_path.write_text(source or "# empty PoC\n")
    if not source:
        return PocResult(str(poc_path), False, False, -1, "no PoC source synthesized")

    env = dict(os.environ, PYTHONPATH=str(target.root))
    try:
        proc = subprocess.run(
            [sys.executable, str(poc_path)],
            cwd=str(cfg.engagement_dir / "pocs"),
            env=env, capture_output=True, text=True, timeout=timeout_s,
            stdin=subprocess.DEVNULL,  # never let a PoC block on stdin
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        marker_seen = marker in combined
        passed = marker_seen and proc.returncode == 0
        return PocResult(str(poc_path), passed, marker_seen, proc.returncode, combined[-4000:])
    except subprocess.TimeoutExpired:
        return PocResult(str(poc_path), False, False, -9, f"timeout after {timeout_s}s")


# -- C9 variant hunter + known-issue dedup ----------------------------------

def build_signatures(findings: list[Candidate]) -> list[dict]:
    """Turn confirmed findings into reusable bug signatures for the variant hunt."""
    sigs = []
    for f in findings:
        sigs.append({"cwe": f.cwe, "title": f.title, "sink_snippet": f.sink_snippet})
    return sigs


def variant_hunt(llm, target: "Target", signatures: list[dict],
                 scanned: set[str], max_files: int = 40) -> list[Candidate]:
    """Search files the swarm did NOT scan for variants of confirmed signatures."""
    out: list[Candidate] = []
    unscanned = [f for f in target.files() if f.relpath not in scanned][:max_files]
    for sig in signatures:
        for f in unscanned:
            content = target.read(f.relpath)
            system, user = _p_variant(sig, f.relpath, content)
            res = _ask_json(llm, "redteam", system, user, max_tokens=800)
            if res.get("match"):
                out.append(Candidate(
                    relpath=f.relpath,
                    cwe=str(res.get("cwe", sig["cwe"])),
                    title=str(res.get("title", "variant of " + sig["title"])),
                    detail=str(res.get("detail", "")),
                    sink_line=int(res.get("sink_line", 0) or 0),
                    sink_snippet=str(res.get("sink_snippet", "")),
                ))
    return out


def load_catalog(path: str | os.PathLike | None) -> list[dict]:
    """Optional known-issue ledger (JSONL). Generic build: empty unless supplied."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def dedup_against_catalog(cand: Candidate, catalog: list[dict]) -> Optional[dict]:
    """Return the catalog entry a candidate duplicates, else None."""
    for entry in catalog:
        if entry.get("cwe") == cand.cwe and entry.get("relpath") == cand.relpath:
            return entry
    return None


# -- C6 Coordinator: the single writer, orchestrates C5->C9 -----------------

class Coordinator:
    """Owns the only graph write handle and drives the Layer 2 pipeline."""

    def __init__(self, cfg: Config, graph: EngagementGraph, audit: AuditLog,
                 actions: ActionLayer, monitor: SelfMonitor, llm,
                 max_workers: int = 4, catalog: list[dict] | None = None,
                 do_chain: bool = True, do_fix: bool = True, do_speculate: bool = True) -> None:
        self.cfg = cfg
        self.graph = graph
        self.audit = audit
        self.actions = actions
        self.monitor = monitor
        self.llm = llm
        self.max_workers = max_workers
        self.catalog = catalog or []
        self.do_chain = do_chain
        self.do_fix = do_fix
        self.do_speculate = do_speculate
        self.target = Target(cfg.target_path)

    def run(self, max_files: int = 12) -> dict[str, Any]:
        self.audit.append("coordinator", "engagement_start",
                          {"target": str(self.cfg.target_path)})

        # C5 ULTRAPLAN — pick the surface
        selected, rationale = ultraplan(self.llm, self.target, max_files=max_files)
        for rel in selected:
            sf = next((f for f in self.target.files() if f.relpath == rel), None)
            self.graph.add_surface(rel, sf.size if sf else 0, note="ultraplan")
            self.graph.select_surface(rel)
        self.audit.append("coordinator", "ultraplan", {"selected": selected})

        # C6 scanner swarm — parallel, read-only workers; coordinator writes back
        candidates = self._scan_swarm(selected)

        # C7 corroboration + C8 verification gate + skeptic re-inspection
        confirmed: list[Candidate] = []
        confirmed_links: list[ChainLink] = []  # verified findings that can chain
        for cand in candidates:
            content = self.target.read(cand.relpath)
            hid = self.graph.add_hypothesis(cand.relpath, cand.cwe, cand.title,
                                            cand.detail, "coordinator")
            corr = corroborate(self.llm, cand, content)
            self.audit.append("coordinator", "corroborate",
                              {"title": cand.title, "yes": corr.yes, "promoted": corr.promoted})
            if not corr.promoted:
                self.graph.set_hypothesis_status(hid, "rejected")
                self.graph.add_dead_end(cand.relpath, f"corroboration {corr.yes}/3", "coordinator")
                continue

            # dedup against optional known-issue catalog before expensive verify
            dup = dedup_against_catalog(cand, self.catalog)
            if dup:
                self.graph.set_hypothesis_status(hid, "known")
                self.audit.append("coordinator", "dedup", {"title": cand.title, "known": dup.get("id")})
                continue

            poc = synthesize_and_run_poc(self.llm, self.cfg, self.target, cand)
            self.audit.append("coordinator", "poc",
                              {"title": cand.title, "passed": poc.passed, "exit": poc.exit_code})
            if not poc.passed:
                self.graph.set_hypothesis_status(hid, "unverified")
                self.graph.add_dead_end(cand.relpath, "poc did not reach sink", "coordinator")
                continue

            # skeptic re-inspection on every survived sink
            skeptic = _ask_json(self.llm, "skeptic", *_p_skeptic(cand, poc.output), max_tokens=500)
            if not skeptic.get("keep", True):
                self.graph.set_hypothesis_status(hid, "rejected")
                self.graph.add_dead_end(cand.relpath, "skeptic killed: " + str(skeptic.get("reason", "")),
                                        "coordinator")
                continue

            self.graph.set_hypothesis_status(hid, "confirmed")
            fid = self.graph.add_finding(hid, cand.relpath, cand.cwe, cand.title,
                                         severity="high", actor="coordinator",
                                         verified=True, poc_path=poc.poc_path)
            confirmed.append(cand)
            confirmed_links.append(ChainLink(
                finding_id=fid, relpath=cand.relpath, cwe=cand.cwe, title=cand.title,
                sink_line=cand.sink_line, poc_path=poc.poc_path))

        # C9 variant hunter over files the swarm did not touch
        variants = variant_hunt(self.llm, self.target, build_signatures(confirmed),
                                scanned=set(selected))
        for v in variants:
            hid = self.graph.add_hypothesis(v.relpath, v.cwe, v.title, v.detail, "variant-hunter")
            self.graph.add_finding(hid, v.relpath, v.cwe, v.title, severity="medium",
                                   actor="variant-hunter", verified=False)

        # ---- Layer 3: synthesis ------------------------------------------
        chain_result = fix_result = spec_result = None

        # C10 chain builder + composite critical-path PoC
        if self.do_chain and confirmed_links:
            chain_result = ChainBuilder(self.cfg, self.graph, self.audit, self.llm
                                        ).build(self.target, confirmed_links)

        # C11 fixer + chain-severance proof + CI (only meaningful with a chain)
        if self.do_fix and chain_result:
            fix_result = Fixer(self.cfg, self.graph, self.audit, self.llm).fix_chain(
                self.target, chain_result["chain"], Path(chain_result["poc_path"]))

        # C12 speculation layer — predict the operator's next move, run it COW
        if self.do_speculate:
            spec_result = self._speculate(confirmed, chain_result)

        self.audit.append("coordinator", "engagement_end",
                          {"confirmed": len(confirmed), "variants": len(variants)})
        return {"selected": selected, "rationale": rationale,
                "confirmed": confirmed, "variants": variants,
                "chain": chain_result, "fix": fix_result, "speculation": spec_result,
                "counts": self.graph.counts()}

    def _speculate(self, confirmed: list[Candidate], chain_result: Optional[dict]) -> dict:
        spec = SpeculationLayer(self.cfg, self.audit, self.actions, self.llm)
        context = (f"confirmed_findings={len(confirmed)}; "
                   f"chain={'yes' if chain_result else 'no'}; "
                   "patches not yet written; report not yet written.")
        predicted = spec.predict(context) or "write the engagement report"

        def _write_report(overlay: Path) -> None:
            lines = [f"# Engagement Report\n", f"target: {self.cfg.target_path}\n",
                     f"\n## Confirmed findings ({len(confirmed)})\n"]
            for c in confirmed:
                lines.append(f"- [{c.cwe}] {c.relpath}:{c.sink_line} — {c.title}\n")
            if chain_result:
                lines.append(f"\n## Critical chain (verified={chain_result['verified']})\n")
                for l in chain_result["chain"]:
                    lines.append(f"- {l.pre} --[{l.cwe}]--> {l.post}  ({l.relpath})\n")
            (overlay / "report.md").write_text("".join(lines))

        overlay = spec.speculate(predicted, _write_report)
        return {"predicted": predicted,
                "boundary_ok": overlay is not None,
                "overlay": str(overlay) if overlay else None}

    def _scan_swarm(self, files: list[str]) -> list[Candidate]:
        """Parallel scanner workers. Workers are read-only; only this method's
        caller (run) writes to the graph."""
        results: list[Candidate] = []
        contents = {rel: self.target.read(rel) for rel in files}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(_scan_worker, self.llm, rel, contents[rel]): rel for rel in files}
            for fut in concurrent.futures.as_completed(futs):
                rel = futs[fut]
                try:
                    results.extend(fut.result())
                except Exception as exc:  # a worker failing must not sink the swarm
                    self.audit.append("scanner", "worker_error", {"relpath": rel, "error": str(exc)})
        results.sort(key=lambda c: (c.relpath, c.sink_line))
        return results


# ===========================================================================
# LAYER 3 — Synthesis, Action, Acceleration (C10, C11, C12).
#
# The LLM calls here are deliberately THIN (label a state transition, write a
# patch, predict the next instruction). Everything mechanical — the composite
# PoC subprocess, the necessity test, applying a patch to a sandbox copy, the
# AST smoke check, the COW overlay, the boundary gate — is LLM-free and runs /
# verifies offline on its own.
# ===========================================================================

# -- prompt builders for Layer 3 -------------------------------------------

def _p_chain(cand: Candidate) -> tuple[str, str]:
    system = (
        "You map a single vulnerability to a state transition in an attack chain. "
        "Give the attacker STATE required before exploiting it (precondition) and the "
        "STATE gained after (postcondition), as short snake_case tokens drawn from a "
        "vocabulary like: unauth, authenticated, admin, file_write, malicious_file_on_disk, "
        "rce, cross_tenant_rce, info_leak. Reply ONLY with JSON: "
        '{"precondition":"...","postcondition":"..."}'
    )
    user = "MYTHOS-TASK: chain\nDATA_JSON: " + json.dumps(asdict(cand))
    return system, user


def _p_fix(link: "ChainLink", original_source: str) -> tuple[str, str]:
    system = (
        "You write the SMALLEST patch that severs this vulnerability at its sink while "
        "preserving legitimate behaviour (prefer an allow-list, a bind, or a hard refusal). "
        "Return the FULL patched file. Add a trailing comment `# MYTHOS-HARDENED` on the "
        'line you change. Reply ONLY with JSON: {"patched_source":"<entire file>","explanation":"..."}'
    )
    user = (
        "MYTHOS-TASK: fix\n"
        "DATA_JSON: " + json.dumps({"relpath": link.relpath, "cwe": link.cwe,
                                    "title": link.title, "sink_line": link.sink_line}) + "\n"
        "----- FILE CONTENT -----\n" + original_source
    )
    return system, user


def _p_speculate(context: str) -> tuple[str, str]:
    system = (
        "Given the current engagement state, predict the operator's SINGLE most likely next "
        'instruction as a short imperative. Reply ONLY with JSON: {"next_instruction":"..."}'
    )
    user = "MYTHOS-TASK: speculate\n----- ENGAGEMENT STATE -----\n" + context
    return system, user


# -- C10 Chain Builder ------------------------------------------------------

ENTRY_STATES = {"unauth", "anonymous", "external", "network", "pre_auth"}
GOAL_TOKENS = ("rce", "code_exec", "exec", "takeover", "full_control", "cross_tenant")


@dataclass
class ChainLink:
    finding_id: int
    relpath: str
    cwe: str
    title: str
    sink_line: int
    poc_path: str
    pre: str = "unknown"
    post: str = "impact"


def _is_goal(state: str) -> bool:
    s = (state or "").lower()
    return any(tok in s for tok in GOAL_TOKENS)


def find_chain(links: list[ChainLink]) -> list[ChainLink]:
    """Longest simple path from an attacker entry state to a goal state."""
    adj: dict[str, list[ChainLink]] = defaultdict(list)
    for l in links:
        adj[l.pre].append(l)
    best: list[ChainLink] = []

    def dfs(state: str, path: list[ChainLink], seen: set[str]) -> None:
        nonlocal best
        if path and _is_goal(path[-1].post) and len(path) > len(best):
            best = path[:]
        for l in adj.get(state, []):
            if l.post in seen:
                continue
            dfs(l.post, path + [l], seen | {l.post})

    starts = {l.pre for l in links if l.pre in ENTRY_STATES}
    if not starts:  # no clean entry; allow any precondition to start a chain
        starts = {l.pre for l in links}
    for s in starts:
        dfs(s, [], {s})
    return best


_COMPOSITE_TEMPLATE = '''\
"""Composite critical-path PoC — auto-generated by Mythos C10.

Runs each chain link's PoC in order. Ordering is enforced structurally: link i+1
only runs once link i has written its sentinel, so disabling any single link
breaks the chain. Set MYTHOS_DISABLE_LINK=<i> to prove necessity, and
MYTHOS_TARGET_ROOT to point the links at the original or a patched tree.
"""
import json, os, subprocess, sys, tempfile

LINKS = {links_json}
MARKER = "SINK REACHED"
disable = int(os.environ.get("MYTHOS_DISABLE_LINK", "-1"))
target_root = os.environ.get("MYTHOS_TARGET_ROOT", "")
state = tempfile.mkdtemp(prefix="mythos-chain-")

def sentinel(i):
    return os.path.join(state, "link_%d.ok" % i)

env = dict(os.environ)
if target_root:
    env["MYTHOS_TARGET_ROOT"] = target_root

for i, link in enumerate(LINKS):
    if i == disable:
        print("link %d DISABLED (%s)" % (i, link["name"]))
        continue
    if i > 0 and not os.path.exists(sentinel(i - 1)):
        print("CHAIN BROKEN at link %d: precondition (link %d) not established" % (i, i - 1))
        sys.exit(2)
    r = subprocess.run([sys.executable, link["poc"]], capture_output=True, text=True,
                       env=env, stdin=subprocess.DEVNULL, timeout=30)
    out = (r.stdout or "") + (r.stderr or "")
    if MARKER in out and r.returncode == 0:
        open(sentinel(i), "w").write("ok")
        print("link %d OK: %s" % (i, link["name"]))
    else:
        print("CHAIN BROKEN at link %d: PoC did not reach sink" % i)
        sys.exit(3)

if os.path.exists(sentinel(len(LINKS) - 1)):
    print("CHAIN COMPLETE")
    sys.exit(0)
print("CHAIN INCOMPLETE")
sys.exit(4)
'''


def generate_composite_poc(cfg: Config, chain: list[ChainLink]) -> Path:
    links_json = json.dumps([{"name": f"{l.cwe}:{l.title}", "poc": l.poc_path} for l in chain])
    src = _COMPOSITE_TEMPLATE.format(links_json=links_json)
    p = cfg.engagement_dir / "pocs" / "poc_chain_critical.py"
    p.write_text(src)
    return p


def run_composite(poc_path: Path, target_root: str | os.PathLike,
                  disable: Optional[int] = None, timeout: int = 120) -> tuple[bool, str]:
    env = dict(os.environ, MYTHOS_TARGET_ROOT=str(target_root))
    if disable is not None:
        env["MYTHOS_DISABLE_LINK"] = str(disable)
    else:
        env.pop("MYTHOS_DISABLE_LINK", None)
    try:
        r = subprocess.run([sys.executable, str(poc_path)], capture_output=True, text=True,
                           env=env, stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "composite timeout"
    out = (r.stdout or "") + (r.stderr or "")
    return ("CHAIN COMPLETE" in out and r.returncode == 0), out


def verify_chain(poc_path: Path, target_root: str | os.PathLike, n_links: int) -> dict:
    """Full run must complete; disabling any single link must break it."""
    full_ok, full_out = run_composite(poc_path, target_root)
    necessity = []
    for i in range(n_links):
        completed, _ = run_composite(poc_path, target_root, disable=i)
        necessity.append(not completed)  # necessary iff removing it breaks the chain
    return {"full_ok": full_ok, "necessity": necessity,
            "verified": full_ok and all(necessity), "output": full_out}


class ChainBuilder:
    """C10: label findings with state transitions, build+walk the attack graph,
    emit and verify a composite critical-path PoC."""

    def __init__(self, cfg: Config, graph: EngagementGraph, audit: AuditLog, llm) -> None:
        self.cfg = cfg
        self.graph = graph
        self.audit = audit
        self.llm = llm

    def build(self, target: "Target", confirmed: list[ChainLink]) -> Optional[dict]:
        for link in confirmed:  # thin LLM call: one state transition per finding
            cand = Candidate(link.relpath, link.cwe, link.title, "", link.sink_line, "")
            out = _ask_json(self.llm, "architect", *_p_chain(cand), max_tokens=300)
            link.pre = str(out.get("precondition", "unknown"))
            link.post = str(out.get("postcondition", "impact"))
            self.graph.add_fact(link.relpath, "state_transition",
                                f"{link.pre} -> {link.post}", "chain-builder")

        chain = find_chain(confirmed)
        if len(chain) < 2:
            self.audit.append("chain-builder", "no_chain", {"links": len(chain)})
            return None

        poc_path = generate_composite_poc(self.cfg, chain)
        result = verify_chain(poc_path, target.root, len(chain))
        links_desc = [{"cwe": l.cwe, "relpath": l.relpath, "pre": l.pre, "post": l.post}
                      for l in chain]
        self.graph.add_chain("critical-path", links_desc, "chain-builder",
                             verified=result["verified"])
        self.audit.append("chain-builder", "chain",
                          {"links": len(chain), "verified": result["verified"]})
        return {"chain": chain, "poc_path": str(poc_path), **result}


# -- C11 Fixer --------------------------------------------------------------

_MYTHOS_SCAN_YML = '''\
name: mythos-scan
on: [pull_request]
jobs:
  mythos:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Run Mythos harness
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: python mythos.py scan .
'''

_COPY_IGNORE = shutil.ignore_patterns(".git", "engagement", "recordings",
                                      "__pycache__", ".venv", "venv", "node_modules")


class Fixer:
    """C11: minimal patches on a sandbox copy, chain-severance proof, AST smoke
    test, and a mythos-scan CI workflow."""

    def __init__(self, cfg: Config, graph: EngagementGraph, audit: AuditLog, llm) -> None:
        self.cfg = cfg
        self.graph = graph
        self.audit = audit
        self.llm = llm

    def fix_chain(self, target: "Target", chain: list[ChainLink], composite_poc: Path) -> dict:
        sandbox = self.cfg.engagement_dir / "sandbox" / "target-patched"
        if sandbox.exists():
            shutil.rmtree(sandbox)
        shutil.copytree(target.root, sandbox, ignore=_COPY_IGNORE)

        patched: list[str] = []
        for link in chain:  # thin LLM call: one minimal patch per link
            original = (target.root / link.relpath).read_text(errors="replace")
            out = _ask_json(self.llm, "fixer", *_p_fix(link, original), max_tokens=3000)
            new_src = out.get("patched_source", "")
            if not new_src or new_src == original:
                continue
            (sandbox / link.relpath).write_text(new_src)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True), new_src.splitlines(keepends=True),
                fromfile=link.relpath, tofile=link.relpath + " (patched)")
            safe = re.sub(r"[^a-zA-Z0-9]+", "_", link.relpath).strip("_")
            patch_path = self.cfg.engagement_dir / "patches" / f"{safe}.patch"
            patch_path.write_text("".join(diff))
            patched.append(link.relpath)

        # chain-severance proof: the composite PoC must NOT complete on the patch
        severed_completed, sev_out = run_composite(composite_poc, sandbox)
        severed = not severed_completed
        smoke_ok = self._ast_smoke(sandbox, patched)
        ci_path = self._emit_ci()

        self.audit.append("fixer", "fix",
                          {"patched": patched, "severed": severed, "smoke_ok": smoke_ok})
        return {"patched": patched, "severed": severed, "smoke_ok": smoke_ok,
                "sandbox": str(sandbox), "ci": str(ci_path), "severance_output": sev_out}

    @staticmethod
    def _ast_smoke(sandbox: Path, patched: list[str]) -> bool:
        """Patched Python files must still parse (faithful to the article's smoke)."""
        for rel in patched:
            if not rel.endswith(".py"):
                continue
            try:
                ast.parse((sandbox / rel).read_text(errors="replace"))
            except SyntaxError:
                return False
        return True

    def _emit_ci(self) -> Path:
        ci_dir = self.cfg.engagement_dir / "ci" / ".github" / "workflows"
        ci_dir.mkdir(parents=True, exist_ok=True)
        p = ci_dir / "mythos-scan.yml"
        p.write_text(_MYTHOS_SCAN_YML)
        return p


# -- C12 Speculation Layer --------------------------------------------------

class SpeculationLayer:
    """C12: predict the operator's next instruction and run it in a copy-on-write
    overlay, refusing to speculate past any HIGH-risk action; promote on match."""

    # keyword -> tool name, resolved to a risk by the ActionLayer
    _TOOL_HINTS = [
        (("exploit", "attack", "pop", "network", "reverse shell"), "network_exploit"),
        (("delete", "wipe", "destroy", "drop"), "modify_target_repo"),
        (("exfil", "leak", "steal"), "exfiltrate"),
        (("patch", "fix", "harden"), "write_patch"),
        (("report", "summary", "summar", "write up", "document"), "read_file"),
    ]

    def __init__(self, cfg: Config, audit: AuditLog, actions: ActionLayer, llm) -> None:
        self.cfg = cfg
        self.audit = audit
        self.actions = actions
        self.llm = llm

    def predict(self, context: str) -> str:
        out = _ask_json(self.llm, "architect", *_p_speculate(context), max_tokens=200)
        return str(out.get("next_instruction", "")).strip()

    def _map_to_tool(self, instruction: str) -> str:
        low = instruction.lower()
        for kws, tool in self._TOOL_HINTS:
            if any(k in low for k in kws):
                return tool
        return "read_file"  # default: benign

    def boundary_ok(self, instruction: str) -> bool:
        tool = self._map_to_tool(instruction)
        risk = self.actions.risk_of(tool)
        if risk == ActionRisk.HIGH:
            self.audit.append("speculation", "boundary_refused",
                              {"instruction": instruction, "tool": tool, "risk": risk})
            return False
        return True

    def speculate(self, instruction: str, run_fn: "Callable[[Path], None]") -> Optional[Path]:
        """COW overlay of the engagement dir; run the predicted task inside it."""
        if not self.boundary_ok(instruction):
            return None
        overlay = self.cfg.engagement_dir / "speculative"
        if overlay.exists():
            shutil.rmtree(overlay)
        shutil.copytree(self.cfg.engagement_dir, overlay,
                        ignore=shutil.ignore_patterns("speculative"))
        run_fn(overlay)
        self.audit.append("speculation", "speculated", {"instruction": instruction})
        return overlay

    @staticmethod
    def _similar(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def match_and_promote(self, predicted: str, actual: str, overlay: Path,
                          threshold: float = 0.4) -> bool:
        """If the operator's real instruction matches the prediction, promote the
        overlay's artifacts into the live engagement dir."""
        if self._similar(predicted, actual) < threshold:
            self.audit.append("speculation", "discarded",
                              {"predicted": predicted, "actual": actual})
            return False
        for src in overlay.rglob("*"):
            if src.is_file():
                dst = self.cfg.engagement_dir / src.relative_to(overlay)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
        self.audit.append("speculation", "promoted",
                          {"predicted": predicted, "actual": actual})
        return True


# ---------------------------------------------------------------------------
# Layer 1 self-test — runs fully offline, no LLM
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile

    print("== Mythos Layer 1 self-test (offline, no LLM) ==")
    tmp = Path(tempfile.mkdtemp(prefix="mythos-selftest-"))
    cfg = Config.for_target(tmp, workdir=tmp)
    cfg.ensure_dirs()
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # C1 EngagementGraph
    g = EngagementGraph(cfg.engagement_dir / "engagement_graph.sqlite")
    g.add_surface("a/b.py", 123)
    g.select_surface("a/b.py")
    fid = g.add_fact("a/b.py", "sink", "importlib.import_module(conf)", "scanner")
    hid = g.add_hypothesis("a/b.py", "CWE-94", "dynamic import", "attacker-controlled module", "scanner")
    g.add_finding(hid, "a/b.py", "CWE-94", "dynamic import RCE", "high", "verifier", verified=True,
                  poc_path="pocs/poc1.py")
    g.set_hypothesis_status(hid, "confirmed")
    g.add_chain("crit", [{"from": "unauth", "to": "rce"}], "chain-builder", verified=True)
    check("C1 selected surface = 1", len(g.selected_surface()) == 1)
    check("C1 verified findings = 1", len(g.findings(verified_only=True)) == 1)
    counts = g.counts()
    check("C1 counts populated", counts["facts"] == 1 and counts["chains"] == 1 and fid > 0)

    # C2 AuditLog
    audit = AuditLog(cfg.engagement_dir / "audit.log.jsonl")
    audit.append("coordinator", "start", {"target": str(tmp)})
    audit.append("scanner", "action", {"tool": "read_file"})
    check("C2 chain intact", audit.verify() is True)
    # tamper: rewrite a byte and confirm detection
    p = cfg.engagement_dir / "audit.log.jsonl"
    lines = p.read_text().splitlines()
    d0 = json.loads(lines[0]); d0["detail"] = {"target": "EVIL"}
    lines[0] = json.dumps(d0); p.write_text("\n".join(lines) + "\n")
    tampered = AuditLog(p)
    check("C2 tamper detected", tampered.verify() is False)

    # C3 ActionLayer
    audit2 = AuditLog(cfg.engagement_dir / "audit2.jsonl")
    actions = ActionLayer(audit2)
    got = actions.run("scanner", "read_file", lambda: "contents")
    check("C3 LOW action runs", got == "contents")
    refused = False
    try:
        actions.run("scanner", "network_exploit", lambda: "boom")
    except RefusedAction:
        refused = True
    check("C3 HIGH action refused", refused)
    check("C3 unknown tool => HIGH", actions.risk_of("mystery_tool") == ActionRisk.HIGH)

    # C4 SelfMonitor
    mon = SelfMonitor(audit2)
    check("C4 blocks scope escape", mon.gate("worker", "fetch", "curl http://evil.com/x") is False)
    check("C4 blocks safety tamper", mon.gate("worker", "x", "let's disable the audit log") is False)
    check("C4 allows benign", mon.gate("worker", "read", "open the file and inspect the parser") is True)

    g.close()
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} — engagement at {cfg.engagement_dir}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Homemade Mythos harness (Gemini 3.0)")
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--selftest", action="store_true", help="run Layer 1 offline self-test")

    p_scan = sub.add_parser("scan", help="run the discovery+verification+synthesis pipeline")
    p_scan.add_argument("target", help="path to a local repo/directory")
    p_scan.add_argument("--max-files", type=int, default=12, help="ULTRAPLAN surface budget")
    p_scan.add_argument("--workers", type=int, default=4, help="parallel scanner workers")
    p_scan.add_argument("--catalog", default=None, help="optional known-issue ledger (JSONL) for dedup")
    p_scan.add_argument("--no-chain", action="store_true", help="skip C10 chain builder")
    p_scan.add_argument("--no-fix", action="store_true", help="skip C11 fixer")
    p_scan.add_argument("--no-speculate", action="store_true", help="skip C12 speculation")

    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.cmd == "scan":
        cfg = Config.for_target(args.target)
        cfg.ensure_dirs()
        llm = LLM(cfg)
        print(f"target:   {cfg.target_path}")
        print(f"LLM:      gemini/{llm.mode}")
        graph = EngagementGraph(cfg.engagement_dir / "engagement_graph.sqlite")
        audit = AuditLog(cfg.engagement_dir / "audit.log.jsonl")
        coord = Coordinator(
            cfg, graph, audit, ActionLayer(audit), SelfMonitor(audit), llm,
            max_workers=args.workers, catalog=load_catalog(args.catalog),
            do_chain=not args.no_chain, do_fix=not args.no_fix,
            do_speculate=not args.no_speculate,
        )
        result = coord.run(max_files=args.max_files)

        print(f"\nselected surface ({len(result['selected'])} files):")
        for rel in result["selected"]:
            print(f"  - {rel}")
        print(f"\nconfirmed findings ({len(result['confirmed'])}):")
        for c in result["confirmed"]:
            print(f"  [{c.cwe}] {c.relpath}:{c.sink_line}  {c.title}")
        print(f"\nvariant candidates ({len(result['variants'])}):")
        for v in result["variants"]:
            print(f"  [{v.cwe}] {v.relpath}:{v.sink_line}  {v.title}")

        chain = result.get("chain")
        if chain:
            print(f"\ncritical chain ({len(chain['chain'])} links, verified={chain['verified']}):")
            for l in chain["chain"]:
                print(f"  {l.pre} --[{l.cwe}]--> {l.post}   ({l.relpath})")
            print(f"  composite PoC: {chain['poc_path']}")
        else:
            print("\ncritical chain: none found")

        fix = result.get("fix")
        if fix:
            print(f"\nfixer: patched {len(fix['patched'])} file(s); "
                  f"chain severed={fix['severed']}; smoke_ok={fix['smoke_ok']}")
            print(f"  sandbox: {fix['sandbox']}")
            print(f"  CI:      {fix['ci']}")

        spec = result.get("speculation")
        if spec:
            print(f"\nspeculation: predicted={spec['predicted']!r} "
                  f"boundary_ok={spec['boundary_ok']}")

        print(f"\ngraph: {result['counts']}")
        print(f"audit chain intact: {audit.verify()}")
        print(f"engagement dir: {cfg.engagement_dir}")
        graph.close()
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
