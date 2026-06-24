#!/usr/bin/env python3
"""Assemble the AST-dependency-graph teaching notebook (writes understanding_the_ast_graph.ipynb)."""
import json
from pathlib import Path

OUT = Path(__file__).with_name("understanding_the_ast_graph.ipynb")

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md(r"""
# Understanding the AST dependency graph

This notebook explains, runnably, the machinery behind
**`claude_code_from_scratch_v3_GRAPH.html`** — the interactive map where every notebook
*code cell* is a node and every *edge* is a real dependency between cells.

The whole graph rests on one idea: **you can read a block of Python without running it** by
parsing it into an **Abstract Syntax Tree (AST)** and asking two questions of every name in it:

- which names does this block **define**? (functions, classes, assignments)
- which names does this block **use**? (loads / references)

If cell *A* defines `llm` and cell *B* uses `llm`, then `B` depends on `A` — that's an edge
`A → B`. Nothing about this needs the model, the network, or even running the code. It is pure
*static analysis* of the source text.

We build that up in five steps:

1. what the parser sees (`ast.dump`)
2. the same thing drawn as a tree
3. the key trick: `Store` vs `Load` = *defines* vs *uses*
4. from two blocks to an edge
5. the real thing — run it on the actual v3 notebook, with a `show_ast(cell)` helper

> Everything here is standard-library `ast`. No third-party packages, no Ollama, no network.
"""))

cells.append(md("## 0. A small specimen\n\nWe'll dissect this one short function for most of the notebook."))

cells.append(code(r'''
import ast, textwrap

SNIPPET = textwrap.dedent("""
    def write_code(filename, content):
        res = lint_python(content)
        path = AGENT_CODE_DIR / filename
        return f"WROTE {filename}"
""").strip()

print(SNIPPET)
'''))

cells.append(md(r"""
## 1. What the parser sees — `ast.dump`

`ast.parse(source)` turns text into a tree of **nodes**, each one a grammar construct:
`Module`, `FunctionDef`, `Assign`, `Call`, `Name`, `Constant`, and so on. `ast.dump` prints
that tree as text. It's verbose, but every field matters — notice especially the `ctx=Store()`
vs `ctx=Load()` markers on each `Name`, which we'll exploit in step 3.
"""))

cells.append(code(r'''
tree = ast.parse(SNIPPET)
print(ast.dump(tree, indent=4))
'''))

cells.append(md(r"""
## 2. The same tree, drawn as a tree

`ast.dump` is the *textual* form. The tree structure is easier to see if we walk it with
`ast.iter_child_nodes` (each node's direct children) and print with indentation. This is exactly
the structure the graph generator traverses.
"""))

cells.append(code(r'''
def draw(node, prefix="", last=True):
    """Pretty-print an AST as an indented tree, annotating the fields that matter."""
    label = type(node).__name__
    extra = ""
    if isinstance(node, ast.Name):          extra = f"  id={node.id!r} ({type(node.ctx).__name__})"
    elif isinstance(node, ast.FunctionDef):  extra = f"  name={node.name!r}"
    elif isinstance(node, ast.ClassDef):     extra = f"  name={node.name!r}"
    elif isinstance(node, ast.arg):          extra = f"  arg={node.arg!r}"
    elif isinstance(node, ast.Constant):     extra = f"  value={node.value!r}"
    elif isinstance(node, ast.Attribute):    extra = f"  attr={node.attr!r}"
    print(prefix + ("└─ " if last else "├─ ") + label + extra)
    kids = list(ast.iter_child_nodes(node))
    child_prefix = prefix + ("   " if last else "│  ")
    for i, ch in enumerate(kids):
        draw(ch, child_prefix, i == len(kids) - 1)

draw(tree)
'''))

cells.append(md(r"""
Read it top-down:

- `Module` is always the root; its `body` is the list of top-level statements.
- `FunctionDef name='write_code'` — the `def`. Its name is a thing this block **creates**.
- inside, two `Assign` nodes create `res` and `path` — note their `Name` has `ctx=Store`.
- the calls/expressions reference `lint_python`, `AGENT_CODE_DIR`, `content`, `filename` —
  those `Name`s have `ctx=Load`.

That `Store` / `Load` distinction is the whole game.
"""))

cells.append(md(r"""
## 3. `Store` vs `Load` = *defines* vs *uses*

Python's compiler tags every `Name` with a **context**:

- **`Store`** — the name is being *bound* (left of `=`, a `for` target, …). Plus `FunctionDef`
  and `ClassDef` *names* are definitions too. → these are what the block **defines**.
- **`Load`** — the name is being *read*. → these are what the block **uses**.

`ast.walk(tree)` yields every node in the tree (the flattened version of `draw`), so we can
collect both sets in a single pass. This is the heart of the generator's `parse_cell`.
"""))

cells.append(code(r'''
def parse_cell(src):
    """Return (defines, uses) for one block of source — module-level definitions, all loads."""
    defines, uses = set(), set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return defines, uses
    # module-level definitions: def / class / top-level assignment targets
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defines.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    defines.add(t.id)
    # uses: every Name read anywhere in the tree
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            uses.add(node.id)
    return defines, uses

d, u = parse_cell(SNIPPET)
print("defines:", sorted(d))
print("uses   :", sorted(u))
'''))

cells.append(md(r"""
Notice `filename` and `content` show up in **uses** even though they're the function's
parameters. That's fine: they aren't *module-level* definitions, so no other cell can possibly
define them — when we resolve uses against the define-map in the next step, they simply match
nothing and produce no edge. The graph stays clean without us special-casing locals.

(`parse_cell` deliberately looks only at **module-level** `def`/`class`/assignment for *defines*
— those are the names a cell exports to the rest of the notebook. It scans **all** `Load` names
for *uses*, because a dependency can hide anywhere inside a function body.)
"""))

cells.append(md(r"""
## 4. From two blocks to an edge

Now the actual graph logic, in miniature. Given several blocks:

1. build a map `symbol → the block that defines it`;
2. for each block, every *use* that resolves to a **different** block's *define* becomes an edge.

Here are two toy cells where the dependency is obvious.
"""))

cells.append(code(r'''
CELL_A = textwrap.dedent("""
    AGENT_CODE_DIR = '/tmp/agent_code'
    def lint_python(code):
        return {'passed': True}
""").strip()

CELL_B = SNIPPET   # uses lint_python AND AGENT_CODE_DIR, both defined in CELL_A

blocks = {0: CELL_A, 1: CELL_B}

# 1) symbol -> defining block (first definer wins)
defmap = {}
parsed = {}
for idx, src in blocks.items():
    defs, uses = parse_cell(src)
    parsed[idx] = (defs, uses)
    for name in defs:
        defmap.setdefault(name, idx)

# 2) edges: a use that resolves to another block
edges = []
for idx, (defs, uses) in parsed.items():
    by_src = {}
    for name in uses:
        src_idx = defmap.get(name)
        if src_idx is not None and src_idx != idx:
            by_src.setdefault(src_idx, []).append(name)
    for src_idx, syms in by_src.items():
        edges.append((src_idx, idx, sorted(set(syms))))

print("symbol -> defining block:", defmap)
print("\nedges (from -> to  via symbols):")
for a, b, syms in edges:
    print(f"  #{a} -> #{b}   {syms}")
'''))

cells.append(md(r"""
That single edge `#0 → #1 via ['AGENT_CODE_DIR', 'lint_python']` is exactly the kind of arrow
you see in the HTML graph, label and all. Scale this loop from 2 blocks to the notebook's 39
code cells and you have the whole picture — no heuristics, no model, just `Store`/`Load`
bookkeeping.
"""))

cells.append(md(r"""
## 5. The real thing — run it on the v3 notebook

Now we point the same `parse_cell` at the actual `claude_code_from_scratch_v3.ipynb`: assign
each code cell to its `## Phase` heading, compute the define-map and the edges, and print the
graph's shape. If the notebook isn't found next to this one, this section just skips.
"""))

cells.append(code(r'''
import json, re
from pathlib import Path

NB = Path.cwd() / "claude_code_from_scratch_v3.ipynb"
have_nb = NB.exists()
print("notebook found:" , have_nb, "->", NB if have_nb else "(skipping section 5)")

cells_v3 = []   # list of (idx, phase, src)
if have_nb:
    nb = json.loads(NB.read_text())
    phase = "Phase 0"
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            for line in src.splitlines():
                if line.startswith("## "):
                    phase = line[3:].strip(); break
        elif src.strip():
            cells_v3.append((i, phase, src))
    print(f"{len(cells_v3)} code cells")
'''))

cells.append(code(r'''
if have_nb:
    # define-map across the real notebook
    defmap_v3, parsed_v3 = {}, {}
    for idx, ph, src in cells_v3:
        defs, uses = parse_cell(src)
        parsed_v3[idx] = (defs, uses)
        for name in defs:
            defmap_v3.setdefault(name, idx)

    edges_v3 = []
    for idx, ph, src in cells_v3:
        defs, uses = parsed_v3[idx]
        deps = sorted({defmap_v3[n] for n in uses if n in defmap_v3 and defmap_v3[n] != idx})
        for src_idx in deps:
            edges_v3.append((src_idx, idx))

    print(f"nodes (code cells): {len(cells_v3)}")
    print(f"edges (dependencies): {len(edges_v3)}")
    # the most depended-upon cells = the notebook's hubs
    from collections import Counter
    indeg_targets = Counter(a for a, _ in edges_v3)   # how often each cell is a *source* (used by others)
    print("\nmost-used cells (a source of many edges):")
    for idx, n in indeg_targets.most_common(5):
        title = next(p for i, p, s in cells_v3 if i == idx)
        print(f"  cell #{idx:2d} used by {n:2d} others   [{title[:34]}]")
'''))

cells.append(md(r"""
The hubs that surface — typically the `llm()` factory cell and the tracer cell — are exactly the
ones the HTML graph shows everything pointing back at. The census matches the numbers baked into
the graph (39 nodes, ~128 edges).

### A `show_ast(cell)` helper

Finally, the on-demand inspector. Give it a cell index from the v3 notebook and it prints that
cell's AST tree plus its computed *defines / depends-on / used-by* — the same three facts the
HTML side panel shows when you click a node.
"""))

cells.append(code(r'''
def show_ast(cell_idx, tree_depth=True):
    """Inspect one v3 code cell: its AST tree + defines / depends-on / used-by."""
    if not have_nb:
        print("v3 notebook not available."); return
    match = [(i, p, s) for i, p, s in cells_v3 if i == cell_idx]
    if not match:
        print(f"cell #{cell_idx} is not a code cell. Try one of:",
              [i for i, _, _ in cells_v3][:20], "..."); return
    idx, phase, src = match[0]
    defs, uses = parsed_v3[idx]
    depends_on = {}
    for n in uses:
        s = defmap_v3.get(n)
        if s is not None and s != idx:
            depends_on.setdefault(s, []).append(n)
    used_by = {}
    for j, (jd, ju) in parsed_v3.items():
        if j == idx: continue
        shared = [n for n in ju if n in defs]
        if shared:
            used_by[j] = sorted(set(shared))

    print(f"=== cell #{idx}  [{phase}] ===\n")
    print("SOURCE:\n" + src + "\n")
    if tree_depth:
        print("AST TREE:")
        draw(ast.parse(src))
        print()
    print("defines    :", sorted(defs))
    print("depends on :", {f"#{k}": v for k, v in sorted(depends_on.items())})
    print("used by    :", {f"#{k}": v for k, v in sorted(used_by.items())})

# Example: cell 16 defines write_code (and the tool registry); it depends on the config cell.
show_ast(16)
'''))

cells.append(md(r"""
Try `show_ast(5)` (the `llm()` factory — a big hub), `show_ast(18)` (the tool-loop graph), or
`show_ast(37)` (the five-subagent team — depends on the most other cells). Pass
`tree_depth=False` to skip the full tree and just see the defines/depends/used-by triple.
"""))

cells.append(md(r"""
## Recap — how this maps to the artifacts

| concept here | in the HTML graph | in the prose `.md` |
|---|---|---|
| a parsed code cell | a **node** (shows the cell's code) | a "cell N" reference |
| `parse_cell` *defines* | the node's **Defines** list | the functions documented in that section |
| `parse_cell` *uses* → resolved | an **edge** `A → B` (labelled with symbols) | "depends on / used by" |
| `show_ast(cell)` | clicking a node → the **side panel** | the per-cell write-up |

The generator that does this for real and emits the HTML is **`_build_v3_graph.py`** (run
`python3 _build_v3_graph.py` to rebuild `claude_code_from_scratch_v3_GRAPH.html`). The only
substantive differences from this notebook are cosmetic: it also carries a human-written title
and one-line description per cell, and renders everything as draggable boxes with SVG edges.

**The takeaway:** a dependency graph over code is not magic or model-driven — it's `ast.parse`
plus the `Store`/`Load` distinction, counted across blocks.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")

if __name__ == "__main__":
    pass
