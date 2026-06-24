#!/usr/bin/env python3
"""
Generate a self-contained interactive HTML graph of claude_code_from_scratch_v3.ipynb.

Each node = one notebook code cell (the "block"), showing its real code.
Edges are computed automatically: cell A -> cell B when B *uses* a top-level
symbol that A *defines*. Edge labels carry the symbol names.

Output: claude_code_from_scratch_v3_GRAPH.html  (open in any browser, no network).
"""
import json, ast, html, re
from pathlib import Path

NB = Path(__file__).with_name("claude_code_from_scratch_v3.ipynb")
OUT = Path(__file__).with_name("claude_code_from_scratch_v3_GRAPH.html")

# ---- human-written one-line summary + short title per code cell -------------
META = {
    2:  ("Imports", "The whole dependency surface: stdlib + the new LangChain/LangGraph stack."),
    3:  ("Logging", "A coloured `agent3` logger with per-subsystem children (llm/tool/graph/subagent)."),
    4:  ("Config", "Every knob in one place: endpoint, model-per-role map, sandbox paths, limits, blocklist."),
    5:  ("Model factory: llm()", "The single chokepoint that builds a cached ChatOllama by role; plus a tags-only healthcheck."),
    7:  ("RichTracer (callbacks)", "A BaseCallbackHandler LangChain narrates into on every model/tool start/stop; CB + run_config()."),
    8:  ("Graph views", "show_graph() draws any compiled graph; stream_run() pretty-prints each node update as it streams."),
    10: ("Prompt + think parsers", "STRONG_SYSTEM_PROMPT and the tolerant strip_think/split_think helpers."),
    11: ("think_then_answer()", "One free-text call with qwen3's thinking channel separated from the answer."),
    12: ("Structured routers", "estimate_difficulty / classify_problem via with_structured_output (Pydantic, not JSON parsing)."),
    13: ("Test-time compute", "self_consistency, verifier_score, asymmetric_solve, adaptive_think — parallel sampling via .batch()."),
    15: ("File/shell tools", "Sandboxed @tool funcs: read/write/revert_file, grep, glob_files, bash + _safe_path guard."),
    16: ("Coding tools", "lint_python, _run_tests, write_code (lint-gated), run_python, run_tests; the TOOLS_BASE registry."),
    18: ("build_agent_graph()", "The v3 replacement for master_loop: agent <-> ToolNode routed by tools_condition. Builds coding_agent."),
    19: ("Subagents (agent-as-tool)", "spawn_subagent runs a focused sub-graph and returns only its final message; make_subagent_tool wraps it."),
    20: ("[demo] draw the loop", "Render the lead coding agent as a diagram."),
    21: ("[demo] run the loop", "One real streamed run of coding_agent on a tiny file task."),
    23: ("architect -> editor", "Reasoning model plans (structured), fast model transcribes. A linear two-node graph."),
    24: ("self-refine loop", "generate -> critique -> refine, looping until the iteration budget is spent."),
    25: ("code-with-tests loop", "generate -> verify; verify ends on pass or loops back with the failure as feedback."),
    26: ("adversarial_probe()", "One structured-output call returning a typed list of attacks (red-teaming)."),
    27: ("[demo] draw self-refine", "Visualise the self-refine loop."),
    28: ("[demo] code-with-tests", "Generate inc(n)=n+1 and verify against a real test."),
    30: ("make_plan()", "A validated Pydantic Plan (goal + dependency-ordered steps) via with_structured_output."),
    31: ("TaskDAG + memory", "Durable sqlite dependency DAG and a bi-temporal fact memory (invalidate, never delete)."),
    32: ("Spec layer", "definition-of-done compiled into a runnable pytest suite; spec_verify runs it."),
    34: ("Context hook", "make_context_hook bounds the model's view (trim->reinject); build_managed_agent wires it into create_react_agent."),
    35: ("[demo] draw managed agent", "Visualise the bounded-window react agent."),
    37: ("The team graph", "Five subagent nodes + the tester->implementer self-correcting loop, as one StateGraph. run_team()."),
    38: ("[demo] draw the team", "Visualise the whole five-subagent team."),
    40: ("[demo] healthcheck", "Is the backend up and are the models present?"),
    41: ("Contract: fizzbuzz", "A definition-of-done for the Phase-8 task."),
    42: ("[demo] run the team", "Run the whole team end-to-end on fizzbuzz, streamed."),
    44: ("[demo] structural census", "Pure introspection of every v3 graph — no model calls."),
    46: ("Offline self-tests", "Everything checkable without the backend: tools, parsers, DAG, memory, spec, schemas, topology."),
    47: ("Self-test roll-up", "Count passes/fails and assert none failed."),
    49: ("Contract: BoundedCounter", "A harder definition-of-done for the Phase-11 build."),
    50: ("[demo] run harder build", "Drive the same team graph on BoundedCounter."),
    51: ("[demo] independent verify", "Re-compile the contract and run it ourselves; read back the artefact + report."),
    52: ("[demo] replay history", "Dump the run's per-node notes the checkpointer kept."),
}

# Short phase keys for colouring / columns
def phase_key(full: str) -> str:
    m = re.match(r"(Phase [\d.]+)", full)
    return m.group(1) if m else full

def parse_cell(src):
    """Return (defines:set, used:set) at module level."""
    defines, used = set(), set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return defines, used
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defines.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    defines.add(t.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return defines, used

def main():
    nb = json.loads(NB.read_text())
    phase = "Phase 0"
    cells = []  # (idx, phase_full, src)
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            for line in src.splitlines():
                if line.startswith("## "):
                    phase = line[3:].strip()
                    break
        elif src.strip():
            cells.append((i, phase, src))

    # defining-cell per symbol (first definer wins)
    defmap = {}
    parsed = {}
    for idx, ph, src in cells:
        d, u = parse_cell(src)
        parsed[idx] = (d, u)
        for name in d:
            defmap.setdefault(name, idx)

    nodes, edges = [], []
    phases_order = []
    for idx, ph, src in cells:
        pk = phase_key(ph)
        if pk not in phases_order:
            phases_order.append(pk)
        d, u = parsed[idx]
        title, desc = META.get(idx, (f"cell {idx}", ""))
        is_demo = title.startswith("[demo]") or (not d) or src.lstrip().startswith("# [LIVE]")
        nodes.append({
            "id": idx, "phase": pk, "phaseFull": ph, "title": title,
            "desc": desc, "code": src, "defines": sorted(d), "demo": bool(is_demo),
        })
        # edges: which other cell defines each used symbol
        bydep = {}
        for name in u:
            src_idx = defmap.get(name)
            if src_idx is not None and src_idx != idx:
                bydep.setdefault(src_idx, []).append(name)
        for src_idx, syms in bydep.items():
            edges.append({"from": src_idx, "to": idx, "syms": sorted(set(syms))})

    data = {"nodes": nodes, "edges": edges, "phases": phases_order}
    OUT.write_text(HTML_TEMPLATE.replace("__DATA__", json.dumps(data)))
    print(f"wrote {OUT}  ({len(nodes)} nodes, {len(edges)} edges, {len(phases_order)} phases)")

# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>claude_code_from_scratch_v3 — code-block graph</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#c9d1d9; --muted:#8b949e;
    --edge:#3d4450; --edgehi:#58a6ff; --accent:#58a6ff;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);
    font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;overflow:hidden}
  #top{position:fixed;top:0;left:0;right:0;height:48px;background:var(--panel);
    border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px;
    padding:0 16px;z-index:30}
  #top b{color:#fff;font-size:14px}
  #top .hint{color:var(--muted);font-size:12px}
  #search{background:#0d1117;border:1px solid var(--line);color:var(--txt);
    border-radius:6px;padding:5px 9px;font-size:12px;width:200px}
  #legend{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto;align-items:center}
  .chip{font-size:11px;padding:2px 8px;border-radius:10px;cursor:pointer;
    border:1px solid var(--line);white-space:nowrap;user-select:none}
  .chip.off{opacity:.32}
  #stage{position:absolute;inset:48px 0 0 0;overflow:hidden;cursor:grab}
  #stage.grabbing{cursor:grabbing}
  #world{position:absolute;top:0;left:0;transform-origin:0 0}
  svg#edges{position:absolute;top:0;left:0;overflow:visible;pointer-events:none}
  .node{position:absolute;width:340px;background:var(--panel);border:1px solid var(--line);
    border-radius:9px;box-shadow:0 4px 14px rgba(0,0,0,.4);overflow:hidden;
    transition:box-shadow .12s,opacity .12s}
  .node.dim{opacity:.18}
  .node.sel{box-shadow:0 0 0 2px var(--accent),0 6px 22px rgba(0,0,0,.55)}
  .node .hd{padding:7px 10px;cursor:grab;border-left:4px solid var(--pc,#58a6ff);
    display:flex;align-items:baseline;gap:7px}
  .node .hd .ci{font-size:10px;color:var(--muted)}
  .node .hd .ti{font-size:12.5px;font-weight:600;color:#fff;line-height:1.2}
  .node .ph{font-size:10px;color:var(--muted);padding:0 10px 5px 14px}
  .node.demo{border-style:dashed}
  .node .code{margin:0;padding:8px 10px;font:11px/1.45 "SF Mono",Menlo,Consolas,monospace;
    background:#0a0e14;max-height:168px;overflow:auto;white-space:pre;border-top:1px solid var(--line)}
  .node.expand .code{max-height:none}
  .node .more{font-size:10px;color:var(--muted);text-align:center;padding:3px;cursor:pointer;
    border-top:1px solid var(--line);background:#10151c}
  /* python highlight */
  .k{color:#ff7b72}.s{color:#a5d6ff}.c{color:#8b949e;font-style:italic}
  .d{color:#d2a8ff}.n{color:#79c0ff}.b{color:#ffa657}
  #side{position:fixed;top:48px;right:0;bottom:0;width:430px;background:var(--panel);
    border-left:1px solid var(--line);z-index:25;transform:translateX(100%);
    transition:transform .18s;display:flex;flex-direction:column}
  #side.open{transform:translateX(0)}
  #side .sh{padding:12px 14px;border-bottom:1px solid var(--line)}
  #side .sh h2{margin:0 0 3px;font-size:15px;color:#fff}
  #side .sh .sub{font-size:11px;color:var(--muted)}
  #side .sb{padding:12px 14px;overflow:auto;font-size:12.5px;line-height:1.5}
  #side .sb h3{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);
    margin:14px 0 6px}
  #side pre{background:#0a0e14;border:1px solid var(--line);border-radius:7px;padding:10px;
    overflow:auto;font:11.5px/1.5 "SF Mono",Menlo,Consolas,monospace;white-space:pre}
  #side .lk{display:inline-block;background:#0d1117;border:1px solid var(--line);border-radius:5px;
    padding:2px 7px;margin:2px 4px 2px 0;font-size:11px;cursor:pointer;color:var(--txt)}
  #side .lk:hover{border-color:var(--accent);color:#fff}
  #side .sym{color:#ffa657}
  #side .close{position:absolute;top:9px;right:12px;cursor:pointer;color:var(--muted);font-size:18px}
  .mini{font-size:11px;color:var(--muted)}
</style></head>
<body>
<div id="top">
  <b>v3 code-block graph</b>
  <input id="search" placeholder="filter cells… (e.g. tester, llm)">
  <span class="hint">drag bg = pan · wheel = zoom · click node = detail · drag header = move</span>
  <span id="legend"></span>
</div>
<div id="stage"><div id="world"><svg id="edges"></svg></div></div>
<div id="side">
  <span class="close" onclick="closeSide()">✕</span>
  <div class="sh"><h2 id="s-title"></h2><div class="sub" id="s-sub"></div></div>
  <div class="sb" id="s-body"></div>
</div>
<script>
const DATA = __DATA__;
const PHASE_COLORS = ["#58a6ff","#56d364","#e3b341","#ff7b72","#d2a8ff","#79c0ff",
  "#ffa657","#7ee787","#f778ba","#a5d6ff","#ff9bce","#bc8cff"];
const pcolor = {}; DATA.phases.forEach((p,i)=>pcolor[p]=PHASE_COLORS[i%PHASE_COLORS.length]);

// ---- python syntax highlight (lightweight) --------------------------------
const KW = new Set("def class return if elif else for while in is not and or import from as with try except finally raise lambda yield global nonlocal pass break continue assert del None True False async await".split(" "));
const BUILT = new Set("print len range list dict set tuple str int float bool any all map filter open isinstance getattr setattr hasattr super self enumerate sorted reversed".split(" "));
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function hl(code){
  let out="", i=0;
  const re=/(\x22\x22\x22[\s\S]*?\x22\x22\x22|\x27\x27\x27[\s\S]*?\x27\x27\x27|\x22(?:\\.|[^\x22\\])*\x22|\x27(?:\\.|[^\x27\\])*\x27|#[^\n]*|\b[A-Za-z_]\w*\b|@\w+)/g;
  let m,last=0;
  while((m=re.exec(code))){
    out+=esc(code.slice(last,m.index));
    const t=m[0];
    if(t.startsWith("#")) out+=`<span class="c">${esc(t)}</span>`;
    else if(t[0]==='"'||t[0]==="'") out+=`<span class="s">${esc(t)}</span>`;
    else if(t[0]==="@") out+=`<span class="d">${esc(t)}</span>`;
    else if(KW.has(t)) out+=`<span class="k">${esc(t)}</span>`;
    else if(BUILT.has(t)) out+=`<span class="b">${esc(t)}</span>`;
    else if(/^[A-Z]/.test(t)) out+=`<span class="n">${esc(t)}</span>`;
    else out+=esc(t);
    last=m.index+t.length;
  }
  out+=esc(code.slice(last));
  return out;
}

// ---- layout: columns by phase ---------------------------------------------
const NW=340, GAPX=120, GAPY=40, COLW=NW+GAPX, TOP=30;
const byPhase={}; DATA.phases.forEach(p=>byPhase[p]=[]);
DATA.nodes.forEach(n=>byPhase[n.phase].push(n));
const pos={}, est={};
DATA.phases.forEach((p,ci)=>{
  let y=TOP;
  byPhase[p].forEach(n=>{
    const h = 92 + Math.min(168, (n.code.split("\n").length*16)+18); // header+ph+code area
    pos[n.id]={x: 60+ci*COLW, y};
    est[n.id]=h;
    y += h + GAPY;
  });
});

const world=document.getElementById("world");
const svg=document.getElementById("edges");
const nodeEl={};
DATA.nodes.forEach(n=>{
  const d=document.createElement("div");
  d.className="node"+(n.demo?" demo":"");
  d.style.left=pos[n.id].x+"px"; d.style.top=pos[n.id].y+"px";
  d.style.setProperty("--pc",pcolor[n.phase]);
  const lines=n.code.split("\n").length;
  d.innerHTML=`<div class="hd" data-drag="1"><span class="ci">#${n.id}</span>
      <span class="ti">${esc(n.title)}</span></div>
    <div class="ph">${esc(n.phase)} · ${lines} lines${n.defines.length?` · defines ${esc(n.defines.slice(0,3).join(", "))}${n.defines.length>3?"…":""}`:""}</div>
    <pre class="code">${hl(n.code)}</pre>
    ${lines>10?'<div class="more">▼ expand</div>':''}`;
  world.appendChild(d);
  nodeEl[n.id]=d;
  d.querySelector(".hd").addEventListener("mousedown",e=>startDrag(e,n.id));
  d.addEventListener("click",e=>{ if(!dragged) selectNode(n.id); });
  const more=d.querySelector(".more");
  if(more) more.addEventListener("click",e=>{e.stopPropagation();
    d.classList.toggle("expand"); more.textContent=d.classList.contains("expand")?"▲ collapse":"▼ expand"; drawEdges();});
});

// ---- edges -----------------------------------------------------------------
const edgeEls=[];
DATA.edges.forEach(e=>{
  const path=document.createElementNS("http://www.w3.org/2000/svg","path");
  path.setAttribute("fill","none"); path.setAttribute("stroke","var(--edge)");
  path.setAttribute("stroke-width","1.4"); svg.appendChild(path);
  edgeEls.push({e,path});
});
function nodeBox(id){const d=nodeEl[id];return{x:pos[id].x,y:pos[id].y,w:d.offsetWidth,h:d.offsetHeight};}
function drawEdges(){
  let maxX=0,maxY=0;
  edgeEls.forEach(({e,path})=>{
    const a=nodeBox(e.from), b=nodeBox(e.to);
    const x1=a.x+a.w, y1=a.y+28, x2=b.x, y2=b.y+28;
    const dx=Math.max(40,Math.abs(x2-x1)*0.5);
    path.setAttribute("d",`M${x1},${y1} C${x1+dx},${y1} ${x2-dx},${y2} ${x2},${y2}`);
  });
  DATA.nodes.forEach(n=>{const bx=nodeBox(n.id);maxX=Math.max(maxX,bx.x+bx.w);maxY=Math.max(maxY,bx.y+bx.h);});
  svg.setAttribute("width",maxX+200); svg.setAttribute("height",maxY+200);
}
drawEdges();

// ---- pan / zoom ------------------------------------------------------------
let scale=0.75, tx=20, ty=10;
const stage=document.getElementById("stage");
function applyT(){world.style.transform=`translate(${tx}px,${ty}px) scale(${scale})`;}
applyT();
stage.addEventListener("wheel",e=>{
  e.preventDefault();
  const r=stage.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const ns=Math.min(2.2,Math.max(0.25,scale*(e.deltaY<0?1.12:0.9)));
  tx=mx-(mx-tx)*(ns/scale); ty=my-(my-ty)*(ns/scale); scale=ns; applyT();
},{passive:false});
let panning=false,sx,sy,dragged=false;
stage.addEventListener("mousedown",e=>{
  if(e.target.closest(".node"))return;
  panning=true;sx=e.clientX-tx;sy=e.clientY-ty;stage.classList.add("grabbing");
});
window.addEventListener("mousemove",e=>{
  if(panning){tx=e.clientX-sx;ty=e.clientY-sy;applyT();}
});
window.addEventListener("mouseup",()=>{panning=false;stage.classList.remove("grabbing");});

// ---- node drag -------------------------------------------------------------
let drag=null;
function startDrag(e,id){
  e.stopPropagation();
  drag={id,ox:e.clientX,oy:e.clientY,px:pos[id].x,py:pos[id].y};
  dragged=false;
}
window.addEventListener("mousemove",e=>{
  if(!drag)return;
  const dx=(e.clientX-drag.ox)/scale, dy=(e.clientY-drag.oy)/scale;
  if(Math.abs(dx)+Math.abs(dy)>3) dragged=true;
  pos[drag.id].x=drag.px+dx; pos[drag.id].y=drag.py+dy;
  const d=nodeEl[drag.id]; d.style.left=pos[drag.id].x+"px"; d.style.top=pos[drag.id].y+"px";
  drawEdges();
});
window.addEventListener("mouseup",()=>{drag=null;setTimeout(()=>dragged=false,30);});

// ---- selection / highlight -------------------------------------------------
let selId=null;
function neighbors(id){
  const ins=DATA.edges.filter(e=>e.to===id), outs=DATA.edges.filter(e=>e.from===id);
  const set=new Set([id]); ins.forEach(e=>set.add(e.from)); outs.forEach(e=>set.add(e.to));
  return {ins,outs,set};
}
function selectNode(id){
  selId=id;
  const {ins,outs,set}=neighbors(id);
  DATA.nodes.forEach(n=>{
    nodeEl[n.id].classList.toggle("dim",!set.has(n.id));
    nodeEl[n.id].classList.toggle("sel",n.id===id);
  });
  edgeEls.forEach(({e,path})=>{
    const hot=e.from===id||e.to===id;
    path.setAttribute("stroke",hot?"var(--edgehi)":"var(--edge)");
    path.setAttribute("stroke-width",hot?"2.2":"1.4");
    path.setAttribute("opacity",hot?"1":"0.25");
  });
  openSide(id,ins,outs);
}
function clearSel(){
  selId=null;
  DATA.nodes.forEach(n=>nodeEl[n.id].classList.remove("dim","sel"));
  edgeEls.forEach(({path})=>{path.setAttribute("stroke","var(--edge)");
    path.setAttribute("stroke-width","1.4");path.setAttribute("opacity","1");});
}
const node=id=>DATA.nodes.find(n=>n.id===id);
function openSide(id,ins,outs){
  const n=node(id);
  document.getElementById("s-title").textContent=`#${id} · ${n.title}`;
  document.getElementById("s-sub").textContent=n.phaseFull;
  const lk=(cid,syms)=>`<span class="lk" onclick="selectNode(${cid})">#${cid} ${esc(node(cid).title)}${syms?` <span class="sym">${esc(syms.join(", "))}</span>`:""}</span>`;
  let h="";
  if(n.desc) h+=`<p>${esc(n.desc)}</p>`;
  if(n.defines.length) h+=`<h3>Defines</h3><div>${n.defines.map(d=>`<span class="lk">${esc(d)}</span>`).join("")}</div>`;
  h+=`<h3>Depends on (uses symbols from)</h3>`;
  h+= ins.length? `<div>${ins.map(e=>lk(e.from,e.syms)).join("")}</div>` : `<div class="mini">— nothing (a root block)</div>`;
  h+=`<h3>Used by</h3>`;
  h+= outs.length? `<div>${outs.map(e=>lk(e.to,e.syms)).join("")}</div>` : `<div class="mini">— nothing (a leaf / demo)</div>`;
  h+=`<h3>Code</h3><pre>${hl(n.code)}</pre>`;
  document.getElementById("s-body").innerHTML=h;
  document.getElementById("side").classList.add("open");
}
function closeSide(){document.getElementById("side").classList.remove("open");clearSel();}
stage.addEventListener("click",e=>{if(!e.target.closest(".node")&&!dragged)closeSide();});

// ---- legend (phase toggles) ------------------------------------------------
const legend=document.getElementById("legend");
DATA.phases.forEach(p=>{
  const c=document.createElement("span");
  c.className="chip"; c.textContent=p;
  c.style.borderColor=pcolor[p]; c.style.color=pcolor[p];
  c.onclick=()=>{
    c.classList.toggle("off");
    const on=!c.classList.contains("off");
    byPhase[p].forEach(n=>nodeEl[n.id].style.display=on?"":"none");
    drawEdges();
  };
  legend.appendChild(c);
});

// ---- search ----------------------------------------------------------------
document.getElementById("search").addEventListener("input",e=>{
  const q=e.target.value.toLowerCase().trim();
  DATA.nodes.forEach(n=>{
    const hit=!q||n.title.toLowerCase().includes(q)||n.code.toLowerCase().includes(q)||
              n.defines.some(d=>d.toLowerCase().includes(q));
    nodeEl[n.id].style.opacity=hit?"1":"0.12";
    nodeEl[n.id].style.outline=hit&&q?`2px solid ${pcolor[n.phase]}`:"none";
  });
});
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
