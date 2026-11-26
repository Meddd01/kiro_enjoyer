"""Dev-only live tracking server (not part of the solution code).

python dashboard.py  ->  open http://localhost:8765/   (page polls live.json every 5 s, no reloads)
- every 10 s: validates every candidate (solutions/**/<name>.json, codex_teamwarmup/) with evaluator.py (cached by mtime),
  keeps the best valid per instance, logs the team total over time, reads LNS worker logs;
- every 120 s: reads the PUBLIC leaderboard https://kiro.enpc.org/classement/ (anonymous GET, no login).
"""
import datetime
import glob
import http.server
import json
import os
import re
import threading
import time
import urllib.request

from evaluator import evaluate, worst_scenario

NAMES = ["alliances", "reliability", "strikes"]
TEAM = "double span"
DIR = "runs/claude/live"
HIST = "runs/claude/history.json"
_cache, board = {}, {"rows": [], "fetched": None, "error": None}


def score(inst, path):
    key = (path, os.path.getmtime(path))
    if key not in _cache:
        try:
            r = evaluate(inst, json.load(open(path)))
            _cache[key] = (r["score"] if r["valid"] else None, worst_scenario(r))
        except (OSError, ValueError, KeyError):
            _cache[key] = (None, None)
    return _cache[key]


def collect(insts):
    best, alln = {}, {}
    for n in NAMES:
        paths = {p for p in glob.glob(f"solutions/**/{n}.json", recursive=True) if "history" not in p} | set(glob.glob(f"codex_teamwarmup/{n}.json"))
        cands = []
        for p in sorted(paths):
            s, w = score(insts[n], p)
            if s is not None:
                cands.append({"score": s, "file": p.replace("\\", "/"), "worst": w})
        cands.sort(key=lambda c: c["score"])
        alln[n] = cands
        best[n] = cands[0] if cands else None
    return best, alln


def workers():
    rows = []
    for p in sorted(glob.glob("runs/claude/warm*.txt") + glob.glob("runs/claude/lns*.txt") + glob.glob("runs/claude/smoke*.txt")):
        age = time.time() - os.path.getmtime(p)
        if age > 1800 and "warm" not in p:
            continue
        txt = open(p, encoding="utf-8", errors="replace").read()
        wins = re.findall(r"^(\d\d:\d\d:\d\d) \S+ (\S+)\s.*?(\d+) -> (\d+)", txt, re.M)
        state = "error" if "Traceback" in txt else ("finished" if re.search(r"final \d+", txt)
                                                     else ("running" if age < 120 else "idle"))
        rows.append({"name": os.path.basename(p)[:-4], "state": state, "wins": len(wins),
                     "gain": int(wins[0][2]) - int(wins[-1][3]) if wins else 0,
                     "last": f"{wins[-1][0]} {wins[-1][1]} -> {int(wins[-1][3]):,}".replace(",", " ") if wins else ""})
    return rows


def fetch_board():
    while True:
        try:
            req = urllib.request.Request("https://kiro.enpc.org/classement/",
                                         headers={"User-Agent": "Mozilla/5.0 (team-dashboard)"})
            html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
            rows = []
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
                tds = [re.sub(r"<[^>]+>", " ", td).strip() for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
                if len(tds) < 7:
                    continue
                team = re.sub(r"\s+(Sur place|À distance)\s*$", "", tds[1]).strip()
                num = lambda x: int(re.sub(r"\D", "", x)) if re.search(r"\d", x) else None  # noqa: E731
                rows.append({"rank": num(tds[0]), "team": team, "total": num(tds[2]),
                             "inst": [num(tds[3]), num(tds[4]), num(tds[5])], "sent": num(tds[6])})
            board.update(rows=rows, fetched=time.time(), error=None)
        except Exception as exc:  # network hiccup: keep the last good board
            board["error"] = repr(exc)
        time.sleep(120)


def loop():
    while True:
        try:
            _loop()
        except Exception as exc:  # never let the data thread die
            print("loop error:", repr(exc), flush=True)
            time.sleep(5)


def _loop():
    insts = {n: json.load(open(f"instances/{n}.json")) for n in NAMES}
    hist = json.load(open(HIST)) if os.path.exists(HIST) else []
    while True:
        best, alln = collect(insts)
        total = sum(b["score"] for b in best.values()) if all(best.values()) else None
        if total and (not hist or hist[-1]["total"] != total):
            hist.append({"t": time.time(), "total": total, **{n: best[n]["score"] for n in NAMES}})
            json.dump(hist, open(HIST, "w"))
        ranked = [r for r in board["rows"] if r["total"]]
        best_inst = [min((r["inst"][i] for r in ranked if r["inst"][i]), default=None) for i in range(3)]
        live = {"updated": time.time(), "best": best, "candidates": {n: alln[n][:6] for n in NAMES}, "total": total,
                "history": hist, "workers": workers(), "board": board, "team": TEAM,
                "board_best_inst": best_inst,
                "rank_if_submitted": (1 + sum(1 for r in ranked if r["team"] != TEAM and r["total"] < total)) if total and ranked else None}
        tmp = os.path.join(DIR, "live.json.tmp")
        json.dump(live, open(tmp, "w"))
        for _ in range(20):  # Windows: the HTTP thread may hold live.json open for a moment
            try:
                os.replace(tmp, os.path.join(DIR, "live.json"))
                break
            except PermissionError:
                time.sleep(0.25)
        time.sleep(10)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>KIRO live</title><style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#2a78d6;--grid:#e6e5e1;--card:#f4f3f0;--good:#0ca30c;--bad:#d03b3b}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--line:#3987e5;--grid:#383835;--card:#232322;--bad:#e66767}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1040px;padding:0 16px}
h1{font-size:18px;margin:0 0 2px}h2{font-size:15px;margin:22px 0 4px}.mut{color:var(--mut)}
.hero{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0}.card{background:var(--card);border-radius:10px;padding:12px 16px}
.big{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid)}th{color:var(--mut);font-weight:500}
.num{text-align:right;font-variant-numeric:tabular-nums}.path{color:var(--mut);font-size:12px}tr.us td{font-weight:650}
.grid{stroke:var(--grid)}.axis{fill:var(--mut);font-size:11px}.ln{fill:none;stroke:var(--line);stroke-width:2}
.ref{stroke:var(--mut);stroke-dasharray:4 4}.dot{fill:var(--line);stroke:var(--bg);stroke-width:2}.cross{stroke:var(--mut)}
.tip{position:absolute;background:var(--card);border:1px solid var(--grid);border-radius:6px;padding:4px 8px;font-size:12px;pointer-events:none}
.st{font-size:12px;padding:1px 8px;border-radius:9px;border:1px solid var(--grid)}.running::before{content:'● ';color:var(--good)}
.error::before{content:'✖ ';color:var(--bad)}.finished::before{content:'✔ '}.idle::before{content:'○ '}
.pulse{display:inline-block;width:8px;height:8px;border-radius:4px;background:var(--good);margin-right:6px}
</style></head><body>
<h1>KIRO 2026 — <span id="team"></span> · live</h1>
<div class="mut"><span class="pulse" id="pulse"></span><span id="upd">connecting…</span></div>
<div class="hero">
 <div class="card"><div class="mut">Team best (sum of best valid per instance)</div><div class="big" id="tot">—</div><div class="mut" id="totsub"></div></div>
 <div class="card"><div class="mut">Rank if submitted now</div><div class="big" id="rank">—</div><div class="mut" id="ranksub"></div></div>
 <div class="card"><div class="mut">Gap to #1</div><div class="big" id="gap">—</div><div class="mut" id="gapsub"></div></div>
</div>
<h2>Best valid solution per instance <span class="mut" style="font-weight:400">(submit these files)</span></h2>
<table><thead><tr><th>instance</th><th class=num>score</th><th class=num>vs our board row</th><th class=num>vs best on board</th><th>worst</th><th>file</th></tr></thead><tbody id="inst"></tbody></table>
<h2>Team best total over time</h2><div id="chart"></div><div id="tip" class="tip" hidden></div>
<h2>LNS workers</h2><table><thead><tr><th>log</th><th>state</th><th class=num>wins</th><th class=num>gain</th><th>last improvement</th></tr></thead><tbody id="wk"></tbody></table>
<h2>Leaderboard (public, refreshed every 2 min) <span class="mut" id="bfetch" style="font-weight:400"></span></h2>
<table><thead><tr><th class=num>#</th><th>team</th><th class=num>total</th><th class=num>alliances</th><th class=num>reliability</th><th class=num>strikes</th><th class=num>sent</th></tr></thead><tbody id="lb"></tbody></table>
<script>
const N=['alliances','reliability','strikes'],f=v=>v==null?'—':v.toLocaleString('fr-FR').replace(/\u202f/g,' '),M=v=>(v>=0?'+':'')+(v/1e6).toFixed(2)+'M';
const tm=t=>new Date(t*1000).toLocaleTimeString('fr-FR');let H=[];
function chart(h,ref){const W=760,Hh=220,L=92,B=28,T=12,el=document.getElementById('chart');if(h.length<2){el.innerHTML='<p class=mut>Chart appears after the second improvement.</p>';return}
const ts=h.map(d=>d.t),vs=h.map(d=>d.total);let lo=Math.min(...vs,ref||Infinity),hi=Math.max(...vs);const p=(hi-lo)*.08||1;lo-=p;hi+=p;
const t0=ts[0],t1=Math.max(Date.now()/1000,ts[0]+1),X=t=>L+(W-L-10)*(t-t0)/(t1-t0),Y=v=>T+(Hh-T-B)*(hi-v)/(hi-lo);
let pts=[];vs.forEach((v,i)=>{if(i)pts.push(X(ts[i])+','+Y(vs[i-1]));pts.push(X(ts[i])+','+Y(v))});pts.push(X(t1)+','+Y(vs[vs.length-1]));
let g='';for(let k=0;k<5;k++){const v=lo+(hi-lo)*k/4;g+=`<line x1=${L} x2=${W-10} y1=${Y(v)} y2=${Y(v)} class=grid /><text x=${L-8} y=${Y(v)+4} class=axis text-anchor=end>${(v/1e6).toFixed(1)}M</text>`}
for(let k=0;k<4;k++){const t=t0+(t1-t0)*k/3;g+=`<text x=${X(t)} y=${Hh-8} class=axis text-anchor=middle>${tm(t).slice(0,5)}</text>`}
if(ref)g+=`<line x1=${L} x2=${W-10} y1=${Y(ref)} y2=${Y(ref)} class=ref /><text x=${W-12} y=${Y(ref)-6} class=axis text-anchor=end>#1 ${(ref/1e6).toFixed(2)}M</text>`;
el.innerHTML=`<svg id=ch viewBox="0 0 ${W} ${Hh}" role=img aria-label="Team best total over time">${g}<polyline points="${pts.join(' ')}" class=ln /><circle cx=${X(ts[ts.length-1])} cy=${Y(vs[vs.length-1])} r=4 class=dot /><line id=cx y1=${T} y2=${Hh-B} class=cross visibility=hidden /><circle id=cd r=5 class=dot visibility=hidden /><rect id=hit x=${L} y=${T} width=${W-L-10} height=${Hh-T-B} fill=transparent /></svg>`;
const s=document.getElementById('ch'),tip=document.getElementById('tip'),hit=document.getElementById('hit'),cx=document.getElementById('cx'),cd=document.getElementById('cd');
hit.onmousemove=e=>{const q=s.createSVGPoint();q.x=e.clientX;q.y=e.clientY;const o=q.matrixTransform(s.getScreenCTM().inverse());let b=0;ts.forEach((t,i)=>{if(Math.abs(X(t)-o.x)<Math.abs(X(ts[b])-o.x))b=i});
cx.setAttribute('x1',X(ts[b]));cx.setAttribute('x2',X(ts[b]));cd.setAttribute('cx',X(ts[b]));cd.setAttribute('cy',Y(vs[b]));cx.setAttribute('visibility','visible');cd.setAttribute('visibility','visible');
const r=s.getBoundingClientRect();tip.hidden=false;tip.innerHTML='<b>'+f(vs[b])+'</b><br>'+N.map(n=>n+' '+f(h[b][n])).join('<br>')+'<br>'+tm(ts[b]);tip.style.left=(r.left+scrollX+X(ts[b])/W*r.width+12)+'px';tip.style.top=(r.top+scrollY+Y(vs[b])/Hh*r.height-10)+'px'};
hit.onmouseleave=()=>{tip.hidden=true;cx.setAttribute('visibility','hidden');cd.setAttribute('visibility','hidden')}}
async function tick(){try{const d=await (await fetch('live.json?'+Date.now())).json();const age=Date.now()/1000-d.updated;
document.getElementById('team').textContent=d.team;document.getElementById('upd').textContent='data '+tm(d.updated)+(age>40?' — STALE ('+Math.round(age)+' s)':'');
document.getElementById('pulse').style.background=age>40?'var(--bad)':'var(--good)';
const rows=d.board.rows.filter(r=>r.total),us=rows.find(r=>r.team===d.team),first=rows[0];
document.getElementById('tot').textContent=f(d.total);document.getElementById('totsub').textContent=us?M(d.total-us.total)+' vs our board row ('+f(us.total)+')':'';
document.getElementById('rank').textContent=d.rank_if_submitted?'#'+d.rank_if_submitted:'—';document.getElementById('ranksub').textContent=us?'board now: #'+us.rank+' of '+rows.length:'';
if(first){document.getElementById('gap').textContent=M(d.total-first.total);document.getElementById('gapsub').textContent=first.team+' '+f(first.total)}
document.getElementById('inst').innerHTML=N.map((n,i)=>{const b=d.best[n];if(!b)return`<tr><td>${n}</td><td colspan=5>no valid file</td></tr>`;
return`<tr><td>${n}</td><td class=num><b>${f(b.score)}</b></td><td class=num>${us?M(b.score-us.inst[i]):'—'}</td><td class=num>${d.board_best_inst[i]?M(b.score-d.board_best_inst[i]):'—'}</td><td>s${b.worst}</td><td class=path>${b.file}</td></tr>`}).join('');
document.getElementById('wk').innerHTML=d.workers.map(w=>`<tr><td>${w.name}</td><td><span class="st ${w.state}">${w.state}</span></td><td class=num>${w.wins}</td><td class=num>${f(w.gain)}</td><td>${w.last}</td></tr>`).join('');
document.getElementById('bfetch').textContent=d.board.fetched?'fetched '+tm(d.board.fetched)+(d.board.error?' (last error: '+d.board.error+')':''):'not fetched yet';
const show=rows.filter((r,i)=>i<12||r.team===d.team);document.getElementById('lb').innerHTML=show.map(r=>`<tr class="${r.team===d.team?'us':''}"><td class=num>${r.rank}</td><td>${r.team}</td><td class=num>${f(r.total)}</td>${r.inst.map(v=>`<td class=num>${f(v)}</td>`).join('')}<td class=num>${r.sent??''}</td></tr>`).join('');
if(JSON.stringify(d.history)!==JSON.stringify(H)){H=d.history;chart(H,first&&first.total)}}catch(e){document.getElementById('upd').textContent='server not reachable — is dashboard.py running?';document.getElementById('pulse').style.background='var(--bad)'}}
tick();setInterval(tick,5000);
</script></body></html>"""


def main():
    os.makedirs(DIR, exist_ok=True)
    open(os.path.join(DIR, "index.html"), "w", encoding="utf-8").write(PAGE)
    threading.Thread(target=fetch_board, daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=DIR, **k)  # noqa: E731
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
    print("dashboard on http://localhost:8765/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", 8765), handler).serve_forever()


if __name__ == "__main__":
    main()
