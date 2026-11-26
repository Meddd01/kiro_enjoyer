"""Dev-only live tracking page (not part of the solution code). Regenerates runs/claude/dashboard.html every 30 s.

python dashboard.py            -> open runs/claude/dashboard.html in a browser (auto-refreshes)
Scans every candidate JSON (solutions/**/<name>.json, codex_teamwarmup/), validates each with evaluator.py,
keeps the best valid one per instance, logs the team total over time, and shows LNS worker status.
"""
import datetime
import glob
import html
import json
import os
import re
import time

from evaluator import evaluate, worst_scenario

NAMES = ["alliances", "reliability", "strikes"]
BOARD = {"submitted": {"alliances": 277871405, "reliability": 288094756, "strikes": 268516647},
         "first": {"team": "Les Bleus", "total": 815910640},
         "best": {"alliances": 272188469, "reliability": 280191746, "strikes": 262319392}}
OUT = "runs/claude/dashboard.html"
HIST = "runs/claude/history.json"
_cache = {}


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
    best = {}
    for n in NAMES:
        paths = set(glob.glob(f"solutions/**/{n}.json", recursive=True)) | set(glob.glob(f"codex_teamwarmup/{n}.json"))
        cands = []
        for p in sorted(paths):
            s, w = score(insts[n], p)
            if s is not None:
                cands.append((s, p.replace("\\", "/"), w))
        best[n] = min(cands) if cands else None
    return best


def workers():
    rows = []
    for p in sorted(glob.glob("runs/claude/warm_*.txt") + glob.glob("runs/claude/lns*.txt")):
        if time.time() - os.path.getmtime(p) > 3 * 3600:
            continue
        txt = open(p, encoding="utf-8", errors="replace").read()
        wins = re.findall(r"^(\d\d:\d\d:\d\d) \S+ (\S+)\s.*?(\d+) -> (\d+)", txt, re.M)
        fin = re.search(r"final (\d+)", txt)
        err = "Traceback" in txt
        last = wins[-1] if wins else None
        alive = time.time() - os.path.getmtime(p) < 120 and not fin
        rows.append(dict(name=os.path.basename(p)[:-4], wins=len(wins), last=last, done=bool(fin), err=err,
                         alive=alive, gain=(int(wins[0][2]) - int(wins[-1][3])) if wins else 0))
    return rows


def fmt(v):
    return f"{v:,}".replace(",", " ") if isinstance(v, int) else "—"


def chart(hist):
    if len(hist) < 2:
        return '<p class="muted">Chart appears after the second data point.</p>'
    W, H, L, B, T = 760, 220, 92, 28, 12
    ts = [h["t"] for h in hist]
    vs = [h["total"] for h in hist]
    lo, hi = min(vs + [BOARD["first"]["total"]]), max(vs)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    t0, t1 = ts[0], max(ts[-1], ts[0] + 1)
    X = lambda t: L + (W - L - 10) * (t - t0) / (t1 - t0)  # noqa: E731
    Y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)  # noqa: E731
    pts = []
    for i, (t, v) in enumerate(zip(ts, vs)):  # step line: the total only changes when a champion improves
        if i:
            pts.append(f"{X(t):.1f},{Y(vs[i - 1]):.1f}")
        pts.append(f"{X(t):.1f},{Y(v):.1f}")
    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid += (f'<line x1="{L}" x2="{W - 10}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" class="axis" text-anchor="end">{v / 1e6:.1f}M</text>')
    for k in range(4):
        t = t0 + (t1 - t0) * k / 3
        grid += (f'<text x="{X(t):.1f}" y="{H - 8}" class="axis" text-anchor="middle">'
                 f'{datetime.datetime.fromtimestamp(t):%H:%M}</text>')
    ref = BOARD["first"]["total"]
    refline = (f'<line x1="{L}" x2="{W - 10}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" class="ref"/>'
               f'<text x="{W - 12}" y="{Y(ref) - 6:.1f}" class="axis" text-anchor="end">#1 {ref / 1e6:.2f}M</text>')
    data = json.dumps([[X(t), Y(v), f"{datetime.datetime.fromtimestamp(t):%H:%M:%S}", fmt(v)] for t, v in zip(ts, vs)])
    last = f'<circle cx="{X(ts[-1]):.1f}" cy="{Y(vs[-1]):.1f}" r="4" class="dot"/>'
    return f"""<svg id="ch" viewBox="0 0 {W} {H}" role="img" aria-label="Team best total over time">
{grid}{refline}<polyline points="{' '.join(pts)}" class="line"/>{last}
<line id="cx" y1="{T}" y2="{H - B}" class="cross" visibility="hidden"/><circle id="cd" r="5" class="dot" visibility="hidden"/>
<rect x="{L}" y="{T}" width="{W - L - 10}" height="{H - T - B}" fill="transparent" id="hit"/></svg>
<div id="tip" class="tip" hidden></div>
<script>const D={data};const s=document.getElementById('ch'),tip=document.getElementById('tip'),cx=document.getElementById('cx'),cd=document.getElementById('cd');
document.getElementById('hit').addEventListener('mousemove',e=>{{const p=s.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const q=p.matrixTransform(s.getScreenCTM().inverse());
let b=D[0];for(const d of D)if(Math.abs(d[0]-q.x)<Math.abs(b[0]-q.x))b=d;cx.setAttribute('x1',b[0]);cx.setAttribute('x2',b[0]);cd.setAttribute('cx',b[0]);cd.setAttribute('cy',b[1]);
cx.setAttribute('visibility','visible');cd.setAttribute('visibility','visible');tip.hidden=false;tip.innerHTML='<b>'+b[3]+'</b><br>'+b[2];
const r=s.getBoundingClientRect();tip.style.left=(r.left+window.scrollX+b[0]/{W}*r.width+12)+'px';tip.style.top=(r.top+window.scrollY+b[1]/{H}*r.height-10)+'px';}});
document.getElementById('hit').addEventListener('mouseleave',()=>{{tip.hidden=true;cx.setAttribute('visibility','hidden');cd.setAttribute('visibility','hidden');}});</script>"""


def render(best, hist, ws):
    total = sum(b[0] for b in best.values() if b) if all(best.values()) else None
    sub = sum(BOARD["submitted"].values())
    first = BOARD["first"]["total"]
    rows = ""
    for n in NAMES:
        b = best[n]
        if not b:
            rows += f"<tr><td>{n}</td><td colspan=5>no valid file</td></tr>"
            continue
        d_sub, d_best = b[0] - BOARD["submitted"][n], b[0] - BOARD["best"][n]
        rows += (f"<tr><td>{n}</td><td class=num><b>{fmt(b[0])}</b></td><td class=num>{fmt(d_sub)}</td>"
                 f"<td class=num>{fmt(d_best)}</td><td>s{b[2]}</td><td class=path>{html.escape(b[1])}</td></tr>")
    wrows = ""
    for w in ws:
        state = "error" if w["err"] else ("finished" if w["done"] else ("running" if w["alive"] else "idle"))
        last = f"{w['last'][0]} {w['last'][1]} → {fmt(int(w['last'][3]))}" if w["last"] else "—"
        wrows += (f"<tr><td>{html.escape(w['name'])}</td><td><span class='st {state}'>{state}</span></td>"
                  f"<td class=num>{w['wins']}</td><td class=num>{fmt(w['gain'])}</td><td>{last}</td></tr>")
    gap = f"{(total - first) / 1e6:+.2f}M vs #1 ({BOARD['first']['team']} {first / 1e6:.2f}M)" if total else ""
    vs_sub = f"{(total - sub) / 1e6:+.2f}M vs submitted board ({sub / 1e6:.2f}M)" if total else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>KIRO live — {fmt(total)}</title><style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#2a78d6;--grid:#e6e5e1;--card:#f4f3f0;--good:#0ca30c;--bad:#d03b3b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--line:#3987e5;--grid:#383835;--card:#232322;--good:#0ca30c;--bad:#e66767}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:980px;padding:0 16px}}
h1{{font-size:18px;margin:0 0 2px}} .muted{{color:var(--mut)}} .hero{{display:flex;gap:16px;margin:16px 0}}
.card{{background:var(--card);border-radius:10px;padding:14px 18px;flex:1}} .big{{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums}}
table{{border-collapse:collapse;width:100%;margin:8px 0 20px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}}
th{{color:var(--mut);font-weight:500}} .num{{text-align:right;font-variant-numeric:tabular-nums}} .path{{color:var(--mut);font-size:12px}}
.grid{{stroke:var(--grid);stroke-width:1}} .axis{{fill:var(--mut);font-size:11px}} .line{{fill:none;stroke:var(--line);stroke-width:2}}
.ref{{stroke:var(--mut);stroke-width:1;stroke-dasharray:4 4}} .dot{{fill:var(--line);stroke:var(--bg);stroke-width:2}} .cross{{stroke:var(--mut);stroke-width:1}}
.tip{{position:absolute;background:var(--card);border:1px solid var(--grid);border-radius:6px;padding:4px 8px;font-size:12px;pointer-events:none}}
.st{{font-size:12px;padding:1px 8px;border-radius:9px;border:1px solid var(--grid)}} .st.running::before{{content:'● ';color:var(--good)}}
.st.error::before{{content:'✖ ';color:var(--bad)}} .st.finished::before{{content:'✔ '}} .st.idle::before{{content:'○ '}}
</style></head><body>
<h1>KIRO 2026 — team double span, live results</h1>
<div class="muted">Updated {datetime.datetime.now():%H:%M:%S} · every file validated by evaluator.py · refreshes every 30 s</div>
<div class="hero"><div class="card"><div class="muted">Team best (sum of best valid per instance)</div><div class="big">{fmt(total)}</div>
<div class="muted">{gap}</div></div><div class="card"><div class="muted">Versus what is on the leaderboard</div>
<div class="big">{(total - sub) / 1e6 if total else 0:+.2f}M</div><div class="muted">{vs_sub}</div></div></div>
<h2 style="font-size:15px">Best valid solution per instance</h2>
<table><tr><th>instance</th><th class=num>score</th><th class=num>vs submitted</th><th class=num>vs best on board</th><th>worst</th><th>file</th></tr>{rows}</table>
<h2 style="font-size:15px">Team best total over time</h2>{chart(hist)}
<h2 style="font-size:15px">LNS workers</h2>
<table><tr><th>log</th><th>state</th><th class=num>wins</th><th class=num>gain</th><th>last improvement</th></tr>{wrows}</table>
</body></html>"""


def main():
    insts = {n: json.load(open(f"instances/{n}.json")) for n in NAMES}
    hist = json.load(open(HIST)) if os.path.exists(HIST) else []
    while True:
        best = collect(insts)
        if all(best.values()):
            total = sum(b[0] for b in best.values())
            if not hist or hist[-1]["total"] != total:
                hist.append({"t": time.time(), "total": total, **{n: best[n][0] for n in NAMES}})
                json.dump(hist, open(HIST, "w"))
        tmp = OUT + ".tmp"
        open(tmp, "w", encoding="utf-8").write(render(best, hist, workers()))
        os.replace(tmp, OUT)
        time.sleep(30)


if __name__ == "__main__":
    main()
