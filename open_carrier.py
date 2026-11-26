"""Structural move: open (or grow) carrier m by freeing the flows with the largest gain on m's routes and letting
Codex's exact MILP pick m's sites, routes and splits (lexicographic objective). Repeats with growing scopes.

PYTHONPATH=.:../kiro_codex_view python open_carrier.py --instance strikes --carrier G --k 45 --sub 600 --out solutions/OC_G
"""
import argparse
import json
import os
import random
import time
from collections import Counter

from evaluator import evaluate
import lns


def gains(inst, sol, m):
    routes, flows = inst["routes"], inst["flows"]
    cost = lns.unit_costs(inst)
    cur = Counter()
    for e in sol["x"]:
        cur[routes[e["route"]]["flow"]] += e["n"] * cost[e["route"]]
    g = {}
    for f in flows:
        rs = [r for r in f["routes"] if routes[r]["carrier"] == m]
        if rs:
            g[f["id"]] = f["n"] * (cur[f["id"]] / f["n"] - min(cost[r] for r in rs))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--carrier", required=True, help="carrier letter, e.g. G")
    ap.add_argument("--k", type=int, default=45)
    ap.add_argument("--sub", type=float, default=600)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--pc", type=int, default=2)
    ap.add_argument("--rl", type=int, default=0, help="route_limit (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="solutions/FINAL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import milp_solver as ms
    from common import Problem
    inst = json.load(open(f"instances/{a.instance}.json"))
    m = next(c["id"] for c in inst["carriers"] if c["name"] == f"CARRIER_{a.carrier}")
    sol = json.load(open(os.path.join(a.init, f"{a.instance}.json")))
    res = evaluate(inst, sol)
    problem = Problem(inst)
    lns.install_lex(ms)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for rd in range(a.rounds):
        g = gains(inst, sol, m)
        ranked = [f for f, v in sorted(g.items(), key=lambda kv: -kv[1]) if v > 0]
        k = a.k + 10 * rd
        pool = ranked[: int(k * 1.3)]
        rng.shuffle(pool)
        free = set(sorted(pool, key=lambda f: -g[f])[:k])
        sc = res["scenario_costs"]
        top = max(sc.values())
        lns.LEX.update(on=True, w={s: (1.0 if c >= top * 0.995 else 0.05) for s, c in sc.items()})
        t0 = time.monotonic()
        trial = ms._candidate(problem, sol, time.monotonic() + a.sub, rng.randrange(10**6), free=free,
                              per_carrier=a.pc, route_limit=a.rl or None, reuse=True)
        cand = ms.pack(*ms.unpack(trial[0])) if trial else None
        r2 = evaluate(inst, cand) if cand else {"valid": False, "score": None}
        load = sum(e["n"] for e in (cand or sol)["x"] if inst["routes"][e["route"]]["carrier"] == m)
        print(f"{time.strftime('%H:%M:%S')} round {rd} |free|={len(free)} trial={r2['score']} inc={res['score']} "
              f"load_{a.carrier}={load} ({time.monotonic() - t0:.0f}s)", flush=True)
        if r2["valid"] and r2["score"] < res["score"]:
            sol, res = cand, r2
            json.dump(sol, open(os.path.join(a.out, f"{a.instance}.json"), "w"))


if __name__ == "__main__":
    main()
