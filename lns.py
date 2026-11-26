"""Claude's LNS driver: chooses neighbourhoods (sets of free flows) and delegates their exact re-optimisation
to Codex's MILP engine (milp_solver, feat/optimization). Accepts only evaluator-valid strict improvements.

python lns.py --instance strikes --time 600 --sub 20 --seed 0 [--init solutions/claude] [--out solutions/<you>]
Needs Codex files (milp_solver.py, common.py, reuse.py) importable, e.g. PYTHONPATH=<codex checkout>.
"""
import argparse
import datetime
import json
import random
import sys
import time
from collections import Counter

from evaluator import evaluate, worst_scenario


_UC = {}


def unit_costs(inst):
    key = id(inst)
    if key not in _UC:
        tc = {(t["id"], e["port"]): e["cost"] for t in inst["sites"] for e in t["ports"]}
        fl = inst["flows"]
        _UC[key] = [2 * tc[(fl[r["flow"]]["t_dep"], r["p_dep"])] + r["cost"] + 2 * tc[(fl[r["flow"]]["t_arr"], r["p_arr"])]
                    for r in inst["routes"]]
    return _UC[key]


def neighbourhoods(inst, sol, res, rng, size):
    routes, flows, carriers = inst["routes"], inst["flows"], inst["carriers"]
    spot = next(c["id"] for c in carriers if c["name"] == "SPOT")
    used = {}
    for e in sol["x"]:
        used.setdefault(routes[e["route"]]["flow"], []).append(e["route"])
    kind = rng.choices(["site", "worst", "carrier", "spot", "regret"], weights=[0.2, 0.2, 0.2, 0.2, 0.2])[0]
    if kind == "site":  # sites weighted by SPOT volume touching them
        w = Counter()
        for e in sol["x"]:
            if routes[e["route"]]["carrier"] == spot:
                f = flows[routes[e["route"]]["flow"]]
                w[f["t_dep"]] += e["n"]
                w[f["t_arr"]] += e["n"]
        sites = rng.choices(list(w), weights=list(w.values()), k=2) if w else [rng.randrange(len(inst["sites"]))]
        free = [f["id"] for f in flows if f["t_dep"] in sites or f["t_arr"] in sites]
    elif kind == "worst":
        s = worst_scenario(res)
        failed = set(inst["scenarios"][s]["routes"])
        free = [f for f, rs in used.items() if any(r in failed for r in rs)]
    elif kind == "carrier":
        m = rng.choice([c["id"] for c in carriers if c["id"] != spot])
        free = [f["id"] for f in flows if any(routes[r]["carrier"] == m for r in f["routes"])]
    elif kind == "regret":  # flows paying most above their cheapest candidate route (nominal inefficiency)
        cost = unit_costs(inst)
        xs = {e["route"]: e["n"] for e in sol["x"]}
        reg = {f: sum(xs[r] * cost[r] for r in rs) - flows[f]["n"] * min(cost[r] for r in flows[f]["routes"])
               for f, rs in used.items()}
        free = sorted(reg, key=lambda f: -reg[f])[: 2 * size]
    else:
        free = sorted(used, key=lambda f: -flows[f]["n"] * any(routes[r]["carrier"] == spot for r in used[f]))[: 2 * size]
    rng.shuffle(free)
    return kind, set(free[:size])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--time", type=float, default=600)
    ap.add_argument("--sub", type=float, default=20, help="seconds per neighbourhood solve")
    ap.add_argument("--size", type=int, default=25, help="max free flows per neighbourhood")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="solutions/claude", help="folder with the starting <name>.json")
    ap.add_argument("--out", default="solutions/claude", help="folder where improved <name>.json is written")
    a = ap.parse_args()
    import milp_solver as ms  # Codex engine (optional dependency, feat/optimization)
    from common import Problem
    name = a.instance
    inst = json.load(open(f"instances/{name}.json"))
    import os
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{name}.json")
    sol = json.load(open(os.path.join(a.init, f"{name}.json")))
    if os.path.exists(path):
        other = json.load(open(path))
        if evaluate(inst, other)["valid"] and evaluate(inst, other)["score"] < evaluate(inst, sol)["score"]:
            sol = other
    res = evaluate(inst, sol)
    assert res["valid"]
    problem = Problem(inst)
    rng = random.Random(a.seed)
    t_end = time.monotonic() + a.time
    stats = Counter()
    while time.monotonic() + 3 < t_end:
        kind, free = neighbourhoods(inst, sol, res, rng, rng.randint(max(5, a.size - 10), a.size + 15))
        if not free:
            continue
        t0 = time.monotonic()
        pc, rl = rng.choice([(2, 6), (2, 8), (3, 10), (3, 12)])
        try:
            trial = ms._candidate(problem, sol, min(t_end, time.monotonic() + a.sub), rng.randrange(10**6),
                                  free=free, per_carrier=pc, route_limit=rl, reuse=True)
            cand = ms.pack(*ms.unpack(ms._best(inst, problem, sol, trial)))
        except Exception as exc:  # solver hiccup: skip this neighbourhood
            print("neighbourhood failed:", repr(exc), file=sys.stderr)
            continue
        r2 = evaluate(inst, cand)
        stats[kind + "_tried"] += 1
        if r2["valid"] and r2["score"] < res["score"]:
            stats[kind + "_won"] += 1
            print(f"{datetime.datetime.now():%H:%M:%S} {name} {kind:7s} |free|={len(free):2d} pc={pc} rl={rl} "
                  f"{res['score']} -> {r2['score']} ({time.monotonic() - t0:.0f}s)", flush=True)
            cur = evaluate(inst, json.load(open(path))) if os.path.exists(path) else {"score": None}
            if cur["score"] is None or r2["score"] < cur["score"]:  # another process may have improved it
                json.dump(cand, open(path, "w"))
                meta = {"when": datetime.datetime.now().isoformat(timespec="seconds"), "instance": name,
                        "score": r2["score"], "worst": worst_scenario(r2), "valid": True,
                        "pipeline": "lns.py neighbourhoods -> codex milp_solver._neighborhood (@ca27544)",
                        "command": " ".join(sys.argv), "scenario_costs": r2["scenario_costs"]}
                json.dump(meta, open(os.path.join(a.out, f"{name}.meta.json"), "w"), indent=1)
            sol, res = cand, r2
    print(f"{name} final {res['score']} stats {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
