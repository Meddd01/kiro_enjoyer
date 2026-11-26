"""Claude's elite pool + MILP crossover for lns.py --mode cross.

Pool: every valid candidate file (solutions/**/<name>.json incl. history/, codex_teamwarmup/), best first, keeping a
solution only if its per-flow support differs from every kept one on >= 30% of flows (Hamming diversity), K <= 20.
Crossover: child starts from parent A with a random half of the disagreeing flows copied from parent B (so both
parents' routes are candidates), then the disagreeing flows (<= cap) are re-solved exactly by Codex's MILP.
"""
import glob
import json

from evaluator import evaluate


def support(inst, sol):
    routes = inst["routes"]
    sup = [()] * len(inst["flows"])
    acc = {}
    for e in sol["x"]:
        acc.setdefault(routes[e["route"]]["flow"], []).append(e["route"])
    for f, rs in acc.items():
        sup[f] = tuple(sorted(rs))
    return sup


def load_pool(inst, name, k=20, min_diff=0.3):
    cands = []
    seen = set()
    for p in glob.glob(f"solutions/**/{name}.json", recursive=True) + glob.glob(f"solutions/**/history/*_{name}.json",
                                                                                 recursive=True) + [f"codex_teamwarmup/{name}.json"]:
        try:
            sol = json.load(open(p))
        except (OSError, ValueError):
            continue
        r = evaluate(inst, sol)
        if not r["valid"]:
            continue
        key = json.dumps(sol["x"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        cands.append((r["score"], p, sol))
    cands.sort(key=lambda c: c[0])
    pool = []
    F = len(inst["flows"])
    for score, p, sol in cands:
        sup = support(inst, sol)
        if all(sum(a != b for a, b in zip(sup, q["sup"])) >= min_diff * F for q in pool):
            pool.append({"score": score, "path": p, "sol": sol, "sup": sup})
        if len(pool) >= k:
            break
    if not pool and cands:
        pool.append({"score": cands[0][0], "path": cands[0][1], "sol": cands[0][2], "sup": support(inst, cands[0][2])})
    return pool


def crossover(inst, problem, ms, A, B, rng, deadline, cap=60):
    """Returns (child_solution or None, number of freed flows)."""
    diff = [f for f in range(len(inst["flows"])) if A["sup"][f] != B["sup"][f]]
    if not diff:
        return None, 0
    free = set(rng.sample(diff, min(cap, len(diff))))
    take_b = set(rng.sample(sorted(free), len(free) // 2))  # every changed flow is free -> repair stays feasible
    routes = inst["routes"]
    x = [e for e in A["sol"]["x"] if routes[e["route"]]["flow"] not in take_b]
    x += [e for e in B["sol"]["x"] if routes[e["route"]]["flow"] in take_b]
    child0 = {"x": sorted(x, key=lambda e: e["route"]), "y": []}  # carries B's routes into the candidate set
    trial = ms._candidate(problem, child0, deadline, rng.randrange(10**6), free=free, per_carrier=2, route_limit=6,
                          reuse=True)
    if trial is None:
        return None, len(free)
    child = ms.pack(*ms.unpack(trial[0]))
    return (child if evaluate(inst, child)["valid"] else None), len(free)
