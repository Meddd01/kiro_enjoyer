"""All-SPOT baseline (always feasible) + greedy reuse helper shared by Claude's solvers."""
import json
import sys


def spot_ids(instance):
    m = [c["id"] for c in instance["carriers"] if c["name"] == "SPOT"]
    p = [q["id"] for q in instance["ports"] if q["code"] == "SPOT"]
    assert len(m) == 1 and len(p) == 1, "SPOT carrier/port not found uniquely"
    return m[0], p[0]


def truck_costs(instance):
    return {(t["id"], e["port"]): e["cost"] for t in instance["sites"] for e in t["ports"]}


def greedy_reuse(instance, x, weight=None):
    """Greedy nominal reuse for allocation x {route: n}.

    Respects: same carrier, t_arr(i) == t_dep(j), sum_out(i) <= x_i, sum_in(j) <= x_j (separately),
    reutil(t) shared by all carriers at site t. Pairs ranked by saving * weight(i, j) (default 1).
    Returns list of {"from","to","n"} sorted by (from, to).
    """
    routes, flows = instance["routes"], instance["flows"]
    tc = truck_costs(instance)
    arr, dep = {}, {}
    for r, n in x.items():
        if n <= 0:
            continue
        ro, f = routes[r], flows[routes[r]["flow"]]
        arr.setdefault((ro["carrier"], f["t_arr"]), []).append(r)
        dep.setdefault((ro["carrier"], f["t_dep"]), []).append(r)
    cand = []
    for key, ins in arr.items():
        outs = dep.get(key)
        if not outs:
            continue
        t = key[1]
        for i in ins:
            b = tc[(t, routes[i]["p_arr"])]
            for j in outs:
                s = b + tc[(t, routes[j]["p_dep"])]
                w = s * (weight(i, j) if weight else 1)
                if s > 0:
                    cand.append((-w, -s, i, j, t))
    cand.sort()
    cap_out, cap_in = dict(x), dict(x)
    cap_site = {t["id"]: t["reutil"] for t in instance["sites"]}
    y = {}
    for _, _, i, j, t in cand:
        q = min(cap_out[i], cap_in[j], cap_site[t])
        if q > 0:
            y[(i, j)] = y.get((i, j), 0) + q
            cap_out[i] -= q
            cap_in[j] -= q
            cap_site[t] -= q
    return [{"from": i, "to": j, "n": n} for (i, j), n in sorted(y.items())]


def to_solution(x, y):
    return {"x": [{"route": r, "n": n} for r, n in sorted(x.items()) if n > 0], "y": y}


def solve(instance, reuse=True):
    spot_m, _ = spot_ids(instance)
    routes = instance["routes"]
    x = {}
    for f in instance["flows"]:
        cand = [r for r in f["routes"] if routes[r]["carrier"] == spot_m]
        assert len(cand) == 1, f"flow {f['id']} has {len(cand)} SPOT routes"
        r = cand[0]
        assert all(r not in s["routes"] for s in instance["scenarios"]), "SPOT route fails somewhere"
        x[r] = f["n"]
    return to_solution(x, greedy_reuse(instance, x) if reuse else [])


if __name__ == "__main__":
    import os
    from evaluator import evaluate, worst_scenario
    os.makedirs("runs/claude", exist_ok=True)
    for path in sys.argv[1:]:
        inst = json.load(open(path))
        name = inst["meta"]["name"]
        for reuse in (False, True):
            sol = solve(inst, reuse)
            res = evaluate(inst, sol)
            reused = sum(e["n"] for e in sol["y"])
            out = f"runs/claude/baseline_{'' if reuse else 'noreuse_'}{name}.json"
            json.dump(sol, open(out, "w"))
            print(f"{name:12s} reuse={reuse!s:5s} valid={res['valid']} score={res['score']} "
                  f"worst={worst_scenario(res)} reusedTEU={reused} -> {out}", res["errors"][:3])
