"""Independent reference evaluator for KIRO 2026 (written from the official PDF, sections 2.b-2.e and 4).

evaluate(instance, solution) -> {"valid": bool, "score": int|None, "scenario_costs": {sid: int}, "errors": [str]}
Never imports solver code. Exact integer arithmetic; alpha thresholds via Fraction(str(alpha)).
"""
import json
import math
import sys
from fractions import Fraction


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _parse(instance, solution, errors):
    """Strict parsing of the solution JSON. Returns (x: {route: n}, y: {(i, j): n}) or None."""
    nR = len(instance["routes"])
    if not isinstance(solution, dict) or set(solution.keys()) - {"x", "y"} or "x" not in solution:
        errors.append("solution must be an object with keys 'x' and optional 'y' only")
        return None
    xs, ys = solution["x"], solution.get("y", [])
    if not isinstance(xs, list) or not isinstance(ys, list):
        errors.append("'x' and 'y' must be lists")
        return None
    x, y = {}, {}
    for e in xs:
        if not isinstance(e, dict) or set(e.keys()) != {"route", "n"}:
            errors.append(f"bad x entry {e!r}")
            continue
        r, n = e["route"], e["n"]
        if not (_is_int(r) and 0 <= r < nR):
            errors.append(f"x: unknown route {r!r}")
        elif not (_is_int(n) and n > 0):
            errors.append(f"x: route {r} has non-positive or non-integer n={n!r}")
        elif r in x:
            errors.append(f"x: route {r} listed twice")
        else:
            x[r] = n
    for e in ys:
        if not isinstance(e, dict) or set(e.keys()) != {"from", "to", "n"}:
            errors.append(f"bad y entry {e!r}")
            continue
        i, j, n = e["from"], e["to"], e["n"]
        if not (_is_int(i) and 0 <= i < nR and _is_int(j) and 0 <= j < nR):
            errors.append(f"y: unknown route in pair ({i!r},{j!r})")
        elif not (_is_int(n) and n > 0):
            errors.append(f"y: pair ({i},{j}) has non-positive or non-integer n={n!r}")
        elif (i, j) in y:
            errors.append(f"y: pair ({i},{j}) listed twice")
        else:
            y[(i, j)] = n
    return None if errors else (x, y)


def evaluate(instance, solution):
    errors = []
    res = {"valid": False, "score": None, "scenario_costs": {}, "errors": errors}
    parsed = _parse(instance, solution, errors)
    if parsed is None:
        return res
    x, y = parsed
    ports, sites, carriers = instance["ports"], instance["sites"], instance["carriers"]
    flows, routes, scenarios = instance["flows"], instance["routes"], instance["scenarios"]
    spot = {c["id"] for c in carriers if c["name"] == "SPOT"}
    alpha = Fraction(str(instance["meta"]["alpha"]))
    min_vol = {c["id"]: math.ceil(alpha * c["trans"]) for c in carriers}
    truck = {}
    for t in sites:
        for e in t["ports"]:
            truck[(t["id"], e["port"])] = e["cost"]

    def tc(t, p):
        if (t, p) not in truck:
            raise KeyError(f"no truck cost between site {t} and port {p}")
        return truck[(t, p)]

    # ---- nominal flow coherence
    used = {f["id"]: [] for f in flows}
    for r, n in x.items():
        used[routes[r]["flow"]].append(r)
    for f in flows:
        rs = used[f["id"]]
        for r in rs:
            if r not in f["routes"]:
                errors.append(f"flow {f['id']}: route {r} is not a candidate of this flow")
        if len(rs) > 2:
            errors.append(f"flow {f['id']}: uses {len(rs)} routes (>2)")
        tot = sum(x[r] for r in rs)
        if tot != f["n"]:
            errors.append(f"flow {f['id']}: assigned {tot} != n_f={f['n']}")
    # ---- nominal reuse compatibility
    for (i, j), n in y.items():
        ri, rj = routes[i], routes[j]
        if ri["carrier"] != rj["carrier"]:
            errors.append(f"y ({i},{j}): different carriers {ri['carrier']} != {rj['carrier']}")
        if flows[ri["flow"]]["t_arr"] != flows[rj["flow"]]["t_dep"]:
            errors.append(f"y ({i},{j}): t_arr(f{ri['flow']}) != t_dep(f{rj['flow']})")
    if errors:
        return res

    def unit_cost(r):
        ro, f = routes[r], flows[routes[r]["flow"]]
        return 2 * tc(f["t_dep"], ro["p_dep"]) + ro["cost"] + 2 * tc(f["t_arr"], ro["p_arr"])

    def saving(i, j):
        ri, rj = routes[i], routes[j]
        return tc(flows[ri["flow"]]["t_arr"], ri["p_arr"]) + tc(flows[rj["flow"]]["t_dep"], rj["p_dep"])

    for sc in scenarios:
        sid = sc["id"]
        failed = set(sc["routes"])
        serr = []
        # automatic rerouting (2.c rules 1-3)
        xs = {}
        for f in flows:
            rs = used[f["id"]]
            alive = [r for r in rs if r not in failed]
            if not alive:
                serr.append(f"scenario {sid}: flow {f['id']} lost all its routes {rs}")
                continue
            if len(alive) == len(rs):
                for r in rs:
                    xs[r] = x[r]
            else:  # two routes, exactly one failed
                xs[alive[0]] = f["n"]
        ys = {(i, j): n for (i, j), n in y.items() if i not in failed and j not in failed}
        # (7) per-route reuse bounds, separately for outgoing and incoming
        out_, in_ = {}, {}
        for (i, j), n in ys.items():
            out_[i] = out_.get(i, 0) + n
            in_[j] = in_.get(j, 0) + n
        for r, v in out_.items():
            if v > xs.get(r, 0):
                serr.append(f"scenario {sid}: route {r} supplies {v} reused > x^s={xs.get(r, 0)}")
        for r, v in in_.items():
            if v > xs.get(r, 0):
                serr.append(f"scenario {sid}: route {r} receives {v} reused > x^s={xs.get(r, 0)}")
        # (8) site reuse capacity (counted at arrival site of the supplying flow)
        site_reuse = {}
        for (i, j), n in ys.items():
            t = flows[routes[i]["flow"]]["t_arr"]
            site_reuse[t] = site_reuse.get(t, 0) + n
        for t, v in site_reuse.items():
            if v > sites[t]["reutil"]:
                serr.append(f"scenario {sid}: site {t} reuse {v} > reutil {sites[t]['reutil']}")
        # (9) port handling: departures + arrivals (self-loop counts twice)
        pload = {}
        for r, n in xs.items():
            pload[routes[r]["p_dep"]] = pload.get(routes[r]["p_dep"], 0) + n
            pload[routes[r]["p_arr"]] = pload.get(routes[r]["p_arr"], 0) + n
        for p, v in pload.items():
            if v > ports[p]["manut"]:
                serr.append(f"scenario {sid}: port {p} handles {v} > manut {ports[p]['manut']}")
        # carriers: transport capacity, minimum volume, number of sites
        cload, csites = {}, {}
        for r, n in xs.items():
            m, f = routes[r]["carrier"], flows[routes[r]["flow"]]
            cload[m] = cload.get(m, 0) + n
            csites.setdefault(m, set()).update((f["t_dep"], f["t_arr"]))
        for m, v in cload.items():
            if v > carriers[m]["trans"]:
                serr.append(f"scenario {sid}: carrier {m} carries {v} > trans {carriers[m]['trans']}")
            if m not in spot and 0 < v < min_vol[m]:
                serr.append(f"scenario {sid}: carrier {m} carries {v} < ceil(alpha*trans)={min_vol[m]}")
        for m, ts in csites.items():
            if len(ts) > carriers[m]["sites"]:
                serr.append(f"scenario {sid}: carrier {m} exchanges with {len(ts)} sites > {carriers[m]['sites']}")
        # cost (5)
        cost = sum(unit_cost(r) * n for r, n in xs.items())
        cost -= sum(saving(i, j) * n for (i, j), n in ys.items())
        cost += sum(carriers[m]["fixed_cost"] * len(ts) for m, ts in csites.items())
        res["scenario_costs"][sid] = cost
        errors.extend(serr)
    if not errors:
        res["valid"] = True
        res["score"] = max(res["scenario_costs"].values())
    return res


def worst_scenario(result):
    sc = result["scenario_costs"]
    return max(sc, key=sc.get) if sc else None


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    r = evaluate(inst, sol)
    print("valid:", r["valid"], "score:", r["score"], "worst scenario:", worst_scenario(r))
    for e in r["errors"][:30]:
        print("  ", e)
    sys.exit(0 if r["valid"] else 1)
