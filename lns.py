"""Claude's LNS driver: chooses neighbourhoods (sets of free flows) and delegates their exact re-optimisation
to Codex's MILP engine (milp_solver, feat/optimization). Accepts only evaluator-valid strict improvements.

python lns.py --instance strikes --time 600 --sub 20 --seed 0 [--init solutions/claude] [--out solutions/<you>]
Needs Codex files (milp_solver.py, common.py, reuse.py) importable, e.g. PYTHONPATH=<codex checkout>.
"""
import argparse
import datetime
import glob
import os
import json
import math
import shutil
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


KINDS = ["site", "worst", "carrier", "spot", "regret", "topk", "port", "swap", "open", "bigspot", "unlock"]


def neighbourhoods(inst, sol, res, rng, size, weights=None):
    routes, flows, carriers = inst["routes"], inst["flows"], inst["carriers"]
    spot = next(c["id"] for c in carriers if c["name"] == "SPOT")
    used = {}
    for e in sol["x"]:
        used.setdefault(routes[e["route"]]["flow"], []).append(e["route"])
    kinds = KINDS
    kind = rng.choices(kinds, weights=weights or [1.0] * len(kinds))[0]
    if kind == "unlock":
        return unlock(inst, sol, rng, size)
    if kind == "bigspot":
        return bigspot(inst, sol, rng, size)
    if kind in ("swap", "open"):
        return structural(inst, sol, res, rng, size, kind)
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
    elif kind == "topk":  # flows hit by several of the (near-)tied top scenarios: lowers them together
        sc = res["scenario_costs"]
        top = sorted(sc, key=sc.get, reverse=True)[: rng.randint(2, 5)]
        hits = Counter()
        for s in top:
            failed = set(inst["scenarios"][s]["routes"])
            for f, rs in used.items():
                if any(r in failed for r in rs):
                    hits[f] += 1
        free = [f for f, _ in hits.most_common(3 * size)]
        rng.shuffle(free)
        free = sorted(free, key=lambda f: -hits[f])[: 2 * size]
    elif kind == "port":  # saturated port + SPOT flows whose cheap real routes need it (capacity swap)
        load = Counter()
        xs = {e["route"]: e["n"] for e in sol["x"]}
        for sc_ in inst["scenarios"]:
            failed = set(sc_["routes"])
            ls = Counter()
            for f, rs in used.items():
                alive = [r for r in rs if r not in failed]
                for r in alive:
                    q = xs[r] if len(alive) == len(rs) else flows[f]["n"]
                    ls[routes[r]["p_dep"]] += q
                    ls[routes[r]["p_arr"]] += q
            for p_, v in ls.items():
                load[p_] = max(load[p_], v)
        cost = unit_costs(inst)
        want = Counter()
        for f, rs in used.items():
            if len(rs) == 1 and routes[rs[0]]["carrier"] == spot:
                for r in flows[f]["routes"]:
                    if routes[r]["carrier"] != spot and cost[r] < cost[rs[0]]:
                        for p_ in (routes[r]["p_dep"], routes[r]["p_arr"]):
                            if load[p_] >= 0.98 * inst["ports"][p_]["manut"]:
                                want[p_] += flows[f]["n"] * (cost[rs[0]] - cost[r])
        if not want:
            return kind, set()
        p_ = rng.choices(list(want), weights=list(want.values()))[0]
        seekers = [f for f, rs in used.items() if len(rs) == 1 and routes[rs[0]]["carrier"] == spot
                   and any(p_ in (routes[r]["p_dep"], routes[r]["p_arr"]) for r in flows[f]["routes"])]
        holders = [f for f, rs in used.items() if any(p_ in (routes[r]["p_dep"], routes[r]["p_arr"]) for r in rs)]
        rng.shuffle(holders)
        free = seekers[: size // 2] + holders
        return kind, set(free[: size + 10])
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


def site_structure(inst, sol):
    """Carrier site sets S[m], users[(m, t)] (flows using m at site t), volume at (m, t), and per-flow
    opening gains: gain[(m, t_new)] = sum over flows whose cheapest m-route needs exactly one site t_new outside S[m]
    of n_f * (current unit cost - that route's unit cost)."""
    routes, flows, carriers = inst["routes"], inst["flows"], inst["carriers"]
    spot = next(c["id"] for c in carriers if c["name"] == "SPOT")
    cost = unit_costs(inst)
    xs = {e["route"]: e["n"] for e in sol["x"]}
    S, users, vol = {}, {}, Counter()
    cur = Counter()
    for r, n in xs.items():
        m, f = routes[r]["carrier"], flows[routes[r]["flow"]]
        cur[f["id"]] += n * cost[r]
        for t in {f["t_dep"], f["t_arr"]}:
            S.setdefault(m, set()).add(t)
            users.setdefault((m, t), set()).add(f["id"])
            vol[(m, t)] += n
    gain, gainers = Counter(), {}
    for f in flows:
        unit = cur[f["id"]] / f["n"]
        best = {}
        for r in f["routes"]:
            m = routes[r]["carrier"]
            if m == spot:
                continue
            miss = {f["t_dep"], f["t_arr"]} - S.get(m, set())
            if not miss:
                continue
            g = f["n"] * (unit - cost[r])
            if g > 0:
                key = (m, frozenset(miss))
                best[key] = max(best.get(key, 0), g)
        for (m, miss), g in best.items():
            for t in miss:  # a flow needing two new sites credits both (half each)
                gain[(m, t)] += g / len(miss)
                gainers.setdefault((m, t), Counter())[f["id"]] += g / len(miss)
    return spot, S, users, vol, gain, gainers


def structural(inst, sol, res, rng, size, kind):
    """'swap': for a carrier at its sites(m) limit, close its lightest site t and open the best-gain site t';
    'open': for a carrier with free site slots (incl. dormant ones), open the best-gain sites.
    Frees every flow using m at t plus the top gainers of the new site(s); repair uses full per-carrier routes."""
    carriers = inst["carriers"]
    spot, S, users, vol, gain, gainers = site_structure(inst, sol)
    cands = []
    for (m, t), g in gain.items():
        full = len(S.get(m, ())) >= carriers[m]["sites"]
        if (kind == "swap") == full:
            cands.append(((m, t), g))
    if not cands:
        return kind, set()
    cands.sort(key=lambda c: -c[1])
    cands = cands[:25]
    (m, t_new), g = rng.choices(cands, weights=[c[1] for c in cands])[0]
    free = set()
    tag = f"{kind}:{carriers[m]['name'][-1]}+{t_new}"
    if kind == "swap":
        light = sorted((vol[(m, t)], t) for t in S[m] if t != t_new)
        t_old = rng.choice(light[:3])[1]
        free |= users[(m, t_old)]
        tag += f"-{t_old}"
        sites_new = [t_new]
    else:
        slots = carriers[m]["sites"] - len(S.get(m, ()))
        others = sorted(((gg, t) for (mm, t), gg in gain.items() if mm == m and t != t_new), reverse=True)
        sites_new = [t_new] + [t for _, t in others[: rng.randint(0, max(0, min(3, slots - 1)))]]
        tag += "".join(f"+{t}" for t in sites_new[1:])
    pool = Counter()
    for t in sites_new:
        pool.update(gainers.get((m, t), {}))
    room = max(8, size - len(free))
    free |= {f for f, _ in pool.most_common(room)}
    touch = [f["id"] for f in inst["flows"] if f["id"] not in free
             and ({f["t_dep"], f["t_arr"]} & (set(sites_new) | ({t_old} if kind == "swap" else set())))]
    rng.shuffle(touch)
    free |= set(touch[: max(0, size - len(free))])
    return tag, free


def bigspot(inst, sol, rng, size):
    """One big SPOT flow (weighted by n * (SPOT unit - cheapest real unit)) + the flows holding the ports and the
    carriers of its cheap real routes: room for a full robust pair (each route able to carry n_f alone)."""
    routes, flows, carriers = inst["routes"], inst["flows"], inst["carriers"]
    spot = next(c["id"] for c in carriers if c["name"] == "SPOT")
    cost = unit_costs(inst)
    xs = {e["route"]: e["n"] for e in sol["x"]}
    used = {}
    for r in xs:
        used.setdefault(routes[r]["flow"], []).append(r)
    w = {}
    for f, rs in used.items():
        sp = sum(xs[r] for r in rs if routes[r]["carrier"] == spot)
        if sp:
            real = [cost[r] for r in flows[f]["routes"] if routes[r]["carrier"] != spot]
            if real:
                w[f] = sp * max(1, cost[next(r for r in rs if routes[r]["carrier"] == spot)] - min(real))
    if not w:
        return "bigspot", set()
    top = sorted(w, key=w.get, reverse=True)[:8]
    f0 = rng.choices(top, weights=[w[f] for f in top])[0]
    cheap = sorted((r for r in flows[f0]["routes"] if routes[r]["carrier"] != spot), key=lambda r: cost[r])[:8]
    ports = {routes[r][k] for r in cheap for k in ("p_dep", "p_arr")}
    ms_ = {routes[r]["carrier"] for r in cheap[:4]}
    score = Counter()
    for f, rs in used.items():
        if f == f0:
            continue
        for r in rs:
            score[f] += xs[r] * ((routes[r]["p_dep"] in ports) + (routes[r]["p_arr"] in ports)
                                 + 0.5 * (routes[r]["carrier"] in ms_))
    hold = [f for f, v in score.most_common(3 * size) if v > 0]
    rng.shuffle(hold)
    return f"bigspot:f{f0}", {f0} | set(hold[: size - 1])


def unlock(inst, sol, rng, size):
    """Targeted unlock of a big SPOT flow f0 (what-if: f54 alone is worth -6..-9M): free f0, the holders of the
    ports of its cheap real routes ranked by cheapest detour (unit cost of their best route avoiding those ports
    minus current unit cost) until ~2.5 n_f0 of port volume, and the users of the lightest site of one full carrier
    that f0's cheap routes need."""
    routes, flows, carriers = inst["routes"], inst["flows"], inst["carriers"]
    spot = next(c["id"] for c in carriers if c["name"] == "SPOT")
    cost = unit_costs(inst)
    xs = {e["route"]: e["n"] for e in sol["x"]}
    used = {}
    for r in xs:
        used.setdefault(routes[r]["flow"], []).append(r)
    w = {}
    for f, rs in used.items():
        sp = [r for r in rs if routes[r]["carrier"] == spot]
        real = [cost[r] for r in flows[f]["routes"] if routes[r]["carrier"] != spot]
        if sp and real:
            w[f] = sum(xs[r] for r in sp) * max(1, cost[sp[0]] - min(real))
    if not w:
        return "unlock", set()
    top = sorted(w, key=w.get, reverse=True)[:4]
    f0 = rng.choices(top, weights=[w[f] ** 1.5 for f in top])[0]
    F0 = flows[f0]
    cheap = sorted((r for r in F0["routes"] if routes[r]["carrier"] != spot), key=lambda r: cost[r])[: rng.choice([4, 6])]
    P = {routes[r][k] for r in cheap for k in ("p_dep", "p_arr")}
    cand = []
    for f, rs in used.items():
        if f == f0:
            continue
        vp = sum(xs[r] * ((routes[r]["p_dep"] in P) + (routes[r]["p_arr"] in P)) for r in rs)
        if not vp:
            continue
        unit = sum(xs[r] * cost[r] for r in rs) / flows[f]["n"]
        alt = [cost[r] for r in flows[f]["routes"] if routes[r]["p_dep"] not in P and routes[r]["p_arr"] not in P]
        det = (min(alt) - unit) if alt else 1e9
        cand.append((det * rng.uniform(0.8, 1.25), vp, f))
    cand.sort()
    free, tot = {f0}, 0
    site_slots = []
    S = {}
    for r in xs:
        m, fl = routes[r]["carrier"], flows[routes[r]["flow"]]
        S.setdefault(m, Counter()).update({fl["t_dep"]: xs[r], fl["t_arr"]: xs[r]})
    need = [routes[r]["carrier"] for r in cheap
            if len(S.get(routes[r]["carrier"], {})) >= carriers[routes[r]["carrier"]]["sites"]
            and ({F0["t_dep"], F0["t_arr"]} - set(S[routes[r]["carrier"]]))]
    tag = f"unlock:f{f0}"
    if need:
        m = rng.choice(need)
        ends = {F0["t_dep"], F0["t_arr"]}
        light = sorted((v, t) for t, v in S[m].items() if t not in ends)[:3]
        t_old = rng.choice(light)[1]
        site_slots = [fl for fl, rs in used.items() if fl != f0 and any(
            routes[r]["carrier"] == m and t_old in (flows[fl]["t_dep"], flows[fl]["t_arr"]) for r in rs)]
        tag += f"/{carriers[m]['name'][-1]}-{t_old}"
    free |= set(site_slots[:12])
    for det, vp, f in cand:
        if len(free) >= size:
            break
        if f not in free:
            free.add(f)
            tot += vp
    if len(free) < size:  # second ring: holders of the ports the freed flows would detour to
        P2 = Counter()
        for f in free:
            for r in sorted(flows[f]["routes"], key=lambda r: cost[r])[:4]:
                for k in ("p_dep", "p_arr"):
                    if routes[r][k] not in P:
                        P2[routes[r][k]] += 1
        ring = [f for f, rs in used.items() if f not in free
                and any(routes[r][k] in P2 for r in rs for k in ("p_dep", "p_arr"))]
        ring.sort(key=lambda f: -sum(P2[routes[r][k]] for r in used[f] for k in ("p_dep", "p_arr")))
        free |= set(ring[: size - len(free)])
    return tag, free


def attribution(inst, sol, res):
    """Per-scenario cost attribution: for each scenario, cost delta vs nominal per flow (rerouting + lost reuse)."""
    routes, flows = inst["routes"], inst["flows"]
    cost = unit_costs(inst)
    tc = {(t["id"], e["port"]): e["cost"] for t in inst["sites"] for e in t["ports"]}
    xs = {e["route"]: e["n"] for e in sol["x"]}
    used = {}
    for r in xs:
        used.setdefault(routes[r]["flow"], []).append(r)
    nominal = {f: sum(xs[r] * cost[r] for r in rs) for f, rs in used.items()}
    out = {}
    for sc in inst["scenarios"]:
        failed = set(sc["routes"])
        if not failed:
            continue
        d = Counter()
        for f, rs in used.items():
            alive = [r for r in rs if r not in failed]
            if len(alive) < len(rs) and alive:
                d[f] += flows[f]["n"] * cost[alive[0]] - nominal[f]
        for e in sol.get("y", []):
            if e["from"] in failed or e["to"] in failed:
                i, j = e["from"], e["to"]
                sv = (tc[(flows[routes[i]["flow"]]["t_arr"], routes[i]["p_arr"])]
                      + tc[(flows[routes[j]["flow"]]["t_dep"], routes[j]["p_dep"])]) * e["n"]
                d[routes[i]["flow"]] += sv / 2
                d[routes[j]["flow"]] += sv / 2
        out[sc["id"]] = d
    return out


def big_or_tail(inst, sol, res, rng, mode, size, frac):
    """Structural neighbourhoods: 'big' = 20-40% of flows by blocks; 'tail' = drivers of the worst scenarios."""
    routes, flows = inst["routes"], inst["flows"]
    F = len(flows)
    sc = res["scenario_costs"]
    ranked = sorted(sc, key=sc.get, reverse=True)
    if mode == "tail":
        att = attribution(inst, sol, res)
        k = rng.choice([1, 1, 2, 3])
        tot = Counter()
        for s in ranked[:k]:
            if s in att:
                tot.update(att[s])
        top = [f for f, v in tot.most_common() if v > 0][: 2 * size]
        rng.shuffle(top)
        free = set(top[:size])
        while len(free) < size:  # pad with random flows sharing carriers/sites with the drivers
            free.add(rng.randrange(F))
        return f"tail{k}", free
    n_free = int(F * frac)
    block = rng.choice(["random", "carrier", "site", "tail"])
    used = {}
    for e in sol["x"]:
        used.setdefault(routes[e["route"]]["flow"], []).append(e["route"])
    if block == "carrier":
        ms_ = rng.sample(sorted({routes[r]["carrier"] for rs in used.values() for r in rs}), 3)
        pool = [f for f, rs in used.items() if any(routes[r]["carrier"] in ms_ for r in rs)]
    elif block == "site":
        ts = set(rng.sample(range(len(inst["sites"])), 8))
        pool = [f["id"] for f in flows if f["t_dep"] in ts or f["t_arr"] in ts]
    elif block == "tail":
        att = attribution(inst, sol, res)
        tot = Counter()
        for s in ranked[:3]:
            tot.update(att.get(s, {}))
        pool = [f for f, _ in tot.most_common(n_free)]
    else:
        pool = list(range(F))
    rng.shuffle(pool)
    free = set(pool[:n_free])
    while len(free) < n_free:
        free.add(rng.randrange(F))
    return f"big-{block}", free


LEX = {"on": False, "w": {}, "eps": 1e-3}


def install_lex(ms):
    """Runtime patch of Codex's Model.run (no file edit): objective = worst + eps * sum_s w_s * C_s, so repairs keep
    pushing the tied near-max scenarios down instead of stopping on the max plateau."""
    import numpy as np
    orig = ms.Model.run

    def run(self, deadline, seed, start):
        if not LEX["on"]:
            return orig(self, deadline, seed, start)
        zeros = np.zeros

        def lex_zeros(n, *args, **kw):
            arr = zeros(n, *args, **kw)
            if n == len(self.lower):
                tot = sum(LEX["w"].get(sid, 0.05) for sid in self.scenario_rows) or 1.0
                for sid, idx in self.scenario_rows.items():
                    w = LEX["eps"] * LEX["w"].get(sid, 0.05) / tot
                    for c, a_ in self.rows[idx][0].items():
                        if c != self.worst:
                            arr[c] += w * a_
            return arr
        ms.np.zeros = lex_zeros
        try:
            return orig(self, deadline, seed, start)
        finally:
            ms.np.zeros = zeros
    ms.Model.run = run


def install_union(ms, inst, folders):
    """Recombination: every repair may also use any route that appears (for that flow) in other validated solutions
    (union of supports). Codex's _candidates uses `warm` only as a set of must-allow routes, so we extend that set;
    the MIP start still comes from the real incumbent."""
    routes = inst["routes"]
    union, n_sol = {}, 0
    for folder in folders:
        for p_ in glob.glob(os.path.join(folder, "**", f"{inst['meta']['name']}.json"), recursive=True):
            try:
                sol_ = json.load(open(p_))
            except (OSError, ValueError):
                continue
            if not evaluate(inst, sol_)["valid"]:
                continue
            n_sol += 1
            for e in sol_["x"]:
                union.setdefault(routes[e["route"]]["flow"], set()).add(e["route"])
    orig = ms._candidates

    def cand(problem, flow, warm, *args, **kw):
        extra = union.get(flow["id"], ())
        w2 = dict(warm)
        w2.update({r: 0 for r in extra if r not in w2})
        return orig(problem, flow, w2, *args, **kw)
    ms._candidates = cand
    return n_sol, sum(len(v) for v in union.values())


def topk_sum(res, k=8):
    return sum(sorted(res["scenario_costs"].values(), reverse=True)[:k])


def kick(inst, problem, ms, sol, rng):
    """ILS perturbation: short low-temperature SA on the allocation, then global split+reuse polish."""
    import heuristic
    md = heuristic.Model(inst)
    opts = heuristic._opt_from_solution(md, sol)
    true, best = heuristic.search(md, opts, 30.0, rng.randrange(10**6), reheat=0.05)
    if best is None or true == math.inf:
        return None
    st = heuristic.State(md)
    for f, o in enumerate(best):
        st.set(f, o)
    cand = {"x": [{"route": r, "n": q} for r, q in sorted(st.x_dict().items())], "y": []}
    trial = ms._candidate(problem, cand, time.monotonic() + 60, rng.randrange(10**6),
                          free={f["id"] for f in inst["flows"]}, support_only=True, reuse=True)
    return ms.pack(*ms.unpack(trial[0])) if trial else cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--time", type=float, default=600)
    ap.add_argument("--sub", type=float, default=20, help="seconds per neighbourhood solve")
    ap.add_argument("--size", type=int, default=25, help="max free flows per neighbourhood")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kick", type=int, default=0, help="ILS: after this many consecutive failures, perturb (0=off)")
    ap.add_argument("--mode", default="alns", choices=["alns", "big", "tail", "mix", "cross"],
                    help="alns: adaptive small neighbourhoods; big: exact repair of 20-40%% of flows; "
                         "tail: drivers of the worst scenarios; mix: alternate big/tail/alns")
    ap.add_argument("--union", default="", help="comma-separated folders of solutions whose routes join the candidates")
    ap.add_argument("--kinds", default="", help="comma-separated subset of ALNS neighbourhood kinds")
    ap.add_argument("--lex", action="store_true", help="lexicographic repair: max, then tied top scenarios")
    ap.add_argument("--destroy", default="0.2-0.4", help="big mode: fraction range of flows freed")
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
    if a.lex:
        install_lex(ms)
    if a.union:
        ns, nr = install_union(ms, inst, a.union.split(","))
        print(f"union candidates: {ns} solutions, {nr} flow-route pairs", flush=True)
    rng = random.Random(a.seed)
    t_end = time.monotonic() + a.time
    stats = Counter()
    fails = 0
    while time.monotonic() + 3 < t_end:
        if a.kick and fails >= a.kick and time.monotonic() + 150 < t_end:
            kicked = kick(inst, problem, ms, sol, rng)
            rk = evaluate(inst, kicked) if kicked else None
            if rk and rk["valid"]:
                print(f"{datetime.datetime.now():%H:%M:%S} {name} KICK {res['score']} -> {rk['score']}", flush=True)
                sol, res, fails = kicked, rk, 0
                stats["kicks"] += 1
            else:
                fails = 0
        mode = a.mode if a.mode != "mix" else rng.choice(["big", "tail", "alns"])
        t0 = time.monotonic()
        if mode == "cross":  # elite pool + MILP crossover; offspring compared with both parents
            import pool as pool_mod
            if stats["cross_tried"] % 5 == 0:
                elite = pool_mod.load_pool(inst, name)
            if len(elite) < 2:
                print("pool too small for crossover", flush=True)
                break
            A, B = rng.sample(elite[: min(len(elite), 8)], 2)
            if B["score"] < A["score"]:
                A, B = B, A
            child, nfree = pool_mod.crossover(inst, problem, ms, A, B, rng, min(t_end, time.monotonic() + a.sub))
            stats["cross_tried"] += 1
            fails += 1
            if child is None:
                continue
            rc = evaluate(inst, child)
            tag = "BEATS-BOTH" if rc["score"] < A["score"] else ("beats-worse" if rc["score"] < B["score"] else "no")
            print(f"{datetime.datetime.now():%H:%M:%S} {name} XOVER {A['score']} x {B['score']} |free|={nfree} "
                  f"-> {rc['score']} {tag}", flush=True)
            if rc["valid"] and rc["score"] < res["score"]:
                kind, free, cand, r2, pc, rl = "cross", set(), child, rc, None, None
            else:
                continue
        elif mode == "alns":
            kinds = KINDS
            wts = [(stats[k + "_won"] + 1) / (stats[k + "_tried"] + 2) for k in kinds]  # adaptive (ALNS)
            if a.kinds:
                wts = [w if k in a.kinds.split(",") else 0.0 for k, w in zip(kinds, wts)]
            size = min(70, int(a.size * 1.25 ** (fails // 5)))  # grow neighbourhoods while stagnating
            kind, free = neighbourhoods(inst, sol, res, rng, rng.randint(max(5, size - 10), size + 15), wts)
            pc, rl = rng.choice([(2, 6), (2, 8), (3, 10), (3, 12), (None, None)])
            if kind.startswith(("swap", "open")):
                pc, rl = rng.choice([(3, None), (2, None)])
            elif kind.startswith("unlock"):
                pc, rl = rng.choice([(3, None), (2, None), (2, 14)])
            elif kind.startswith("bigspot"):
                pc, rl = rng.choice([(2, None), (2, 12), (3, 16)])
            elif rl is None:  # full candidate catalogue: keep the neighbourhood small
                free = set(rng.sample(sorted(free), min(len(free), 14)))
        else:
            lo, hi = map(float, a.destroy.split("-"))
            kind, free = big_or_tail(inst, sol, res, rng, mode, rng.randint(a.size, 2 * a.size), rng.uniform(lo, hi))
            pc, rl = rng.choice([(2, 6), (1, 5)]) if mode == "big" else rng.choice([(2, 6), (3, 10)])
        if mode != "cross" and not free:
            continue
        if a.lex:
            sc_ = res["scenario_costs"]
            top_ = max(sc_.values())
            LEX.update(on=True, w={s_: (1.0 if c_ >= top_ * 0.995 else 0.05) for s_, c_ in sc_.items()})
        try:
            if mode == "cross":
                raise StopIteration
            trial = ms._candidate(problem, sol, min(t_end, time.monotonic() + a.sub), rng.randrange(10**6),
                                  free=free, per_carrier=pc, route_limit=rl, reuse=True)
            cand = ms.pack(*ms.unpack(ms._best(inst, problem, sol, trial)))
        except StopIteration:
            pass
        except Exception as exc:  # solver hiccup: skip this neighbourhood
            print("neighbourhood failed:", repr(exc), file=sys.stderr)
            continue
        if mode != "cross":
            if a.lex and trial is not None:  # take the raw repair (plateau moves), not only strict improvements
                raw = ms.pack(*ms.unpack(trial[0]))
                rr = evaluate(inst, raw)
                if rr["valid"] and rr["score"] == res["score"] and topk_sum(rr) < topk_sum(res):
                    stats[kind.split(":")[0] + "_plateau"] += 1
                    print(f"{datetime.datetime.now():%H:%M:%S} {name} PLATEAU {kind} top8 {topk_sum(res)} -> "
                          f"{topk_sum(rr)}", flush=True)
                    sol, res = raw, rr
                    fails = max(0, fails - 2)
                elif rr["valid"] and rr["score"] < res["score"]:
                    cand = raw
            r2 = evaluate(inst, cand)
            if ":" in kind:  # structural scopes: log every attempt (trial None = no feasible repair in time)
                print(f"{datetime.datetime.now():%H:%M:%S} {name} try {kind} |free|={len(free)} pc={pc} "
                      f"trial={'none' if trial is None else evaluate(inst, ms.pack(*ms.unpack(trial[0])))['score']} "
                      f"inc={res['score']} ({time.monotonic() - t0:.0f}s)", flush=True)
            stats[kind.split(":")[0] + "_tried"] += 1
            fails += 1
        if r2["valid"] and r2["score"] < res["score"]:
            stats[kind.split(":")[0] + "_won"] += 1
            fails = 0
            print(f"{datetime.datetime.now():%H:%M:%S} {name} {kind:7s} |free|={len(free):2d} pc={pc} rl={rl} "
                  f"{res['score']} -> {r2['score']} ({time.monotonic() - t0:.0f}s)", flush=True)
            cur = evaluate(inst, json.load(open(path))) if os.path.exists(path) else {"score": None}
            if cur["score"] is None or r2["score"] < cur["score"]:  # another process may have improved it
                if os.path.exists(path):  # timestamped backup of the file being replaced
                    os.makedirs(os.path.join(a.out, "history"), exist_ok=True)
                    shutil.copy(path, os.path.join(a.out, "history", f"{datetime.datetime.now():%H%M%S}_{name}.json"))
                json.dump(cand, open(path, "w"))
                meta = {"when": datetime.datetime.now().isoformat(timespec="seconds"), "instance": name,
                        "score": r2["score"], "worst": worst_scenario(r2), "valid": True,
                        "pipeline": f"lns.py --mode {a.mode} ({kind}) -> codex milp_solver._candidate (@ca27544)",
                        "command": " ".join(sys.argv), "scenario_costs": r2["scenario_costs"]}
                json.dump(meta, open(os.path.join(a.out, f"{name}.meta.json"), "w"), indent=1)
            sol, res = cand, r2
    print(f"{name} final {res['score']} stats {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
