"""Claude's robust heuristic: penalised simulated annealing over per-flow route supports with exact
per-scenario bookkeeping (no reuse inside the search), then greedy reuse + reference evaluation.

solve(instance, initial_solution=None, time_limit=120.0, seed=0) -> validated official solution dict
"""
import json
import math
import random
import sys
import time
from fractions import Fraction

import numpy as np

import baseline
from evaluator import evaluate


class Model:
    def __init__(self, inst, k_pairs=40):
        self.inst = inst
        P, T, M = len(inst["ports"]), len(inst["sites"]), len(inst["carriers"])
        F, R, S = len(inst["flows"]), len(inst["routes"]), len(inst["scenarios"])
        self.P, self.T, self.M, self.F, self.R, self.S = P, T, M, F, R, S
        self.spot_m, self.spot_p = baseline.spot_ids(inst)
        tc = baseline.truck_costs(inst)
        alpha = Fraction(str(inst["meta"]["alpha"]))
        car = inst["carriers"]
        self.trans = np.array([c["trans"] for c in car], dtype=np.int64)
        self.minv = np.array([math.ceil(alpha * c["trans"]) for c in car], dtype=np.int64)
        self.minv[self.spot_m] = 0
        self.fc = np.array([c["fixed_cost"] for c in car], dtype=np.int64)
        self.maxsites = np.array([c["sites"] for c in car], dtype=np.int64)
        self.manut = np.array([p["manut"] for p in inst["ports"]], dtype=np.int64)
        fl, ro = inst["flows"], inst["routes"]
        self.n = [f["n"] for f in fl]
        self.tdep = [f["t_dep"] for f in fl]
        self.tarr = [f["t_arr"] for f in fl]
        self.rm = [r["carrier"] for r in ro]
        self.rf = [r["flow"] for r in ro]
        self.pd = [r["p_dep"] for r in ro]
        self.pa = [r["p_arr"] for r in ro]
        self.D = [2 * tc[(fl[r["flow"]]["t_dep"], r["p_dep"])] + r["cost"] + 2 * tc[(fl[r["flow"]]["t_arr"], r["p_arr"])]
                  for r in ro]
        self.fail = np.zeros((R, S), dtype=bool)
        for s, sc in enumerate(inst["scenarios"]):
            assert sc["id"] == s
            self.fail[sc["routes"], s] = True
        self.mask = [sum(1 << int(s) for s in np.nonzero(self.fail[r])[0]) for r in range(R)]
        self.nominal = [s for s in range(S) if not inst["scenarios"][s]["routes"]][0]
        self.options = [self._options(f, k_pairs) for f in range(F)]

    def _cap_ok(self, r, q):
        need_p = q * (2 if self.pd[r] == self.pa[r] else 1)
        return q <= self.trans[self.rm[r]] and need_p <= self.manut[self.pd[r]] and need_p <= self.manut[self.pa[r]]

    def _options(self, f, k_pairs):
        rs = self.inst["flows"][f]["routes"]
        n = self.n[f]
        singles = [(r, -1) for r in rs if self.mask[r] == 0 and self._cap_ok(r, n)]
        pairs = []
        if n >= 2:
            for a in range(len(rs)):
                for b in range(a + 1, len(rs)):
                    r1, r2 = rs[a], rs[b]
                    if self.mask[r1] & self.mask[r2]:
                        continue
                    if self.mask[r2] and not self._cap_ok(r1, n):
                        continue
                    if self.mask[r1] and not self._cap_ok(r2, n):
                        continue
                    if self.D[r2] < self.D[r1]:
                        r1, r2 = r2, r1  # r1 = cheaper
                    pairs.append((r1, r2))
        # heuristic pruning: keep cheapest by worst-route cost and by cheap-route cost, diverse carriers
        by_max = sorted(pairs, key=lambda p: (max(self.D[p[0]], self.D[p[1]]), self.D[p[0]]))
        by_min = sorted(pairs, key=lambda p: (self.D[p[0]], self.D[p[1]]))
        keep, seen = [], set()
        for p in by_max[:k_pairs] + by_min[:k_pairs]:
            if p not in seen:
                seen.add(p)
                keep.append(p)
        seen_c = set()
        for p in by_max:  # one per carrier combination for diversity
            key = (self.rm[p[0]], self.rm[p[1]])
            if key not in seen_c and p not in seen:
                seen_c.add(key)
                seen.add(p)
                keep.append(p)
        return singles + keep


class State:
    """Exact per-scenario loads/costs for a support assignment opt[f] = (r1, r2, k)."""

    def __init__(self, md):
        self.md = md
        S, M, P, T = md.S, md.M, md.P, md.T
        self.var = np.zeros(S, dtype=np.int64)
        self.cl = np.zeros((S, M), dtype=np.int64)
        self.pl = np.zeros((S, P), dtype=np.int64)
        self.ref = np.zeros((S, M, T), dtype=np.int64)
        self.opt = [None] * md.F

    def _route(self, r, xs, f, sign):
        md = self.md
        self.var += sign * md.D[r] * xs
        m = md.rm[r]
        self.cl[:, m] += sign * xs
        self.pl[:, md.pd[r]] += sign * xs
        self.pl[:, md.pa[r]] += sign * xs
        pos = (xs > 0).astype(np.int64) * sign
        self.ref[:, m, md.tdep[f]] += pos
        self.ref[:, m, md.tarr[f]] += pos

    def xs(self, f, o):
        md = self.md
        r1, r2, k = o
        n = md.n[f]
        f1 = md.fail[r1]
        if r2 < 0:
            return (np.where(f1, 0, n),)
        f2 = md.fail[r2]
        return np.where(f1, 0, np.where(f2, n, k)), np.where(f2, 0, np.where(f1, n, n - k))

    def apply(self, f, o, sign):
        v = self.xs(f, o)
        self._route(o[0], v[0], f, sign)
        if o[1] >= 0:
            self._route(o[1], v[1], f, sign)

    def set(self, f, o):
        if self.opt[f] is not None:
            self.apply(f, self.opt[f], -1)
        self.opt[f] = o
        self.apply(f, o, +1)

    def costs(self):
        return self.var + ((self.ref > 0) * self.md.fc[None, :, None]).sum(axis=(1, 2))

    def violation(self):
        md = self.md
        v = np.maximum(self.pl - md.manut[None, :], 0).sum()
        v += np.maximum(self.cl - md.trans[None, :], 0).sum()
        v += np.where((self.cl > 0) & (self.cl < md.minv[None, :]), md.minv[None, :] - self.cl, 0).sum()
        refn = self.ref[md.nominal]
        nsites = (refn > 0).sum(axis=1)
        for m in np.nonzero(nsites > md.maxsites)[0]:  # graded: smallest site supports must be removed
            pos = np.sort(refn[m][refn[m] > 0])
            v += 50 * int(pos[: nsites[m] - md.maxsites[m]].sum())
        return int(v)

    def uses(self, f, m):
        o = self.opt[f]
        return self.md.rm[o[0]] == m or (o[1] >= 0 and self.md.rm[o[1]] == m)

    def x_dict(self):
        x = {}
        for f, (r1, r2, k) in enumerate(self.opt):
            if r2 < 0:
                x[r1] = self.md.n[f]
            else:
                x[r1], x[r2] = k, self.md.n[f] - k
        return x


def _opt_from_solution(md, sol):
    by_flow = {}
    for e in sol["x"]:
        by_flow.setdefault(md.rf[e["route"]], []).append((e["route"], e["n"]))
    opts = []
    for f in range(md.F):
        rs = by_flow[f]
        if len(rs) == 1:
            o = (rs[0][0], -1, md.n[f])
        else:
            (a, xa), (b, xb) = sorted(rs, key=lambda t: md.D[t[0]])
            o = (a, b, xa)
        if (o[0], o[1]) not in md.options[f]:
            md.options[f].append((o[0], o[1]))
        opts.append(o)
    return opts


def _split_choices(md, f, r1, r2, rng):
    n = md.n[f]
    return [n - 1, 1, n // 2, rng.randint(1, n - 1)]


def finalize(md, x, rng=None):
    """Greedy reuse weighted towards the currently critical scenarios, then reference evaluation."""
    inst = md.inst
    plain = baseline.to_solution(x, [])
    res0 = evaluate(inst, plain)
    if not res0["valid"]:
        return plain, res0
    sc = res0["scenario_costs"]
    top = max(sc.values())
    best_sol, best_res = plain, res0
    for thr, mult in ((None, 0), (0.01, 10), (0.003, 10), (0.03, 10), (0.01, 100), (0.05, 3)):
        w = None
        if thr is not None:
            crit = np.array([s for s, c in sc.items() if c >= top - thr * abs(top)], dtype=int)
            w = (lambda i, j, crit=crit, mult=mult: 1 + mult * int((~(md.fail[i, crit] | md.fail[j, crit])).sum()))
        sol = baseline.to_solution(x, baseline.greedy_reuse(inst, x, w))
        res = evaluate(inst, sol)
        if res["valid"] and res["score"] < best_res["score"]:
            best_sol, best_res = sol, res
    try:  # optional: Codex's exact reuse optimiser (feat/optimization), used only if importable
        import reuse as codex_reuse
        sol = codex_reuse.optimize_reuse(inst, best_sol, time_limit=20.0, seed=0)
        res = evaluate(inst, sol)
        if res["valid"] and res["score"] < best_res["score"]:
            best_sol, best_res = sol, res
    except Exception as exc:  # ImportError or solver failure: keep greedy reuse
        print(f"codex reuse unavailable/failed: {exc!r}", file=sys.stderr)
    return best_sol, best_res


def search(md, opts, time_limit, seed, log=False, reheat=1.0):
    rng = random.Random(seed)
    st = State(md)
    for f, o in enumerate(opts):
        st.set(f, o)
    eps, lam, LMIN, LMAX = 0.02, 1000.0, 20.0, 1e6

    def obj():
        c = st.costs()
        return float(c.max() + eps * c.mean()), st.violation(), int(c.max())

    def random_single(f):
        old = st.opt[f]
        if rng.random() < 0.7 or old[1] < 0:
            r1, r2 = rng.choice(md.options[f])
            k = md.n[f] if r2 < 0 else rng.choice(_split_choices(md, f, r1, r2, rng))
        else:
            r1, r2 = old[0], old[1]
            k = max(1, min(md.n[f] - 1, old[2] + rng.choice([-1, 1]) * rng.randint(1, max(1, md.n[f] // 4))))
        return (r1, r2, k)

    def evacuate(m, t):
        """Compound move: greedily move every flow using carrier m (at site t if given) elsewhere."""
        fl = [f for f in range(md.F) if st.uses(f, m) and (t is None or t in (md.tdep[f], md.tarr[f]))]
        if not fl or len(fl) > 40:
            return None
        olds = [(f, st.opt[f]) for f in fl]
        rng.shuffle(fl)
        for f in fl:
            cands = [o for o in md.options[f] if md.rm[o[0]] != m and (o[1] < 0 or md.rm[o[1]] != m)]
            if not cands:
                continue
            bestc = None
            for r1, r2 in rng.sample(cands, min(8, len(cands))):
                o = (r1, r2, md.n[f] if r2 < 0 else md.n[f] - 1)
                st.set(f, o)
                Jc, Vc, _ = obj()
                if bestc is None or Jc + lam * Vc < bestc[0]:
                    bestc = (Jc + lam * Vc, o)
            st.set(f, bestc[1])
        return olds

    def bundle_open(m):
        """Compound move: open carrier m by moving the most promising flows onto options using it."""
        cand = []
        for f in range(md.F):
            if st.uses(f, m):
                continue
            os_ = [o for o in md.options[f] if md.rm[o[0]] == m or (o[1] >= 0 and md.rm[o[1]] == m)]
            if os_:
                cand.append((min(md.D[o[0]] for o in os_) - md.D[st.opt[f][0]] + rng.random() * 500, f, os_))
        if not cand:
            return None
        cand.sort()
        olds, load = [], 0
        for _, f, os_ in cand:
            if load >= 1.2 * md.minv[m] or len(olds) >= 25:
                break
            r1, r2 = min(rng.sample(os_, min(4, len(os_))), key=lambda o: md.D[o[0]] + md.D[o[1]] if o[1] >= 0 else 2 * md.D[o[0]])
            olds.append((f, st.opt[f]))
            st.set(f, (r1, r2, md.n[f] if r2 < 0 else md.n[f] - 1))
            load = int(st.cl[:, m][st.cl[:, m] > 0].min()) if (st.cl[:, m] > 0).any() else 0
        return olds

    def worst_flows():
        sw = int(np.argmax(st.costs()))
        return [f for f in range(md.F) if md.fail[st.opt[f][0], sw] or (st.opt[f][1] >= 0 and md.fail[st.opt[f][1], sw])]

    J, V, true = obj()
    best = (true if V == 0 else math.inf, list(st.opt))
    wf = worst_flows()
    # temperature calibration from sampled uphill deltas (target ~50% acceptance at start)
    ups = []
    for _ in range(300):
        f = rng.randrange(md.F)
        old, new = st.opt[f], random_single(f)
        st.set(f, new)
        J2, V2, _ = obj()
        d = (J2 + lam * V2) - (J + lam * V)
        if d > 0:
            ups.append(d)
        st.set(f, old)
    T0 = reheat * (float(np.median(ups)) / math.log(2) if ups else 1e5)
    T1 = T0 * 1e-4 / reheat
    t0 = time.monotonic()
    it = last_feas = last_best = 0
    while True:
        el = time.monotonic() - t0
        if el > time_limit:
            break
        frac = el / time_limit
        temp = T0 * (T1 / T0) ** frac
        it += 1
        u = rng.random()
        if u < 0.001:
            closed = [m for m in range(md.M) if m != md.spot_m and st.cl[md.nominal, m] == 0]
            if not closed:
                continue
            olds = bundle_open(rng.choice(closed))
            if olds is None:
                continue
            undo = lambda olds=olds: [st.set(f, o) for f, o in reversed(olds)]  # noqa: E731
        elif u < 0.005:
            open_c = [m for m in range(md.M) if m != md.spot_m and st.cl[md.nominal, m] > 0]
            if not open_c:
                continue
            m = rng.choice(open_c)
            t = None
            if rng.random() < 0.7:
                refm = st.ref[md.nominal, m]
                ts = np.nonzero(refm > 0)[0]
                t = int(ts[np.argmin(refm[ts])]) if rng.random() < 0.5 else int(rng.choice(list(ts)))
            olds = evacuate(m, t)
            if olds is None:
                continue
            undo = lambda olds=olds: [st.set(f, o) for f, o in olds]  # noqa: E731
        else:
            f = rng.choice(wf) if wf and rng.random() < 0.5 else rng.randrange(md.F)
            old, new = st.opt[f], random_single(f)
            if new == old:
                continue
            st.set(f, new)
            undo = lambda f=f, old=old: st.set(f, old)  # noqa: E731
        J2, V2, true2 = obj()
        d = (J2 + lam * V2) - (J + lam * V)
        if d <= 0 or rng.random() < math.exp(-d / temp):
            J, V = J2, V2
            if V == 0:
                last_feas = it
                if true2 < best[0]:
                    best = (true2, list(st.opt))
                    last_best = it
        else:
            undo()
        if it % 1000 == 0:
            wf = worst_flows()
            if frac > 0.9:
                lam = LMAX
            else:
                lam = min(LMAX, lam * 1.3) if it - last_feas > 1000 else max(LMIN, lam / 1.3)
            J, V, _ = obj()
            if best[0] < math.inf and it - last_best > 30000 and it - last_feas > 10000:
                for f, o in enumerate(best[1]):
                    st.set(f, o)
                J, V, _ = obj()
                last_best = last_feas = it
            if log and it % 10000 == 0:
                print(f"it={it} t={el:.0f}s T={temp:.3g} J={J:.4e} V={V} lam={lam:.0f} best={best[0]}",
                      file=sys.stderr)
    return best


def solve(instance, initial_solution=None, time_limit=120.0, seed=0, reheat=None):
    t_start = time.monotonic()
    md = Model(instance)
    fallback = initial_solution if initial_solution is not None else baseline.solve(instance)
    fres = evaluate(instance, fallback)
    if not fres["valid"]:
        fallback = baseline.solve(instance)
        fres = evaluate(instance, fallback)
    opts = _opt_from_solution(md, fallback)
    budget = max(1.0, time_limit - (time.monotonic() - t_start) - 5.0)
    spot_share = sum(e["n"] for e in fallback["x"] if md.rm[e["route"]] == md.spot_m) / sum(md.n)
    if reheat is None:
        reheat = 1.0 if spot_share > 0.9 else 0.03  # warm start from a good solution: do not destroy it
    true, best_opts = search(md, opts, budget, seed, log=True, reheat=reheat)
    if best_opts is None or true == math.inf:
        return fallback
    st = State(md)
    for f, o in enumerate(best_opts):
        st.set(f, o)
    sol, res = finalize(md, st.x_dict())
    if res["valid"] and res["score"] < fres["score"]:
        return sol
    return fallback


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    sol = solve(inst, time_limit=tl)
    r = evaluate(inst, sol)
    print("valid", r["valid"], "score", r["score"])
