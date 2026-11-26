"""Robust integer container reuse for an existing allocation."""

from collections import defaultdict, deque
from itertools import product
from time import monotonic

from common import Problem, pack, unpack


class ReuseArcs:
    def __init__(self, model, problem, nominal, compress):
        # Routes in a group have identical scenario survival and truck savings
        # on this side of the reuse transaction.
        self.groups = [defaultdict(list), defaultdict(list)]
        self.route_group = [{}, {}]
        by_site = [defaultdict(set), defaultdict(set)]
        for r in sorted(nominal):
            origin, dest = problem.ends[r]
            carrier = problem.routes[r]["carrier"]
            for side, site, saving in (
                (0, dest, problem.truck_in[r]), (1, origin, problem.truck_out[r])
            ):
                group = (carrier, site, problem.mask[r], saving, -1 if compress else r)
                self.groups[side][group].append(r)
                self.route_group[side][r] = group
                by_site[side][carrier, site].add(group)

        bounds = [{
            group: sum(n if c is None else model.upper[c] for c, n in
                       (nominal[r] for r in routes))
            for group, routes in groups.items()
        } for groups in self.groups]
        limits = [defaultdict(list), defaultdict(list)]
        sites = defaultdict(list)
        self.arcs = {}
        for (carrier, site), arriving in sorted(by_site[0].items()):
            cap = problem.sites[site]["reutil"]
            if not cap:
                continue
            for a, b in product(sorted(arriving), sorted(by_site[1][carrier, site])):
                c = model.var(hi=min(cap, bounds[0][a], bounds[1][b]), integer=True)
                self.arcs[a, b] = c, a[3] + b[3], a[2] | b[2]
                limits[0][a].append((c, 1))
                limits[1][b].append((c, 1))
                sites[site].append((c, 1))
        for side, groups in enumerate(limits):
            for group, terms in groups.items():
                availability = [(c, -n) for c, n in
                                (nominal[r] for r in self.groups[side][group])]
                model.row([*terms, *availability], hi=0)
        for site, terms in sites.items():
            model.row(terms, hi=problem.sites[site]["reutil"])

    def start(self, reuse):
        point = {c: 0 for c, _, _ in self.arcs.values()}
        for (i, j), n in reuse.items():
            pair = self.route_group[0].get(i), self.route_group[1].get(j)
            if pair not in self.arcs:
                return None
            point[self.arcs[pair][0]] += n
        return point

    def extract(self, point, quantities):
        """Disaggregate integral group totals over complete compatible blocks."""
        remaining = [quantities.copy(), quantities.copy()]
        available = [{
            group: deque(r for r in routes if quantities[r] > 0)
            for group, routes in groups.items()
        } for groups in self.groups]
        reuse = {}
        for (a, b), (c, _, _) in self.arcs.items():
            amount = point[c]
            left, right = available[0][a], available[1][b]
            while amount:
                i, j = left[0], right[0]
                n = min(amount, remaining[0][i], remaining[1][j])
                reuse[i, j] = n
                amount -= n
                remaining[0][i] -= n
                remaining[1][j] -= n
                if not remaining[0][i]:
                    left.popleft()
                if not remaining[1][j]:
                    right.popleft()
        return reuse


def optimize_reuse(instance, solution, time_limit=20.0, seed=0):
    """Keep all route quantities fixed and return a checked, nonworse solution."""
    from milp_solver import _best, _candidate, _independent_score

    deadline = monotonic() + time_limit
    problem = Problem(instance)
    x, _ = unpack(solution)
    if _independent_score(instance, problem, solution) is None:
        solution = pack(x, {})
        if _independent_score(instance, problem, solution) is None:
            raise ValueError("Reuse optimization requires a feasible route allocation")
    trial = _candidate(problem, solution, deadline, seed, free=set(), reuse=True)
    return pack(*unpack(_best(instance, problem, solution, trial)))
