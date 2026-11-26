"""Finite-scenario route allocation with exact automatic rerouting."""

from collections import defaultdict
from itertools import combinations, product
import logging
from math import inf
from time import monotonic

import highspy
import numpy as np

from common import Problem, pack, unpack


log = logging.getLogger(__name__)


class Model:
    def __init__(self):
        self.lower, self.upper, self.integer = [], [], []
        self.rows, self.cost_rows = [], []
        self.worst = self.var()
        self.infeasible = False

    def var(self, lo=0, hi=inf, integer=False):
        index = len(self.lower)
        self.lower.append(lo)
        self.upper.append(hi)
        if integer:
            self.integer.append(index)
        return index

    def row(self, terms, lo=-inf, hi=inf):
        coefficients = defaultdict(int)
        for column, amount in terms:
            coefficients[column] += amount
        offset = coefficients.pop(None, 0)
        coefficients = {c: a for c, a in coefficients.items() if a}
        if coefficients:
            self.rows.append((coefficients, lo - offset, hi - offset))
        elif not lo <= offset <= hi:
            self.infeasible = True

    def cost(self, terms):
        self.row([*terms, (self.worst, -1)], hi=0)
        self.cost_rows.append(len(self.rows) - 1)

    def check(self, point):
        """Check the extracted integer point and recalculate the epigraph in USD."""
        if self.infeasible:
            return None
        point[self.worst] = max(0, *(
            sum(a * point[c] for c, a in row.items() if c != self.worst) - hi
            for row, _, hi in (self.rows[i] for i in self.cost_rows)
        ))
        if any(not lo <= n <= hi for n, lo, hi in zip(point, self.lower, self.upper)):
            return None
        for row, lo, hi in self.rows:
            if not lo <= sum(a * point[c] for c, a in row.items()) <= hi:
                return None
        return point[self.worst]

    def run(self, deadline, seed, start):
        if self.infeasible or deadline <= monotonic():
            return None
        solver = highspy.Highs()
        solver.setOptionValue("output_flag", log.isEnabledFor(logging.DEBUG))
        solver.setOptionValue("random_seed", seed)
        solver.setOptionValue("threads", 1)
        solver.setOptionValue("mip_rel_gap", 0.0)
        solver.setOptionValue("mip_abs_gap", 0.1)
        solver.setOptionValue("mip_feasibility_tolerance", 1e-7)
        solver.setOptionValue("mip_heuristic_effort", 0.2)

        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = len(self.lower), len(self.rows)
        costs = np.zeros(lp.num_col_)
        costs[self.worst] = 1
        lp.col_cost_ = costs
        lp.col_lower_, lp.col_upper_ = self.lower, self.upper
        lp.row_lower_ = [lo for _, lo, _ in self.rows]
        lp.row_upper_ = [hi for _, _, hi in self.rows]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        pointers = [0]
        for row, _, _ in self.rows:
            pointers.append(pointers[-1] + len(row))
        lp.a_matrix_.start_ = pointers
        lp.a_matrix_.index_ = np.fromiter(
            (c for row, _, _ in self.rows for c in row), np.int32, pointers[-1]
        )
        lp.a_matrix_.value_ = np.fromiter(
            (a for row, _, _ in self.rows for a in row.values()), np.float64, pointers[-1]
        )
        integrality = [highspy.HighsVarType.kContinuous] * lp.num_col_
        for c in self.integer:
            integrality[c] = highspy.HighsVarType.kInteger
        lp.integrality_ = integrality
        if solver.passModel(lp) == highspy.HighsStatus.kError:
            raise RuntimeError("HiGHS could not load the allocation model")
        if start:
            solver.setSolution(len(start), list(start), list(start.values()))
        remaining = deadline - monotonic()
        if remaining <= 0:
            return None
        solver.setOptionValue("time_limit", remaining)
        solver.run()
        info = solver.getInfo()
        log.info("Model %s rows, %s columns, %s nonzeros | %s | %.2fs | objective %.2f | model bound %.2f",
                 lp.num_row_, lp.num_col_, pointers[-1], solver.getModelStatus().name,
                 solver.getRunTime(), info.objective_function_value, info.mip_dual_bound)
        result = solver.getSolution()
        if not result.value_valid:
            return None
        point = [round(v) for v in result.col_value]
        score = self.check(point)
        if score is None:
            log.warning("HiGHS incumbent failed the exact integer row check")
            return None
        return point, score


def _candidates(problem, flow, warm, per_carrier, route_limit, safe_only):
    """Retain incumbent routes, remove exact duplicates, then optionally prune."""
    groups = defaultdict(list)
    seen = set()
    demand = flow["n"]
    for r in sorted(flow["routes"], key=lambda r: (problem.cost[r], r)):
        route = problem.routes[r]
        carrier = route["carrier"]
        signature = carrier, route["p_dep"], route["p_arr"], problem.mask[r]
        if safe_only and problem.mask[r] and r not in warm:
            continue
        if problem.capacity[r] <= 0 or (signature in seen and r not in warm):
            continue
        seen.add(signature)
        groups[carrier].append(r)
    allowed = set(warm)
    for carrier, routes in groups.items():
        if per_carrier is None or carrier == problem.spot:
            allowed.update(routes)
        else:
            allowed.update(routes[:per_carrier])
            safe = next((r for r in routes if not problem.mask[r]), None)
            if safe is not None:
                allowed.add(safe)
    if route_limit is not None:
        spot = {r for r in allowed if problem.routes[r]["carrier"] == problem.spot}
        regular = allowed - spot
        ranked = sorted(regular, key=lambda r: (
            problem.capacity[r] < demand, problem.cost[r], problem.mask[r].bit_count(), r
        ))
        allowed = spot | set(ranked[:route_limit]) | set(warm)

    viable = {r for r in allowed if not problem.mask[r] and problem.capacity[r] >= demand}
    if demand >= 2:
        for i, j in combinations(sorted(allowed), 2):
            a, b = problem.mask[i], problem.mask[j]
            ci, cj = problem.capacity[i], problem.capacity[j]
            if not a & b and ci + cj >= demand and (not b or ci >= demand) and (not a or cj >= demand):
                viable.update((i, j))
    return sorted(viable | set(warm))


def _allocation(model, problem, incumbent, free, support_only, per_carrier, route_limit, safe_only):
    by_flow = defaultdict(dict)
    for r, n in incumbent.items():
        by_flow[problem.routes[r]["flow"]][r] = n
    nominal, selected, starts = {}, {}, {}
    realized = [dict() for _ in problem.scenarios]
    for flow in problem.flows:
        f, demand = flow["id"], flow["n"]
        warm = by_flow[f]
        frozen = f not in free
        if not demand and not warm:
            continue
        routes = sorted(warm) if frozen or support_only else _candidates(
            problem, flow, warm, per_carrier, route_limit, safe_only
        )
        x = {}
        for r in routes:
            hi = min(demand, problem.capacity[r])
            if frozen:
                x[r] = (None, warm[r])
            else:
                c = model.var(lo=int(support_only), hi=hi, integer=True)
                x[r] = (c, 1)
                starts[c] = warm.get(r, 0)
                if not support_only:
                    selected[r] = z = model.var(hi=1, integer=True)
                    starts[z] = int(r in warm)
                    model.row([(c, 1), (z, -hi)], hi=0)
                    model.row([(c, 1), (z, -1)], lo=0)
        nominal.update(x)
        model.row(x.values(), lo=demand, hi=demand)
        if frozen or support_only:
            if len(routes) > 2:
                model.infeasible = True
                continue
        else:
            model.row([(selected[r], 1) for r in routes], hi=2)

        cases = {tuple(routes): x}
        for s, failed in enumerate(problem.failed):
            surviving = tuple(r for r in routes if r not in failed)
            if surviving not in cases:
                rerouted = {}
                if not surviving:
                    model.infeasible = True
                elif frozen or support_only:
                    r, = surviving
                    rerouted[r] = (None, demand)
                else:
                    model.row([(selected[r], 1) for r in surviving], lo=1)
                    gone = set(routes) & failed
                    if len(gone) > 1:
                        model.row([(selected[r], 1) for r in gone], hi=1)
                    if len(surviving) == 1:
                        rerouted[surviving[0]] = (None, demand)
                    else:
                        for r in surviving:
                            hi = min(demand, problem.capacity[r])
                            c = model.var(hi=hi)
                            rerouted[r] = (c, 1)
                            model.row([(c, 1), (x[r][0], -1)], lo=0)
                            model.row([(c, 1), (selected[r], -hi)], hi=0)
                        model.row(rerouted.values(), lo=demand, hi=demand)
                cases[surviving] = rerouted
            realized[s].update(cases[surviving])
    return nominal, selected, realized, starts


def _reuse_arcs(model, problem, nominal):
    inbound, outbound = defaultdict(list), defaultdict(list)
    for r in nominal:
        i, j = problem.ends[r]
        m = problem.routes[r]["carrier"]
        inbound[m, j].append(r)
        outbound[m, i].append(r)
    by_from, by_to, by_site = defaultdict(list), defaultdict(list), defaultdict(list)
    arcs = {}
    for (carrier, site), arriving in inbound.items():
        cap = problem.sites[site]["reutil"]
        if not cap:
            continue
        for i, j in product(arriving, outbound[carrier, site]):
            savings = problem.truck_in[i] + problem.truck_out[j]
            ci, qi = nominal[i]
            cj, qj = nominal[j]
            hi = min(cap, qi if ci is None else model.upper[ci], qj if cj is None else model.upper[cj])
            column = model.var(hi=hi, integer=True)
            arcs[i, j] = (column, savings, problem.mask[i] | problem.mask[j])
            by_from[i].append((column, 1))
            by_to[j].append((column, 1))
            by_site[site].append((column, 1))
    for side in (by_from, by_to):
        for r, terms in side.items():
            column, amount = nominal[r]
            model.row([*terms, (column, -amount)], hi=0)
    for site, terms in by_site.items():
        model.row(terms, hi=problem.sites[site]["reutil"])
    return arcs


def _build(problem, x, free, support_only, per_carrier, reuse, route_limit=None, safe_only=False):
    model = Model()
    nominal, selected, realized, starts = _allocation(
        model, problem, x, free, support_only, per_carrier, route_limit, safe_only
    )
    arcs = _reuse_arcs(model, problem, nominal) if reuse else {}
    activations = {}
    nominal_carriers = defaultdict(int)
    for r in nominal:
        nominal_carriers[problem.routes[r]["carrier"]] += 1

    def active(routes):
        # A fixed positive route makes the entire OR constant.
        if any(r not in selected for r in routes):
            return None, 1
        columns = tuple(sorted(selected[r] for r in routes))
        if len(columns) == 1:
            return columns[0], 1
        if columns not in activations:
            e = model.var(hi=1)
            for z in columns:
                model.row([(e, 1), (z, -1)], lo=0)
            model.row([(e, -1), *((z, 1) for z in columns)], lo=0)
            activations[columns] = e
        return activations[columns], 1

    seen = set()
    for s in [problem.nominal, *(i for i in range(len(realized)) if i != problem.nominal)]:
        affected = problem.failed[s].intersection(nominal)
        if affected in seen:
            continue
        seen.add(affected)
        quantities = realized[s]
        ports, carriers, contacts, costs = defaultdict(list), defaultdict(list), defaultdict(list), []
        for r, term in quantities.items():
            route = problem.routes[r]
            carrier = route["carrier"]
            ports[route["p_dep"]].append(term)
            ports[route["p_arr"]].append(term)
            carriers[carrier].append(r)
            for site in set(problem.ends[r]):
                contacts[carrier, site].append(r)
            c, a = term
            costs.append((c, a * problem.cost[r]))
        for p, terms in ports.items():
            model.row(terms, hi=problem.ports[p]["manut"])
        by_carrier = defaultdict(list)
        for (m, site), routes in contacts.items():
            fee = problem.carriers[m]["fixed_cost"]
            if fee or s == problem.nominal:
                c, a = active(routes)
                by_carrier[m].append((c, a))
                if fee:
                    costs.append((c, a * fee))
        for m, routes in carriers.items():
            carrier = problem.carriers[m]
            volume = [quantities[r] for r in routes]
            model.row(volume, hi=carrier["trans"])
            if s == problem.nominal:
                model.row(by_carrier[m], hi=carrier["sites"])
            if problem.minimum[m]:
                # A carrier with no failed candidates only gains volume since nominal.
                if s == problem.nominal or len(routes) != nominal_carriers[m]:
                    c, a = active(routes)
                    model.row([*volume, (c, -a * problem.minimum[m])], lo=0)
        costs.extend(
            (c, -saving) for c, saving, mask in arcs.values() if not mask & (1 << s)
        )
        model.cost(costs)
    return model, nominal, arcs, starts


def _score(problem, solution):
    try:
        x, y = unpack(solution)
    except (KeyError, TypeError, ValueError):
        return None
    if any(r >= len(problem.routes) for r in x):
        return None
    if sum(x.values()) != sum(f["n"] for f in problem.flows):
        return None
    model, _, arcs, _ = _build(problem, x, set(), False, None, bool(y))
    if any(pair not in arcs for pair in y):
        return None
    point = [0] * len(model.lower)
    for pair, n in y.items():
        point[arcs[pair][0]] = n
    return model.check(point)


def _candidate(problem, initial, deadline, seed, *, free, support_only=False,
               per_carrier=None, reuse=False, route_limit=None, safe_only=False):
    x, y = unpack(initial) if initial is not None else ({}, {})
    model, nominal, arcs, start = _build(
        problem, x, free, support_only, per_carrier, reuse, route_limit, safe_only
    )
    for pair, (c, _, _) in arcs.items():
        start[c] = y.get(pair, 0)
    result = model.run(deadline, seed, start if initial is not None else {})
    if result is None:
        return None
    point, score = result
    return pack(
        {r: n if c is None else point[c] for r, (c, n) in nominal.items()},
        {pair: point[c] for pair, (c, _, _) in arcs.items()},
    ), score


def _independent_score(instance, problem, solution):
    try:
        from evaluator import evaluate
    except ModuleNotFoundError as error:
        if error.name != "evaluator":
            raise
        return _score(problem, solution)
    result = evaluate(instance, solution)
    return result["score"] if result["valid"] else None


def _best(instance, problem, incumbent, trial):
    if trial is None:
        return incumbent
    candidate, model_score = trial
    score = _independent_score(instance, problem, candidate)
    if score != model_score:
        log.warning("Discarded allocation: model score %s, checked score %s", model_score, score)
        return incumbent
    old_score = _independent_score(instance, problem, incumbent) if incumbent is not None else None
    return candidate if old_score is None or score < old_score else incumbent


def _neighborhood(instance, problem, initial_solution, free, deadline, seed):
    if _independent_score(instance, problem, initial_solution) is None:
        raise ValueError("Neighborhood optimization requires a feasible incumbent")
    trial = _candidate(problem, initial_solution, deadline, seed, free=free,
                       per_carrier=2, reuse=True)
    return pack(*unpack(_best(instance, problem, initial_solution, trial)))


def solve(instance, initial_solution=None, time_limit=120.0, seed=0):
    """Return the best checked allocation found within the wall-clock budget."""
    start = monotonic()
    problem = Problem(instance)
    free = {f["id"] for f in problem.flows}
    best = initial_solution
    if best is None or _independent_score(instance, problem, best) is None:
        best = problem.spot_solution()
        if best is not None and _independent_score(instance, problem, best) is None:
            best = None
    per_carrier = 1 if len(problem.routes) > 500 else None
    trial = _candidate(problem, best, start + 0.75 * time_limit, seed,
                       free=free, per_carrier=per_carrier)
    if trial is not None:
        improved = _candidate(problem, trial[0], start + time_limit, seed,
                              free=free, support_only=True, reuse=True)
        best = _best(instance, problem, best, trial)
        best = _best(instance, problem, best, improved)
    if best is None:
        raise RuntimeError("No feasible allocation was found within the time limit")
    return pack(*unpack(best))
