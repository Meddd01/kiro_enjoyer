"""Finite-scenario route allocation with exact automatic rerouting."""

from collections import defaultdict
from itertools import combinations
import logging
from math import inf
from time import monotonic

import highspy
import numpy as np

from common import Problem, pack, unpack
from reuse import ReuseArcs


log = logging.getLogger(__name__)


class Model:
    def __init__(self):
        self.lower, self.upper, self.integer = [], [], []
        self.rows, self.cost_rows = [], []
        self.scenario_rows = {}
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
        index = len(self.rows) - 1
        self.cost_rows.append(index)
        return index

    def scenario_cost(self, point, index):
        row, _, hi = self.rows[index]
        return sum(a * point[c] for c, a in row.items() if c != self.worst) - hi

    def check(self, point):
        """Check the extracted integer point and recalculate the epigraph in USD."""
        if self.infeasible:
            return None
        point[self.worst] = max(0, *(self.scenario_cost(point, i) for i in self.cost_rows))
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
            log.warning("HiGHS could not load the allocation model")
            return None
        if start:
            point = [start.get(c, 0) for c in range(lp.num_col_)]
            if self.check(point) is not None:
                solver.setSolution(lp.num_col_, list(range(lp.num_col_)), point)
        remaining = deadline - monotonic()
        if remaining <= 0:
            return None
        solver.setOptionValue("time_limit", remaining)
        if solver.run() == highspy.HighsStatus.kError:
            log.warning("HiGHS failed while solving the allocation model")
            return None
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

    singletons = {r for r in allowed if not problem.mask[r] and problem.capacity[r] >= demand}
    partners = defaultdict(list)
    if demand >= 2:
        for i, j in combinations(sorted(allowed), 2):
            a, b = problem.mask[i], problem.mask[j]
            ci, cj = problem.capacity[i], problem.capacity[j]
            if not a & b and ci + cj >= demand and (not b or ci >= demand) and (not a or cj >= demand):
                partners[i].append(j)
                partners[j].append(i)
    return sorted(singletons | partners.keys() | warm.keys()), {
        r: mates for r, mates in partners.items() if r not in singletons
    }


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
        if frozen or support_only:
            routes, partners = sorted(warm), {}
        else:
            routes, partners = _candidates(problem, flow, warm, per_carrier, route_limit, safe_only)
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
            for r, mates in partners.items():
                model.row([(selected[r], 1), *((selected[j], -1) for j in mates)], hi=0)

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
                        warm_alive = set(warm).intersection(surviving)
                        for r in surviving:
                            hi = min(demand, problem.capacity[r])
                            c = model.var(hi=hi)
                            rerouted[r] = (c, 1)
                            starts[c] = demand if warm_alive == {r} else warm.get(r, 0)
                            model.row([(c, 1), (x[r][0], -1)], lo=0)
                            model.row([(c, 1), (selected[r], -hi)], hi=0)
                        model.row(rerouted.values(), lo=demand, hi=demand)
                cases[surviving] = rerouted
            realized[s].update(cases[surviving])
    return nominal, selected, realized, starts


def _build(problem, x, free, support_only, per_carrier, reuse, route_limit=None,
           safe_only=False, compress_reuse=True):
    model = Model()
    nominal, selected, realized, starts = _allocation(
        model, problem, x, free, support_only, per_carrier, route_limit, safe_only
    )
    reuse_plan = ReuseArcs(model, problem, nominal, compress_reuse) if reuse else None
    arcs = reuse_plan.arcs.values() if reuse_plan is not None else ()
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
            starts[e] = max(starts[z] for z in columns)
        return activations[columns], 1

    seen = {}
    for s in [problem.nominal, *(i for i in range(len(realized)) if i != problem.nominal)]:
        affected = problem.failed[s].intersection(nominal)
        scenario_id = problem.scenarios[s]["id"]
        if affected in seen:
            model.scenario_rows[scenario_id] = seen[affected]
            continue
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
            (c, -saving) for c, saving, mask in arcs if not mask & (1 << s)
        )
        model.scenario_rows[scenario_id] = seen[affected] = model.cost(costs)
    return model, nominal, reuse_plan, starts


def _scenario_costs(problem, solution):
    try:
        x, y = unpack(solution)
    except (KeyError, TypeError, ValueError):
        return None
    if any(r >= len(problem.routes) for r in x):
        return None
    if sum(x.values()) != sum(f["n"] for f in problem.flows):
        return None
    model, _, reuse_plan, _ = _build(
        problem, x, set(), False, None, bool(y), compress_reuse=False
    )
    point = [0] * len(model.lower)
    if reuse_plan is not None:
        values = reuse_plan.start(y)
        if values is None:
            return None
        for c, n in values.items():
            point[c] = n
    if model.check(point) is None:
        return None
    return {s: model.scenario_cost(point, i) for s, i in model.scenario_rows.items()}


def _score(problem, solution):
    costs = _scenario_costs(problem, solution)
    return max(costs.values()) if costs is not None else None


def _candidate(problem, initial, deadline, seed, *, free, support_only=False,
               per_carrier=None, reuse=False, route_limit=None, safe_only=False,
               compress_reuse=True):
    if deadline <= monotonic():
        return None
    x, y = unpack(initial) if initial is not None else ({}, {})
    model, nominal, reuse_plan, start = _build(
        problem, x, free, support_only, per_carrier, reuse, route_limit, safe_only,
        compress_reuse,
    )
    if reuse_plan is not None:
        values = reuse_plan.start(y)
        if values is not None:
            start.update(values)
    result = model.run(deadline, seed, start if initial is not None else {})
    if result is None:
        return None
    point, score = result
    x = {r: n if c is None else point[c] for r, (c, n) in nominal.items()}
    y = reuse_plan.extract(point, x) if reuse_plan is not None else {}
    return pack(x, y), score


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
    if initial_solution is None or _independent_score(instance, problem, initial_solution) is None:
        raise ValueError("Neighborhood optimization requires a feasible incumbent")
    if not free <= {f["id"] for f in problem.flows}:
        raise ValueError("Neighborhood includes unknown flow IDs")
    trial = _candidate(problem, initial_solution, deadline, seed, free=free,
                       per_carrier=2, route_limit=6, reuse=True)
    return pack(*unpack(_best(instance, problem, initial_solution, trial)))


def _construct(problem, incumbent, deadline, seed, safe_only):
    """Seed the robust model by freeing blocks of high-cost flows first."""
    ranked = sorted(problem.flows, key=lambda f: (
        -f["n"] * max((problem.cost[r] for r in f["routes"]), default=0), f["id"]
    ))
    best_score = _score(problem, incumbent)
    for offset in range(0, len(ranked), 16):
        if deadline - monotonic() < 0.15:
            break
        free = {f["id"] for f in ranked[offset:offset + 16]}
        trial = _candidate(problem, incumbent, min(deadline, monotonic() + 5), seed,
                           free=free, per_carrier=1, route_limit=5, safe_only=safe_only)
        if trial is not None and trial[1] < best_score:
            incumbent, best_score = trial
    return incumbent, best_score


def _polish(instance, problem, best, allocation, free, deadline, seed):
    allocation, score = allocation
    if score != _independent_score(instance, problem, allocation):
        log.warning("Discarded no-reuse allocation after validation")
        return best
    best = _best(instance, problem, best, (allocation, score))
    fixed = _candidate(problem, allocation, min(deadline, monotonic() + 5), seed,
                       free=set(), reuse=True)
    best = _best(instance, problem, best, fixed)
    start = fixed[0] if fixed is not None else allocation
    trial = _candidate(problem, start, deadline, seed, free=free,
                       support_only=True, reuse=True)
    return _best(instance, problem, best, trial)


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
    deadline = start + time_limit
    if best is not None and len(problem.routes) > 500:
        # The official reliability instance has a large always-surviving catalog.
        # On that catalog the scenario rows collapse to one set.
        safe_regular = sum(
            not problem.mask[r["id"]] and r["carrier"] != problem.spot
            for r in problem.routes
        )
        safe_only = safe_regular * 3 > len(problem.routes)
        warm = pack(unpack(best)[0], {})
        construction = _construct(problem, warm, start + 0.4 * time_limit, seed, safe_only)
        best = _polish(instance, problem, best, construction, free,
                       min(deadline, monotonic() + 10), seed)
        if safe_only and deadline - monotonic() > 5:
            trial = _candidate(problem, construction[0], deadline - 4, seed,
                               free=free, per_carrier=1, route_limit=3, safe_only=True)
            if trial is not None:
                best = _polish(instance, problem, best, trial, free, deadline, seed)
    else:
        trial = _candidate(problem, best, start + 0.75 * time_limit, seed, free=free)
        if trial is not None:
            best = _polish(instance, problem, best, trial, free, deadline, seed)
    if best is None:
        raise RuntimeError("No feasible allocation was found within the time limit")
    return pack(*unpack(best))


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from common import load_instance

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("instance")
    cli.add_argument("output", type=Path)
    cli.add_argument("--initial")
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument("--free-flows", type=int, nargs="*")
    modes.add_argument("--fixed-support", action="store_true")
    cli.add_argument("--time", type=float, default=120)
    cli.add_argument("--seed", type=int, default=0)
    args = cli.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    instance = load_instance(args.instance)
    initial = load_instance(args.initial) if args.initial else None
    problem = Problem(instance)
    if args.free_flows is not None:
        if initial is None:
            cli.error("--free-flows requires --initial")
        solution = _neighborhood(
            instance, problem, initial, set(args.free_flows), monotonic() + args.time, args.seed
        )
    elif args.fixed_support:
        if initial is None or _independent_score(instance, problem, initial) is None:
            cli.error("--fixed-support requires a feasible --initial solution")
        trial = _candidate(problem, initial, monotonic() + args.time, args.seed,
                           free={f["id"] for f in problem.flows}, support_only=True, reuse=True)
        solution = _best(instance, problem, initial, trial)
    else:
        solution = solve(instance, initial, args.time, args.seed)
    costs = _scenario_costs(problem, solution)
    if costs is None:
        raise RuntimeError("The solution failed the exact model checks")
    worst = [s for s, value in costs.items() if value == max(costs.values())]
    print(f"Model-checked score {max(costs.values())}; worst scenarios {worst}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")
