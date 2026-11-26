"""Robust integer container reuse for an existing allocation."""

from time import monotonic

from common import Problem, pack, unpack
from milp_solver import _best, _candidate, _independent_score


def optimize_reuse(instance, solution, time_limit=20.0, seed=0):
    """Keep all route quantities fixed and return a checked, nonworse solution."""
    deadline = monotonic() + time_limit
    problem = Problem(instance)
    x, _ = unpack(solution)
    if _independent_score(instance, problem, solution) is None:
        solution = pack(x, {})
        if _independent_score(instance, problem, solution) is None:
            raise ValueError("Reuse optimization requires a feasible route allocation")
    trial = _candidate(problem, solution, deadline, seed, free=set(), reuse=True)
    return pack(*unpack(_best(instance, problem, solution, trial)))
