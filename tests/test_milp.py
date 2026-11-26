import unittest
from time import monotonic

from common import Problem, pack
from milp_solver import _candidate, _neighborhood, _score, solve
from reuse import optimize_reuse


def tiny(*, demand=5, scenarios=((), (0,)), caps=(10, 20), alpha=0, fee=0):
    return {
        "meta": {"alpha": alpha},
        "ports": [
            {"id": i, "manut": cap, "code": code}
            for i, (cap, code) in enumerate(zip(caps, ("REG", "SPOT")))
        ],
        "sites": [{"id": 0, "reutil": 3, "ports": [{"port": 0, "cost": 2},
                                                 {"port": 1, "cost": 3}]}],
        "carriers": [
            {"id": 0, "name": "REG", "trans": 10, "sites": 1, "fixed_cost": fee},
            {"id": 1, "name": "SPOT", "trans": 20, "sites": 1, "fixed_cost": 0},
        ],
        "flows": [{"id": 0, "t_dep": 0, "t_arr": 0, "n": demand, "routes": [0, 1]}],
        "routes": [
            {"id": i, "flow": 0, "carrier": i, "p_dep": i, "p_arr": i, "cost": 10 * i}
            for i in range(2)
        ],
        "scenarios": [{"id": i, "routes": list(failed)} for i, failed in enumerate(scenarios)],
    }


class AllocationTests(unittest.TestCase):
    def test_automatic_rerouting_and_disappearing_contract(self):
        p = Problem(tiny(fee=3))
        self.assertEqual(_score(p, pack({0: 4, 1: 1}, {})), 110)
        self.assertIsNone(_score(p, pack({0: 5}, {})))
        self.assertIsNone(_score(p, {"x": [{"route": 0, "n": 5}, {"route": 1, "n": 0}], "y": []}))
        self.assertIsNone(_score(Problem(tiny(scenarios=((), (0, 1)))), pack({0: 4, 1: 1}, {})))

    def test_reliable_pair_may_split_when_neither_can_take_demand(self):
        instance = tiny(scenarios=((),), caps=(6, 4))
        self.assertEqual(solve(instance, time_limit=4), pack({0: 3, 1: 2}, {(0, 0): 1, (1, 1): 2}))

    def test_disruption_overloads_the_backup_and_self_loop_counts_twice(self):
        p = Problem(tiny(caps=(10, 8)))
        self.assertIsNone(_score(p, pack({0: 4, 1: 1}, {})))

    def test_minimum_volume_after_partial_loss(self):
        instance = tiny(alpha=0.4)
        instance["flows"].append({"id": 1, "t_dep": 0, "t_arr": 0, "n": 1, "routes": [2]})
        instance["routes"].append({"id": 2, "flow": 1, "carrier": 0,
                                   "p_dep": 0, "p_arr": 0, "cost": 0})
        self.assertIsNone(_score(Problem(instance), pack({0: 4, 1: 1, 2: 1}, {})))

    def test_reuse_disappears_and_uses_shared_site_capacity(self):
        instance = tiny()
        initial = pack({0: 2, 1: 3}, {})
        result = optimize_reuse(instance, initial, time_limit=4)
        self.assertEqual(result, pack({0: 2, 1: 3}, {(1, 1): 3}))
        self.assertEqual(_score(Problem(instance), result), 92)
        self.assertEqual(initial["y"], [])
        self.assertIsNone(_score(Problem(instance), pack({0: 2, 1: 3}, {(0, 1): 1})))
        self.assertIsNone(_score(Problem(instance), pack({0: 2, 1: 3}, {(0, 0): 2, (1, 1): 2})))

    def test_frozen_flow_and_fixed_support(self):
        instance = tiny(scenarios=((),))
        p = Problem(instance)
        initial = pack({0: 1, 1: 4}, {})
        fixed = _neighborhood(instance, p, initial, set(), monotonic() + 4, 0)
        self.assertEqual(fixed["x"], initial["x"])
        improved, score = _candidate(p, initial, monotonic() + 4, 0, free={0},
                                     support_only=True, reuse=True)
        self.assertEqual(improved["x"], pack({0: 4, 1: 1}, {})["x"])
        self.assertEqual(score, 40)


if __name__ == "__main__":
    unittest.main()
