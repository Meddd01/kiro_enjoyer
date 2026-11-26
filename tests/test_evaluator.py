"""Adversarial tests for evaluator.py on a hand-computable mini instance (dev-only, not production code)."""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import evaluate  # noqa: E402
import baseline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mini(alpha=0.25, scen=None):
    # sites 0,1,2 ; ports 0..3 real, 4 SPOT ; carriers 0=A 1=B 2=SPOT
    def site(i, reutil):
        return {"id": i, "name": f"S{i}", "loc": {}, "reutil": reutil,
                "ports": [{"port": p, "cost": 10 * (i + 1) + p} for p in range(5)]}
    return {
        "meta": {"name": "mini", "alpha": alpha, "units": {}, "sizes": {}, "description": ""},
        "ports": [{"id": p, "code": f"P{p}", "name": "", "manut": 100, "loc": {}} for p in range(4)]
        + [{"id": 4, "code": "SPOT", "name": "", "manut": 1000, "loc": {}}],
        "sites": [site(0, 100), site(1, 3), site(2, 100)],
        "carriers": [{"id": 0, "name": "A", "fixed_cost": 1000, "trans": 20, "sites": 3},
                     {"id": 1, "name": "B", "fixed_cost": 500, "trans": 40, "sites": 3},
                     {"id": 2, "name": "SPOT", "fixed_cost": 0, "trans": 1000, "sites": 3}],
        "flows": [{"id": 0, "t_dep": 0, "t_arr": 1, "n": 10, "routes": [0, 1, 2]},
                  {"id": 1, "t_dep": 1, "t_arr": 2, "n": 5, "routes": [3, 4]}],
        "routes": [{"id": 0, "flow": 0, "carrier": 0, "p_dep": 0, "p_arr": 1, "cost": 100},
                   {"id": 1, "flow": 0, "carrier": 1, "p_dep": 0, "p_arr": 1, "cost": 150},
                   {"id": 2, "flow": 0, "carrier": 2, "p_dep": 4, "p_arr": 4, "cost": 400},
                   {"id": 3, "flow": 1, "carrier": 0, "p_dep": 1, "p_arr": 2, "cost": 120},
                   {"id": 4, "flow": 1, "carrier": 2, "p_dep": 4, "p_arr": 4, "cost": 500}],
        "scenarios": scen if scen is not None else [{"id": 0, "routes": []}, {"id": 1, "routes": [0]}],
    }


def tc(t, p):
    return 10 * (t + 1) + p


def D(inst, r):
    ro = inst["routes"][r]
    f = inst["flows"][ro["flow"]]
    return 2 * tc(f["t_dep"], ro["p_dep"]) + ro["cost"] + 2 * tc(f["t_arr"], ro["p_arr"])


def sol(x, y=()):
    return {"x": [{"route": r, "n": n} for r, n in x.items()],
            "y": [{"from": i, "to": j, "n": n} for i, j, n in y]}


def bad(res, needle):
    assert not res["valid"] and res["score"] is None, res
    assert any(needle in e for e in res["errors"]), res["errors"]


def test_exact_cost_with_split_and_reroute():
    inst = mini()
    res = evaluate(inst, sol({0: 9, 2: 1, 4: 5}))
    assert res["valid"], res["errors"]
    s0 = 9 * D(inst, 0) + 1 * D(inst, 2) + 5 * D(inst, 4) + 1000 * 2  # A touches sites 0,1
    s1 = 10 * D(inst, 2) + 5 * D(inst, 4)  # all of f0 on SPOT, A disappears -> fee vanishes
    assert res["scenario_costs"] == {0: s0, 1: s1}
    assert res["score"] == max(s0, s1)


def test_single_failing_route_invalid():
    bad(evaluate(mini(), sol({0: 10, 4: 5})), "lost all its routes")


def test_both_routes_fail_invalid():
    inst = mini(scen=[{"id": 0, "routes": []}, {"id": 1, "routes": [0, 1]}])
    bad(evaluate(inst, sol({0: 5, 1: 5, 4: 5})), "lost all its routes")


def test_demand_three_routes_wrong_flow_dupes_types():
    inst = mini()
    bad(evaluate(inst, sol({0: 4, 2: 5, 4: 5})), "!= n_f")
    bad(evaluate(inst, sol({0: 4, 1: 3, 2: 3, 4: 5})), "(>2)")
    bad(evaluate(inst, sol({1: 10, 2: 0, 4: 5})), "non-positive")
    bad(evaluate(inst, {"x": [{"route": 2, "n": 10}, {"route": 2, "n": 10}, {"route": 4, "n": 5}]}), "twice")
    bad(evaluate(inst, {"x": [{"route": 2, "n": 10.0}, {"route": 4, "n": 5}]}), "non-integer")
    bad(evaluate(inst, {"x": [{"route": 2, "n": True}, {"route": 4, "n": 5}]}), "non-integer")
    bad(evaluate(inst, {"x": [{"route": 99, "n": 10}, {"route": 4, "n": 5}]}), "unknown route")
    bad(evaluate(inst, {"x": [], "y": [], "extra": 1}), "keys")
    # route 3 belongs to flow 1: putting it in flow 0's place breaks flow 0 demand
    bad(evaluate(inst, sol({2: 10, 3: 2, 4: 5})), "!= n_f")


def test_backup_1teu_failure_and_capacity_only_in_disruption():
    inst = mini(alpha=0)
    inst["ports"][1]["manut"] = 14  # nominal 9 (r1) + 5 (r3) = 14 ok ; scenario 1 not touching r1
    res = evaluate(inst, sol({1: 9, 2: 1, 3: 5}))
    assert res["valid"], res["errors"]
    inst2 = mini(alpha=0, scen=[{"id": 0, "routes": []}, {"id": 1, "routes": [2]}])
    inst2["ports"][1]["manut"] = 14  # SPOT backup fails -> r1 gets 10 -> 10 + 5 = 15 > 14
    bad(evaluate(inst2, sol({1: 9, 2: 1, 3: 5})), "port 1 handles 15")


def test_spot_self_loop_counts_twice():
    inst = mini()
    inst["ports"][4]["manut"] = 29  # SPOT routes: 2*10 + 2*5 = 30
    bad(evaluate(inst, sol({2: 10, 4: 5})), "port 4 handles 30")
    inst["ports"][4]["manut"] = 30
    assert evaluate(inst, sol({2: 10, 4: 5}))["valid"]


def test_alpha_threshold_and_spot_exempt():
    inst = mini(alpha=0.3)  # ceil(0.3*20) = 6 for carrier A ; SPOT exempt (1 TEU on SPOT is fine)
    bad(evaluate(inst, sol({0: 5, 2: 5, 4: 5})), "ceil(alpha*trans)=6")
    assert evaluate(inst, sol({0: 6, 2: 4, 4: 5}))["valid"]
    inst = mini(alpha=0.3, scen=[{"id": 0, "routes": []}, {"id": 1, "routes": [3]}])
    # nominal A = 4 + 5 = 9 ok ; scenario 1 kills r3 -> A keeps 4 < 6 -> invalid
    bad(evaluate(inst, sol({0: 4, 2: 6, 3: 4, 4: 1})), "scenario 1: carrier 0 carries 4")


def test_trans_and_sites():
    inst = mini(alpha=0)
    inst["carriers"][0]["trans"] = 13
    bad(evaluate(inst, sol({0: 9, 2: 1, 3: 5})), "trans 13")
    inst = mini(alpha=0)
    inst["carriers"][0]["sites"] = 2  # A on f0 (sites 0,1) and f1 (sites 1,2) -> 3 distinct sites
    bad(evaluate(inst, sol({0: 9, 2: 1, 3: 4, 4: 1})), "3 sites > 2")


def test_reuse_rules():
    inst = mini(alpha=0)
    base = {0: 9, 2: 1, 3: 4, 4: 1}
    sav = tc(1, 1) + tc(1, 1)  # arrival of r0 at site 1 via port 1 ; departure of r3 from site 1 via port 1
    ok = evaluate(inst, sol(base, [(0, 3, 3)]))
    assert ok["valid"], ok["errors"]
    ref = evaluate(inst, sol(base))
    assert ok["scenario_costs"][0] == ref["scenario_costs"][0] - 3 * sav
    assert ok["scenario_costs"][1] == ref["scenario_costs"][1]  # r0 fails -> reuse vanishes, none created
    bad(evaluate(inst, sol(base, [(0, 3, 4)])), "reutil 3")  # site 1 reutil = 3
    inst["sites"][1]["reutil"] = 100
    bad(evaluate(inst, sol(base, [(0, 3, 5)])), "receives 5 reused > x^s=4")  # incoming bound on r3
    bad(evaluate(inst, sol(base, [(2, 3, 1)])), "different carriers")
    bad(evaluate(inst, sol(base, [(3, 0, 1)])), "t_arr")  # f1 arrives at 2, f0 departs from 0
    bad(evaluate(inst, sol(base, [(0, 3, 1), (0, 3, 1)])), "twice")
    # supply bound on r0 counted separately from its own receipt
    bad(evaluate(inst, sol({0: 9, 2: 1, 3: 4, 4: 1}, [(0, 3, 10)])), "supplies 10")


def test_reuse_cycle_and_give_and_receive():
    inst = mini(scen=[{"id": 0, "routes": []}])
    inst["flows"].append({"id": 2, "t_dep": 1, "t_arr": 0, "n": 7, "routes": [5]})
    inst["routes"].append({"id": 5, "flow": 2, "carrier": 0, "p_dep": 1, "p_arr": 0, "cost": 90})
    inst["carriers"][0]["trans"] = 40
    inst["sites"][1]["reutil"] = 100
    # r0: 0->1 ; r5: 1->0 ; cycle r0->r5 at site 1 and r5->r0 at site 0 ; r0 both gives and receives
    res = evaluate(inst, sol({0: 10, 3: 5, 5: 7}, [(0, 5, 5), (5, 0, 7), (0, 3, 5)]))
    assert res["valid"], res["errors"]
    bad(evaluate(inst, sol({0: 10, 3: 5, 5: 7}, [(0, 5, 6), (5, 0, 7), (0, 3, 5)])), "supplies 11")


def test_no_mutation():
    inst = mini()
    s = sol({0: 9, 2: 1, 4: 5})
    i2, s2 = copy.deepcopy(inst), copy.deepcopy(s)
    evaluate(inst, s)
    assert inst == i2 and s == s2


def test_real_instances_baselines():
    for name in ("alliances", "reliability", "strikes"):
        inst = json.load(open(os.path.join(ROOT, "instances", f"{name}.json")))
        assert evaluate(inst, baseline.solve(inst, reuse=False))["score"] == 515357081
        assert evaluate(inst, baseline.solve(inst, reuse=True))["score"] == 505183761
