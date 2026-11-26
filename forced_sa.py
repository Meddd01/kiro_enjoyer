"""Penalty-path experiment: forbid SPOT for the given big flows (what-if: f54 alone is worth -6..-9M), let the
penalised SA (heuristic.search: infeasible states allowed, lambda-weighted violation) repair ports / sites / carriers
around them, then reuse + Codex global split/reuse polish. Writes <out>/<name>.json only if evaluator-valid.

python forced_sa.py reliability 54 600 1 solutions/FSA_rel
"""
import json
import math
import os
import sys
import time

import heuristic
from evaluator import evaluate


def main():
    name, flows_, sa_t, seed, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    forced = [int(f) for f in flows_.split(",")]
    inst = json.load(open(f"instances/{name}.json"))
    sol = json.load(open(f"solutions/team_best/{name}.json"))
    base = evaluate(inst, sol)["score"]
    md = heuristic.Model(inst)
    opts = heuristic._opt_from_solution(md, sol)
    for f in forced:  # real routes only; start on the cheapest robust option
        md.options[f] = [o for o in md.options[f] if md.rm[o[0]] != md.spot_m and (o[1] < 0 or md.rm[o[1]] != md.spot_m)]
        o = min(md.options[f], key=lambda o: md.D[o[0]] + (md.D[o[1]] if o[1] >= 0 else md.D[o[0]]))
        opts[f] = (o[0], o[1], md.n[f] - 1 if o[1] >= 0 else md.n[f])
    print(f"{name} base {base} forced {forced} start {[opts[f] for f in forced]}", flush=True)
    true, best = heuristic.search(md, opts, sa_t, seed, log=False, reheat=0.03)
    if best is None or true == math.inf:
        print(f"{name} SA found no feasible state with {forced} forced", flush=True)
        return
    st = heuristic.State(md)
    for f, o in enumerate(best):
        st.set(f, o)
    cand, res = heuristic.finalize(md, st.x_dict())
    print(f"{name} SA feasible {true} -> with reuse {res['score']}", flush=True)
    try:
        import milp_solver as ms
        from common import Problem
        problem = Problem(inst)
        trial = ms._candidate(problem, cand, time.monotonic() + 120, seed, free={f["id"] for f in inst["flows"]},
                              support_only=True, reuse=True)
        if trial:
            pol = ms.pack(*ms.unpack(trial[0]))
            rp = evaluate(inst, pol)
            if rp["valid"] and rp["score"] < res["score"]:
                cand, res = pol, rp
        print(f"{name} after polish {res['score']} (base {base}, delta {res['score'] - base})", flush=True)
    except Exception as exc:
        print("polish failed", repr(exc), flush=True)
    if res["valid"]:
        os.makedirs(out, exist_ok=True)
        json.dump(cand, open(os.path.join(out, f"{name}.json"), "w"))
        print(f"{name} SAVED {out} {res['score']}", flush=True)


if __name__ == "__main__":
    main()
