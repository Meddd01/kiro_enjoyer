"""Run a solver on instances, validate with the reference evaluator, log results, keep per-instance champions.

python benchmark.py --solver heuristic --instances all --time 120 --seed 0 [--init solutions/claude]
Champions: solutions/claude/<name>.json (+ .meta.json), replaced only if valid AND strictly better.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

from evaluator import evaluate, worst_scenario

NAMES = ["alliances", "reliability", "strikes"]
CHAMP = "solutions/claude"
RUNS = "runs/claude"


def champion_score(inst, name):
    p = os.path.join(CHAMP, f"{name}.json")
    if not os.path.exists(p):
        return None, None
    sol = json.load(open(p))
    res = evaluate(inst, sol)
    return (res["score"] if res["valid"] else None), sol


def commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except OSError:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="heuristic", choices=["baseline", "heuristic"])
    ap.add_argument("--instances", default="all")
    ap.add_argument("--time", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="folder with <name>.json warm starts (e.g. solutions/claude)")
    a = ap.parse_args()
    names = NAMES if a.instances == "all" else a.instances.split(",")
    os.makedirs(CHAMP, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)
    for name in names:
        inst = json.load(open(f"instances/{name}.json"))
        init = None
        if a.init and os.path.exists(os.path.join(a.init, f"{name}.json")):
            init = json.load(open(os.path.join(a.init, f"{name}.json")))
        t0 = time.monotonic()
        if a.solver == "baseline":
            import baseline
            sol = baseline.solve(inst)
        else:
            import heuristic
            sol = heuristic.solve(inst, initial_solution=init, time_limit=a.time, seed=a.seed)
        wall = time.monotonic() - t0
        res = evaluate(inst, sol)
        out = os.path.join(RUNS, f"{a.solver}_{name}_s{a.seed}_t{int(a.time)}.json")
        json.dump(sol, open(out, "w"))
        best, _ = champion_score(inst, name)
        improved = res["valid"] and (best is None or res["score"] < best)
        row = {"when": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit(),
               "instance": name, "solver": a.solver, "seed": a.seed, "time_limit": a.time,
               "wall": round(wall, 1), "valid": res["valid"], "score": res["score"],
               "worst": worst_scenario(res), "champion_before": best, "improved": improved}
        new = not os.path.exists(os.path.join(RUNS, "results.csv"))
        with open(os.path.join(RUNS, "results.csv"), "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{name:12s} valid={res['valid']} score={res['score']} worst=s{row['worst']} "
              f"wall={wall:.1f}s champion_before={best} improved={improved}", res["errors"][:3])
        if improved:
            json.dump(sol, open(os.path.join(CHAMP, f"{name}.json"), "w"))
            meta = dict(row, command=" ".join([os.path.basename(sys.executable)] + sys.argv),
                        scenario_costs=res["scenario_costs"])
            json.dump(meta, open(os.path.join(CHAMP, f"{name}.meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
