"""Single entry point that regenerates our KIRO 2026 solutions with the production pipeline (team "double span").

python regenerate.py [--instances alliances,reliability,strikes] [--from DIR] [--out solutions/regen]
                     [--sa 600] [--small 1800] [--unlock 1800] [--big 1200] [--seed 0]

Per instance: start (baseline -> penalised SA from heuristic.py, or the warm start in --from) -> lns.py stages
(1) adaptive small neighbourhoods, (2) 'unlock' scope (big SPOT flow + port holders by cheapest detour + one site
of a full carrier), (3) big exact repairs (15-25% of the flows) -> global split+reuse polish (Codex MILP) ->
evaluator check. Every stage only ever keeps evaluator-valid strict improvements, so the output is always valid.
The submitted solutions came from this pipeline run for several hours on 8 parallel workers with different seeds
(each worker restarting from the best file found so far); one sequential run with the default budget is shorter,
so it reaches slightly higher costs.
All files sit in one folder (Codex's milp_solver.py, common.py, reuse.py next to ours); needs numpy + highspy.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from evaluator import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [  # (name, budget arg, lns.py args)
    ("small", "small", ["--mode", "alns", "--sub", "25", "--size", "25", "--lex"]),
    ("unlock", "unlock", ["--mode", "alns", "--kinds", "unlock,bigspot,swap", "--sub", "280", "--size", "60", "--lex"]),
    ("big", "big", ["--mode", "big", "--destroy", "0.15-0.25", "--sub", "110", "--size", "30", "--lex"]),
]


def start(name, inst, a, path):
    if a.warm:
        sol = json.load(open(os.path.join(a.warm, f"{name}.json")))
    else:
        import baseline
        import heuristic
        sol = heuristic.solve(inst, initial_solution=baseline.solve(inst), time_limit=a.sa, seed=a.seed)
    json.dump(sol, open(path, "w"))
    return evaluate(inst, sol)["score"]


def polish(name, inst, src, dst, secs):
    import milp_solver as ms
    from common import Problem
    sol = json.load(open(src))
    p = Problem(inst)
    trial = ms._candidate(p, sol, time.monotonic() + secs, 0, free={f["id"] for f in inst["flows"]},
                          support_only=True, reuse=True)
    out = ms.pack(*ms.unpack(ms._best(inst, p, sol, trial)))
    r0, r1 = evaluate(inst, sol), evaluate(inst, out)
    json.dump(out if r1["valid"] and r1["score"] < r0["score"] else sol, open(dst, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="alliances,reliability,strikes")
    ap.add_argument("--from", dest="warm", default="", help="folder with warm-start <name>.json (skips baseline+SA)")
    ap.add_argument("--out", default="solutions/regen")
    ap.add_argument("--sa", type=float, default=600)
    ap.add_argument("--small", type=float, default=1800)
    ap.add_argument("--unlock", type=float, default=1800)
    ap.add_argument("--big", type=float, default=1200)
    ap.add_argument("--polish", type=float, default=90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.environ.get("PYTHONPATH", "")]))
    total = 0
    for name in a.instances.split(","):
        inst = json.load(open(f"instances/{name}.json"))
        stage_dir = os.path.join(a.out, "stage0")
        os.makedirs(stage_dir, exist_ok=True)
        print(f"{name}: start {start(name, inst, a, os.path.join(stage_dir, f'{name}.json'))}", flush=True)
        for k, (label, budget, args) in enumerate(STAGES, 1):
            secs = getattr(a, budget)
            if secs <= 0:
                continue
            nxt = os.path.join(a.out, f"stage{k}_{label}")
            subprocess.run([sys.executable, os.path.join(HERE, "lns.py"), "--instance", name, "--init", stage_dir,
                            "--out", nxt, "--time", str(secs), "--seed", str(a.seed + k)] + args,
                           env=env, check=True, stdout=subprocess.DEVNULL)
            if os.path.exists(os.path.join(nxt, f"{name}.json")):  # lns.py writes only when it improved
                stage_dir = nxt
            print(f"{name}: after {label} {evaluate(inst, json.load(open(os.path.join(stage_dir, f'{name}.json'))))['score']}",
                  flush=True)
        dst = os.path.join(a.out, f"{name}.json")
        polish(name, inst, os.path.join(stage_dir, f"{name}.json"), dst, a.polish) if a.polish > 0 else \
            json.dump(json.load(open(os.path.join(stage_dir, f"{name}.json"))), open(dst, "w"))
        r = evaluate(inst, json.load(open(dst)))
        assert r["valid"], f"{name}: invalid output {r['errors'][:3]}"
        total += r["score"]
        print(f"{name}: FINAL {r['score']} -> {dst}", flush=True)
    print("TOTAL", total)


if __name__ == "__main__":
    main()
