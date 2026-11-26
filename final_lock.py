"""Final lock (dev tool): best valid per instance -> global split+reuse polish -> double validation -> FINAL + ZIP.

PYTHONPATH=.:../kiro_codex_view python final_lock.py [--polish 90] [--zip]
Never writes a FINAL file that is not evaluator-valid AND confirmed by Codex's exact scenario costs.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
import zipfile

from evaluator import evaluate

NAMES = ["alliances", "reliability", "strikes"]
SOURCES = ["lns.py", "heuristic.py", "baseline.py", "evaluator.py",  # Claude
           "../kiro_codex_view/milp_solver.py", "../kiro_codex_view/common.py", "../kiro_codex_view/reuse.py"]  # Codex


def best_file(n, inst):
    best = None
    for p in glob.glob(f"solutions/**/{n}.json", recursive=True) + [f"codex_teamwarmup/{n}.json"]:
        if "history" in p.replace("\\", "/") or "FINAL" in p:
            continue
        try:
            r = evaluate(inst, json.load(open(p)))
        except (OSError, ValueError):
            continue
        if r["valid"] and (best is None or r["score"] < best[0]):
            best = (r["score"], p)
    return best


def polish(args):
    n, path, secs = args
    import milp_solver as ms
    from common import Problem
    inst = json.load(open(f"instances/{n}.json"))
    sol = json.load(open(path))
    p = Problem(inst)
    trial = ms._candidate(p, sol, time.monotonic() + secs, 0, free={f["id"] for f in inst["flows"]},
                          support_only=True, reuse=True)
    out = ms.pack(*ms.unpack(ms._best(inst, p, sol, trial)))
    r0, r1 = evaluate(inst, sol), evaluate(inst, out)
    return n, (out if r1["valid"] and r1["score"] < r0["score"] else sol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polish", type=float, default=90)
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    import milp_solver as ms
    from common import Problem
    insts = {n: json.load(open(f"instances/{n}.json")) for n in NAMES}
    picks = {n: best_file(n, insts[n]) for n in NAMES}
    for n in NAMES:
        print(f"{n}: best {picks[n][0]} from {picks[n][1]}", flush=True)
    jobs = [(n, picks[n][1], a.polish) for n in NAMES] if a.polish > 0 else []
    polished = {}
    if jobs:
        with mp.Pool(3) as pool:
            polished = dict(pool.map(polish, jobs))
    os.makedirs("solutions/FINAL", exist_ok=True)
    total, report = 0, {}
    for n in NAMES:
        sol = polished.get(n) or json.load(open(picks[n][1]))
        r = evaluate(insts[n], sol)
        codex = ms._scenario_costs(Problem(insts[n]), sol)
        agree = codex is not None and max(codex.values()) == r["score"] and all(
            codex[s] == c for s, c in r["scenario_costs"].items() if s in codex)
        assert r["valid"] and agree, f"{n}: validation disagreement (valid={r['valid']}, codex agree={agree})"
        dst = f"solutions/FINAL/{n}.json"
        if os.path.exists(dst):
            old = evaluate(insts[n], json.load(open(dst)))
            if old["valid"] and old["score"] <= r["score"]:
                print(f"{n}: existing FINAL {old['score']} kept (new {r['score']} not better)")
                total += old["score"]
                continue
            shutil.copy(dst, f"solutions/FINAL/{time.strftime('%H%M%S')}_{n}.bak.json")
        json.dump(sol, open(dst, "w"))
        sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        report[n] = {"score": r["score"], "sha256": sha, "source": picks[n][1]}
        total += r["score"]
        print(f"{n}: FINAL {r['score']} (polish {picks[n][0] - r['score']:+d}) sha256 {sha[:16]}", flush=True)
    print("FINAL TOTAL", total)
    json.dump(report, open("solutions/FINAL/report.json", "w"), indent=1)
    if a.zip:
        lines = 0
        with zipfile.ZipFile("submission_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for src in SOURCES:
                z.write(src, os.path.basename(src))
                lines += sum(1 for ln in open(src, encoding="utf-8") if ln.strip())
        print(f"submission_source.zip: {len(SOURCES)} files, {lines} non-empty lines (limit 3000)")


if __name__ == "__main__":
    main()
