"""Claude's worker portfolio launcher for lns.py (dev orchestration; the solution code is lns.py + Codex engine).

python portfolio.py --until 17:20 [--slots 6]
- distinct configs (mode, destroy size, sub-time, instance), each worker writes solutions/pf_<slot>/ and logs
  runs/claude/warm_pf_<slot>.txt (shown on the dashboard);
- every 2 min: parse logs; reseed stalled workers (fast families 10 min, big/tail/cross 25 min) from a fresh team-best
  snapshot; a family with >=1 win keeps its slot family, otherwise it switches to the best family by wins/min;
- every 10 min: an instance with 20+ min of zero wins gives one worker (keeping >=1) to the instance with the highest
  (wins/min) x (gap to the best on the board).
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

FAMILIES = {  # name: (lns args, stall minutes); all use the lexicographic (plateau-walking) repair
    "alns": (["--mode", "alns", "--sub", "25", "--size", "25", "--lex"], 10),
    "alns-kick": (["--mode", "alns", "--sub", "25", "--size", "25", "--kick", "12", "--lex"], 10),
    "big": (["--mode", "big", "--destroy", "0.15-0.25", "--sub", "110", "--size", "30", "--lex"], 25),
    "tail": (["--mode", "tail", "--sub", "45", "--size", "30", "--lex"], 25),
}  # "cross" dropped: pool has only 2 diverse members and a 60-flow crossover repair returned nothing in 104 s
INITIAL = [("reliability", "tail"), ("reliability", "alns"), ("strikes", "tail"), ("strikes", "alns-kick"),
           ("alliances", "alns"), ("strikes", "big")]
ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, "..", "kiro_codex_view")]))


def snapshot():
    subprocess.run([PY, "runs/claude/snapshot.py"], cwd=ROOT, env=ENV, capture_output=True)


def board_gaps():
    try:
        d = json.load(open(os.path.join(ROOT, "runs/claude/live/live.json")))
        return {n: max(1.0, d["best"][n]["score"] - (d["board_best_inst"][i] or 0))
                for i, n in enumerate(["alliances", "reliability", "strikes"])}
    except (OSError, ValueError, KeyError, TypeError):
        return {"alliances": 5e6, "reliability": 7e6, "strikes": 6e6}


class Slot:
    def __init__(self, k, inst, fam, seed):
        self.k, self.inst, self.fam, self.seed, self.proc = k, inst, fam, seed, None

    def start(self, until):
        snapshot()
        secs = max(60, int(until - time.time()))
        log = os.path.join(ROOT, f"runs/claude/warm_pf_{self.k}.txt")
        self.seed += 100
        cmd = [PY, "lns.py", "--instance", self.inst, "--init", "solutions/team_best", "--out", f"solutions/pf_{self.k}",
               "--time", str(secs), "--seed", str(self.seed)] + FAMILIES[self.fam][0]
        with open(log, "a") as fh:
            fh.write(f"### {datetime.datetime.now():%H:%M:%S} START slot {self.k} {self.inst} {self.fam}\n")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=open(log, "a"), stderr=subprocess.STDOUT)
        self.started, self.last_win, self.wins = time.time(), time.time(), 0

    def poll(self):
        log = os.path.join(ROOT, f"runs/claude/warm_pf_{self.k}.txt")
        txt = open(log, encoding="utf-8", errors="replace").read().split(f"START slot {self.k}")[-1]
        wins = len(re.findall(r" -> \d+", txt))
        if wins > self.wins:
            self.wins, self.last_win = wins, time.time()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True, help="local HH:MM to stop all workers")
    ap.add_argument("--slots", type=int, default=6)
    a = ap.parse_args()
    hh, mm = map(int, a.until.split(":"))
    now = datetime.datetime.now()
    until = now.replace(hour=hh, minute=mm, second=0).timestamp()
    slots = [Slot(k, i, f, 1000 + k) for k, (i, f) in enumerate(INITIAL[: a.slots])]
    fam_wins, fam_time = {f: 0 for f in FAMILIES}, {f: 1e-9 for f in FAMILIES}
    inst_last_win = {n: time.time() for n in ["alliances", "reliability", "strikes"]}
    for s in slots:
        s.start(until)
    last_realloc = time.time()
    while time.time() < until - 30:
        time.sleep(120)
        for s in slots:
            before = s.wins
            s.poll()
            fam_wins[s.fam] += s.wins - before
            fam_time[s.fam] += 2.0
            if s.wins > before:  # wins = strict improvements (plateau moves are logged separately)
                inst_last_win[s.inst] = time.time()
        rate = {f: fam_wins[f] / fam_time[f] for f in FAMILIES}
        for s in slots:
            stall = (time.time() - s.last_win) / 60
            if s.proc.poll() is not None or stall >= FAMILIES[s.fam][1]:
                s.stop()
                if fam_wins[s.fam] == 0 and s.proc.poll() is not None or stall >= FAMILIES[s.fam][1] and fam_wins[s.fam] == 0:
                    s.fam = max(rate, key=rate.get)
                print(f"{datetime.datetime.now():%H:%M} reseed slot {s.k} -> {s.inst}/{s.fam} (stall {stall:.0f} min)",
                      flush=True)
                s.start(until)
        if time.time() - last_realloc >= 600:
            last_realloc = time.time()
            gaps = board_gaps()
            per = {n: [s for s in slots if s.inst == n] for n in gaps}
            irate = {n: sum(s.wins for s in per[n]) / max(1.0, sum((time.time() - s.started) / 60 for s in per[n]))
                     for n in gaps}
            hot = max(gaps, key=lambda n: (irate[n] + 0.01) * gaps[n])
            for n in gaps:
                if n != hot and len(per[n]) > 1 and time.time() - inst_last_win[n] > 1200:
                    s = per[n][-1]
                    s.stop()
                    s.inst = hot
                    print(f"{datetime.datetime.now():%H:%M} move slot {s.k} {n} -> {hot}", flush=True)
                    s.start(until)
                    break
        json.dump([{"slot": s.k, "instance": s.inst, "family": s.fam, "wins": s.wins,
                    "stall_min": round((time.time() - s.last_win) / 60, 1)} for s in slots],
                  open(os.path.join(ROOT, "runs/claude/portfolio.json"), "w"))
    for s in slots:
        s.stop()
    snapshot()
    print("portfolio finished", flush=True)


if __name__ == "__main__":
    main()
