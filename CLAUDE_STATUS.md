# CLAUDE_STATUS — Claude's handoff log (Claude-owned; Codex/humans read-only)

## File ownership (anti-merge-conflict contract)
- Claude edits ONLY: `evaluator.py`, `baseline.py`, `heuristic.py`, `benchmark.py`, `tests/test_evaluator.py`, `CLAUDE_STATUS.md`, `solutions/claude/`.
- Claude never edits: `common.py`, `milp_solver.py`, `reuse.py`, `tests/test_milp.py`, `tests/test_reuse.py` (Codex); `main.py`, `requirements.txt`, `README.md`, `.gitignore`, `best/`, ZIP (humans); `instances/` (read-only).
- Branch `feat/claude-heuristics` (from `main` @ dfe04b7). Staging by explicit path only. No push/merge without the user's go.
- Untracked and left for the humans: `instances/`, PDFs, `prompt.md`, `claude+codex.md`, `codex_output..md`, `last_output_from_codex.md`, `.venv/`, `runs/`.
- Proposed `.gitignore` additions (humans decide): `.venv/`, `runs/`, `__pycache__/`.

## Interfaces (unchanged from brief)
- `evaluator.evaluate(instance: dict, solution: dict) -> {"valid", "score", "scenario_costs", "errors"}`. CLI: `python evaluator.py instances/X.json sol.json`
- `baseline.solve(instance, reuse=True) -> dict` (all-SPOT + greedy reuse). Also `baseline.greedy_reuse(instance, x: {route: n}, weight=None) -> y list`.
- `heuristic.solve(instance, initial_solution=None, time_limit=120.0, seed=0) -> dict` (penalised SA + compound evacuate moves; always returns an evaluator-valid solution, falls back to initial/baseline)
- Env: Python 3.12 + numpy (pytest for tests). No solver needed for Claude's code.

## Checkpoint 1 — evaluator + baselines (validated)
- `python -m pytest -q tests/test_evaluator.py` → 12 passed (adversarial checklist on a hand-computable mini instance + real baselines).
- `python baseline.py instances/*.json` / `python benchmark.py --solver baseline`:

| instance | all-SPOT no reuse | all-SPOT + reuse (21 004 TEU) |
|---|---:|---:|
| alliances | 515 357 081 | **505 183 761** |
| reliability | 515 357 081 | **505 183 761** |
| strikes | 515 357 081 | **505 183 761** |
| combined | 1 546 071 243 | **1 515 551 283** |

Both Codex reference values reproduced exactly. Champions: `solutions/claude/<name>.json` (+ `.meta.json` provenance).

## Checkpoint 2 — heuristic v1 (validated by evaluator.py)
Command: `python benchmark.py --solver heuristic --instances <name> --time <T> --seed 0 --init solutions/claude`

| instance | baseline | heuristic | T (s) | worst scenario |
|---|---:|---:|---:|---|
| alliances | 505 183 761 | **319 873 861** | 90 | s2 |
| reliability | 505 183 761 | **308 103 758** | 120 | s19 |
| strikes | 505 183 761 | **360 058 343** | 120 | s13 |
| combined | 1 515 551 283 | **988 035 962** | | |

Solutions + provenance: `solutions/claude/<name>.json` / `.meta.json` (committed on feat/claude-heuristics). Codex: these are valid warm starts for the MILP (MIP start).
Method: per-flow options = robust singletons + disjoint-failure-mask pairs (top-K); exact per-scenario bookkeeping (loads, contract refcounts);
objective max_s C_s + 0.02 mean_s C_s + lambda*violation (adaptive lambda, graded site penalty); SA with T0 calibrated from sampled uphill deltas;
0.4% compound "evacuate carrier m (at site t)" moves; greedy reuse post-pass weighted to critical scenarios. Still improving at time limit.

## Notes for Codex's MILP (insights, no file edits)
1. Rerouted x^s exact & big-M-free: for affected flow f in s, continuous x^s_r with x^s_r = 0 (r failed), x^s_r >= x_r, x^s_r <= n_f u_r, sum_r x^s_r = n_f. Given integral (x,u) it is exact AND enforces survival. Unaffected flows: x^s = x.
2. y-feasibility (7),(8) and sites(m) are scenario-monotone → nominal rows only. Scenario-dependent: manut, trans, alpha (partial carrier failure; not in alliances), contract fees e^s.
3. Reuse saving is separable (B_i + A_j); within (carrier, site) the y-block can be aggregated by failure-signature groups (alliances: 1 signature per carrier → one pool per (m,t)).

## Line count (production path)
evaluator.py + baseline.py + benchmark.py — see `wc -l` at each checkpoint.
