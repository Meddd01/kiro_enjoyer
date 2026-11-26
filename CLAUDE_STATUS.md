# CLAUDE_STATUS — Claude's handoff log (Claude-owned; Codex/humans read-only)

## File ownership (anti-merge-conflict contract)
- Claude edits ONLY: `evaluator.py`, `baseline.py`, `heuristic.py`, `benchmark.py`, `lns.py`, `tests/test_evaluator.py`, `CLAUDE_STATUS.md`, `solutions/claude/`, `.gitignore` (added at user's request).
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

## Checkpoint 3 — hybrid SA -> Codex MILP polish (validated, 13:37)
| instance | SA v1 600s | + codex --fixed-support 60s | + codex solve(initial) 120s |
|---|---:|---:|---:|
| alliances | 306 338 810 | 304 683 421 | **304 552 062** |
| reliability | 298 565 039 | 296 270 219 | **294 787 358** |
| strikes | 301 665 205 | 299 463 965 | **295 994 714** |
| combined | 906 569 054 | 900 417 605 | **895 334 134** |

- Codex `origin/feat/optimization` @ca27544 run from a read-only worktree (`../kiro_codex_view`), highspy 1.16.0. Codex's model-checked scores == evaluator.py scores exactly on all runs (independent cross-validation).
- Champions committed in `solutions/claude/` with pipeline/command in `.meta.json`. Final ZIP must contain both Claude and Codex sources used.
- `.gitignore` added on this branch at the user's request (Codex: don't add a second one → add/add conflict).

## Checkpoint 4 — leaderboard gap (14:00) — team "double span" not yet submitted
Leaderboard https://kiro.enpc.org/classement/ (39 ranked): #1 v2v 817.51M (272.98 / 280.24 / 264.29); best per instance 272.19 / 280.19 / 263.57.
Ours: 304.55 / 293.52 / 295.74 = 893.81M (~#28). Gap ~30M on alliances & strikes, ~13M on reliability.
Diagnosis: carrier `sites(m)` limits are saturated for almost all carriers; some carriers never opened (alliances: E; strikes: G,B,D);
9.5k–14k TEU stay on SPOT (~2.4x cost). Per-scenario LB (cheapest surviving route, no fixed/reuse): 224.2 / 228.5 / 216.2M.
=> The key decision is the carrier->site sets. **Codex suggestion:** binaries e_{m,t} (11x48) with sum_t e_{m,t} <= sites(m),
u_r <= e_{m,t_dep}, u_r <= e_{m,t_arr}; candidate routes per flow restricted by those sets — likely what top teams solve.
Experiments running: SA v3 site-swap compound move; `lns.py` = neighbourhood selection (SPOT-heavy sites / worst scenario /
carrier / big SPOT flows) delegated to `milp_solver._neighborhood` (Codex private fn; would switch to public `free_flows` kwarg).

## Checkpoint 5 — MILP-LNS (14:15): alliances 290.40M, reliability 292.33M, strikes 286.81M = 869.54M (all evaluator-valid)
Engine: `lns.py` picks neighbourhoods (site / worst-scenario / carrier / SPOT-heavy / regret) of 20-40 flows and re-solves them with
Codex `milp_solver._candidate(free=..., per_carrier 2-3, route_limit 6-12, reuse=True)`; accepts only evaluator-valid strict improvements.
**To run it on the Codex machine (extra compute):**
`PYTHONPATH=<path to feat/optimization checkout> python lns.py --instance alliances --time 1500 --sub 25 --seed 7 --out solutions/codex_lns`
(`--init` defaults to `solutions/claude`; `--out` keeps champions separate -> no file conflicts; humans pick the best per instance.)

## Checkpoint 6 (14:37): 284.03 / 290.77 / 282.35 = 857.15M (all evaluator-valid), via lns.py rounds 1-2
- Failed experiment: `option_milp.py` (option-based robust MILP with e[m,t] site binaries, exact splits; not committed).
  HiGHS: 19k cols / 91k rows / 2.4M nnz; 240s spent in presolve probing, no B&B progress; site-LNS variants (109-211 free flows)
  found nothing in 90s. Discarded per the 20-min rule; cores returned to lns.py.
- KIRO site (public pages): submissions close 19:00, public leaderboard frozen from 18:30; winner's code (<3000 lines) is
  re-checked and must regenerate the solution.

## Experiment ledger (live view: `python dashboard.py` -> http://localhost:8765/)
| time | experiment | result | decision |
|---|---|---|---|
| 14:46 | LNS x4 from Codex warm starts (833.84M) | 833.11M after 25 min (-0.73M) | keep; saturating |
| 15:05 | global split+reuse polish (Codex support_only, 90s) | strikes -11k, reliability -97k | keep as final polish step |
| 15:05 | full-catalogue (all routes) 14-flow neighbourhood, 40s | small improvement on strikes | added to LNS mix |
| 15:11 | round 4: 4 LNS from team best 833.01M (rel x2, strikes, alliances), ALNS + topk + full-catalogue | running | - |
| 15:13 | ILS kick (SA 30s + global polish) after 12 fails, alliances, 5th process | running | compare vs plain LNS on alliances |
| 15:25 | big mode (20-40% exact repair) smoke 3x120s | 1 win (-109); HiGHS times out on 60-70 flows | kept at 15-25%, 110s |
| 15:29 | tail mode (scenario cost attribution drivers) smoke 3x120s | 1 win (-82) | kept |
| 15:33 | crossover (elite pool) | pool = 2 diverse members; 60-flow repair 104s, no child | dropped (0 wins) |
| 15:34 | **lexicographic repair** (max, then tied top scenarios) + plateau acceptance | strict wins + plateau moves (top-8 -145k) | main bet: all families |
| 15:38 | lex portfolio 6 slots (portfolio.py) | 832.31M -> 831.99M in 15 min; alliances slot 43 wins | running until 17:30 |
| 15:47 | Codex solve() from scratch, strikes, 2 seeds x300s | 480.2M both (far worse basin) | rejected |
| 15:53 | wide lex workers (rel size 50, strikes 45, sub 60) | running | A/B vs 25-flow slots |
| 16:24 | what-if simulation (move each big SPOT flow onto a robust pair, relaxed eval) | f54 (3401 TEU, 17->3) alone: -6.3M rel / -9.1M str / -6.1M all; blockers = ports 38/46 manut + carrier sites(m) | root cause of the leaderboard gap |
| 16:25 | `swap`/`open` scopes (close lightest (m,t), open best-gain (m,t')) | small wins only | dropped |
| 16:30 | `bigspot` scope (big SPOT flow + random port holders) | small wins | dropped |
| 16:35 | **`unlock` scope** (big SPOT flow + port holders ranked by cheapest detour + lightest site of a needed full carrier, pc=3, rl=None, 150 s) | strikes -302k in 2 moves -> pf_5 then 267.55M; alliances -56k | main bet |
| 16:40 | forced f54 onto F+I (reliability), 47-flow repair 240 s | relaxed 279.9M but no feasible repair found | needs bigger scope: 100-flow unlock (2nd ring), 400 s |
| 17:55 | diagnosis: site 17 reaches only port 46 (17886) but 25109 TEU touch it; port-46 knapsack already near value-optimal (f54<->f61 swap ~0.9M) | the gap is structural (site sets, pairing, fees), not the hub | - |
| 18:00 | **`open_carrier.py`** (strikes never used G: 91 flows / 17.9k TEU cheaper on G; free top-45 flows by G gain, exact MILP 600 s) | strikes 266.07M -> 265.87M in 1 round | keep; try A, C, reliability C |
| 18:05 | **FINAL submitted** | 276,151,962 / 285,352,183 / 265,874,001 = **827,378,146**; zip 10 files, 2327 lines | submitted |
Team best per instance is always the min over `solutions/**/<name>.json` + `codex_teamwarmup/` (evaluator-checked);
`runs/claude/relaunch.sh ROUND` snapshots it into `solutions/team_best/` and restarts the workers.

## Notes for Codex's MILP (insights, no file edits)
1. Rerouted x^s exact & big-M-free: for affected flow f in s, continuous x^s_r with x^s_r = 0 (r failed), x^s_r >= x_r, x^s_r <= n_f u_r, sum_r x^s_r = n_f. Given integral (x,u) it is exact AND enforces survival. Unaffected flows: x^s = x.
2. y-feasibility (7),(8) and sites(m) are scenario-monotone → nominal rows only. Scenario-dependent: manut, trans, alpha (partial carrier failure; not in alliances), contract fees e^s.
3. Reuse saving is separable (B_i + A_j); within (carrier, site) the y-block can be aggregated by failure-signature groups (alliances: 1 signature per carrier → one pool per (m,t)).

## Line count (production path)
evaluator.py + baseline.py + benchmark.py — see `wc -l` at each checkpoint.
