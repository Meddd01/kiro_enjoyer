# Optimization handoff

- Environment: Python 3.12 and HiGHS 1.16.0 with NumPy. Gurobi 14.0.0 was installed and tested, but its active license rejected a 3,000-variable MIP.
- Entry points: `common.load_instance(path)`, `milp_solver.solve(instance, initial_solution=None, time_limit=120.0, seed=0)`, and `reuse.optimize_reuse(instance, solution, time_limit=20.0, seed=0)`. Install `highspy==1.16.0` for these files. The allocation and reuse APIs accept the official raw dictionaries.
- The engine starts with a robust no-reuse MILP and polishes quantities and integer reuse. Large instances benefit from solving small blocks first. Reliability also benefited from a broad solve over always-surviving routes. Candidate and neighborhood MILP bounds apply to the loaded model only.
- Integer reuse groups routes on each side by carrier, site, truck-leg cost, and failure mask. The optimizer restores individual pairs with exact integer transportation. On matched neighborhoods the compressed and direct models agreed on all three scores. Solver times in seconds were 14.93 → 2.05 on Alliances, 5.89 → 2.11 on Reliability, and 10.51 → 12.34 on Strikes.
- For the heuristics track, proposed API: add keyword-only `free_flows=None` to `milp_solver.solve`. A set of flow IDs would freeze every other nominal allocation exactly and allow reuse to change globally. The mode requires a feasible incumbent and an empty set means reuse only. The positional API is unchanged while confirmation is pending. The CLI already supports the same operation through `--free-flows`.
- The team evaluator is imported automatically when `evaluator.py` is available. The optimization tests and model checks are available independently of it. Production source in this branch is 718 physical Python lines, plus 114 lines of development tests. The 3,000-line limit applies to the combined generating source.

## Commands

```text
python -m unittest discover -s tests -v
python milp_solver.py instances/alliances.json allocation.json --time 60 --seed 0
python milp_solver.py instances/alliances.json improved.json --initial allocation.json --time 15 --fixed-support
python milp_solver.py instances/alliances.json neighborhood.json --initial improved.json --time 26 --free-flows 0 1
```

Run the same commands with the relevant instance path. Time-limited incumbents can vary across machines. Any candidate intended for submission should be checked through the independent team evaluator. The local runs keep each incumbent and the exact argument list separately from the source branch.
