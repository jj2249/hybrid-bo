# hybrid-bo

Bayesian optimisation for partially known systems using hybrid models
([arXiv 2603.11199](https://arxiv.org/abs/2603.11199)). The system is
`f(u, x)` subject to known equations `h_known(u, x) = 0` and one unknown equation
`h_unknown`, which is learned by a GP. The acquisition problem is solved over
GP samples using CasADi. Standard BO, Latin hypercube sampling and uniform
sampling are included as baselines.

The roadmap is in [ROADMAP.md](ROADMAP.md). Check it before starting work, and
tick items off in the PR that completes them.

## Commands

Environments are managed with pixi. `dev` includes everything.

```bash
pixi install -e dev
pixi run -e dev check        # lint (with --fix) + format + typecheck
pixi run -e dev lint | format | typecheck
pixi run -e dev docs         # sphinx API docs -> docs/_build/html
pixi run -e dev setup        # install pre-commit hooks
pixi run -e quality pre-commit run --hook-stage manual ty   # ty hook is manual for now
pixi run -e dev python examples/main.py   # example and routine are hardcoded at the top of main.py
```

There are no tests yet (ROADMAP stage 2).

## Architecture

- `routines/`: the four entry points. `do_hybrid_bo`, `do_standard_bo`,
  `do_latin_hypercube_sampling` and `do_uniform_sampling` all have the signature
  `(problem, config, gp, u_initial, [optimizer], results_dir, create_plots)`.
  They return `None` and pickle `ResultsBO`/`ResultsGP` into `results_dir`.
  Most of the loop is duplicated across the four files.
- `problem.py`: `Problem` holds the CasADi functions `f`, `h_known`, `h_unknown`
  and the bounds. It evaluates the "true" system with multistart `fsolve`.
  Examples subclass it and override the non-`_original` methods.
- `gp.py`, `kernels.py`, `means.py`: an exact GP (Cholesky) with kernels and
  means. `kernel.k` uses the current parameter values, and `k_variable` is the
  symbolic version used for training.
- `parameterization.py`: `Parameter` (CasADi symbol, value, bounds, transform)
  and `Parameterized`, the base of GP, Kernel and Mean.
- `optimizers.py`: CasADi/IPOPT, SciPy and multistart optimisers, which return
  `OptimizationResult`.
- `array_operations/`: a dispatch layer, so the same code runs on numpy arrays
  and CasADi SX/MX/DM.
- `config.py`: `Config` is a frozen dataclass built 1:1 from an example's
  `config.yaml`.
- `examples/<name>/`: `config.yaml`, `custom_problem.py` and `user_definitions.py`
  (GP, optimiser and initial points), plus plot functions.

## Gotchas

- Shapes are mixed. Points passed to `Problem` and the optimisers are column
  vectors `(n, 1)`, while data sets are row-major `(n_points, n)`. Expect `.T`
  calls.
- Aliases in `type_aliases.py` and `kernels.py` (e.g. `RBFKernel`) are used at
  runtime (`SymbolicType.sym(...)`, `isinstance(x, CasadiType)`). Keep them as
  plain assignments, never `type X = ...` statements.
- Routines catch and print exceptions for each run. A failing run does not raise.
- Result pickles are never overwritten. Delete old results dirs before
  re-running.

## Workflow

- `refactor/initial` is the working branch. `main` stays frozen until the roadmap
  is complete. Never commit to or open PRs against `main`.
- One PR per issue into `refactor/initial`. Branch from it using the name
  `<stage>-<topic>` (e.g. `s1-hooks`).
- Agree scope in plan mode before editing.
- Before opening a PR, run `pixi run -e dev check` (plus the tests once they
  exist), then `/code-review`.
- Use the PR template. Add the `ai-assisted` label to PRs with AI-written code.
  Commits get a `Co-Authored-By` trailer.
- Do not change logic in `src/` until regression tests exist (stage 2). Treat
  non-logic and logic changes as separate PRs.
- Keep docs concise. Sphinx autoapi already covers the API, so don't duplicate it.
- When a style question comes up, ask, then record the answer under Conventions.

## Conventions

<!-- Grown incrementally: record decisions here as they are made. -->
