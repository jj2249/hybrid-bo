# hybrid-bo

Bayesian optimization of partially known systems with hybrid models
([arXiv 2603.11199](https://arxiv.org/abs/2603.11199)): minimize `f(x, y, u)` subject to
known equations `0 = g(x, y, u)` and unknown equations `y = h(x, u)`, which a GP learns.

Keep it small. It has one maintainer. The library fits GPs with CasADi and solves one
iteration of hybrid or standard BO; the user writes the loop (see README.md). Prefer
deleting code to adding it, and report the net lines of a change.

```bash
pixi run -e dev test                      # pytest, seconds
pixi run -e dev test --update-baselines   # only when results are meant to change
pixi run -e dev check                     # ruff lint (--fix), format, ty
pixi run -e dev docs                      # intro page and API docs
```

`tests/test_regression.py` compares a plain loop with `tests/data/regression/*.json`.

- Frozen dataclasses for data, plain functions for behavior. Functions take what they use
  and return new values. Randomness is an explicit `rng`.
- Rows are points: a point is `(n,)`, a set is `(n_points, n)`, `f` is `(n_points,)`.
- Every function gets a docstring. American spelling.
