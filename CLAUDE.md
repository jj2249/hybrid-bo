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

- Write for a scientist reading the code: names from the paper, and long functions that
  read like the math rather than helpers that hide it.
- Frozen dataclasses for data. Methods are welcome when they read like a property or return
  a new value (`system.states`, `fitted.predict(X)`). Nothing mutates: no void `gp.fit()`.
  Randomness is an explicit `rng`.
- Rows are points: a point is `(n,)`, a set is `(n_points, n)`, `f` is `(n_points,)`.
- Every function gets a docstring. American spelling.

Decisions to keep, so they aren't undone:

- `h` is not part of the problem: its form exists only in a user's oracle, and the library
  always solves `g` for the unmeasured states.
- Equations are callables traced once by CasADi, for exact derivatives. No finite
  differences.
- GPs are data: frozen dataclasses from a closed set of kernels, each holding its own math
  as methods. No custom kernels.
- A default may set how hard we search, never what we search for: `acquisition`,
  `temperature` and `noise_std` are required.
- The library has no loop or runner. The user writes the loop and the measurement.
- Figures evaluate the acquisition problem over a grid from the stored solution, so they
  show what was minimized, on the solver's branch.
