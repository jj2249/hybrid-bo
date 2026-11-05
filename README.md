# Hybrid-BO

Hybrid-BO is a Python package for
[Bayesian optimization for partially known systems using hybrid models](https://arxiv.org/abs/2603.11199).
It optimizes a system whose equations are partly known: the known equations are used
as they are, and a Gaussian process learns the unknown ones from measurements. It is
built on [CasADi](https://web.casadi.org/).

## Install

Not on PyPI yet, so install a release from the repository, with Python 3.12 or later and
git:

```bash
pip install "hybrid-bo[plots] @ git+https://github.com/jj2249/hybrid-bo.git@v0.6.0"
```

`plots` adds matplotlib, for the figures. The tag at the end fixes the version, so the
library can't change under your work: the
[releases](https://github.com/jj2249/hybrid-bo/releases) list the newer ones, and
changing the tag upgrades. Write your scripts anywhere, in a repository of your own.

To work on the library itself, install
[pixi](https://pixi.sh), then fork and clone it: the first `pixi run` installs
everything the project needs.

## Use

A script states the problem and how it is measured, then calls a method:

```python
import casadi as cas
import numpy as np

import hybrid_bo as hb

u = hb.Var("u", lower=-2.0, upper=2.0)
x = hb.Var("x", start=(-2.5, 1.0))
y = hb.Var("y", start=(-2.0, 2.0))
system = hb.System(
    u=[u],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],  # the known equation, 0 = g(x, y, u)
    h_inputs=[u],  # the unknown equation is y = h(u), which a GP learns
    measured=[y],
)
problem = hb.Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))
measure = hb.oracle_measure(system, lambda v: [cas.sin(v.u)], noise_std=0.0)

run = hb.hybrid_bo(
    problem,
    measure,
    n_initial=2,
    n_iterations=5,
    rng=np.random.default_rng(0),
    gp=hb.gp.GP(mean=hb.gp.Zero(), kernel=hb.gp.Matern52(), noise_std=0.0),
    acquisition=hb.EI(),
    approximation=hb.SAA(n_scenarios=25),
)
print(hb.incumbents(run.dataset)[-1])
```

`hb.standard_bo` and the baselines are called the same way. Pass `initial_points=`
instead of `n_initial=` to start from points of your own.

To compare methods, loop over seeds, then save and plot what you got:

```python
runs = {"hybrid_bo": [...], "standard_bo": [...]}  # a list of runs per label
hb.save_runs(runs, "results/first-try")

figure, ax = plt.subplots()
hb.plots.regret(runs, optimum, ax)
figure.savefig("regret.png")
```

`tests/illustrative_example.py` is a complete script.

With measurements you already have, you need no `measure` and no loop: solve the known
equations for the unmeasured states, fit the GPs, then solve the hybrid model once with
each GP replaced by its mean:

```python
gp = hb.gp.GP(mean=hb.gp.Zero(), kernel=hb.gp.Matern52(), noise_std=0.0)
rng = np.random.default_rng(0)
dataset = hb.reconstruct_states(
    problem, u=u, measured=measured, rng=rng
)  # a row per point
fitted = hb.fit_surrogate(system, dataset, gp=gp, rng=rng)
solution = hb.optimize_hybrid_model(problem, fitted, rng, n_starts=30)
print(solution.u, solution.f)
```

There is no uncertainty in this problem, so it trusts the GP's mean even far from the
data, and the optimum can sit where the model is only extrapolating.

## Docs

```bash
pixi run docs
```

then open `docs/_build/html/index.html`. Getting started runs a first example,
Writing an experiment explains how to set up your own problem, and Reading results what
the results mean.

## License

This project is licensed under the MIT license.
Be aware that dependencies are not necessarily licensed under the MIT license.
