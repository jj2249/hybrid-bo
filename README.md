# Hybrid-BO

Hybrid-BO is a Python package for
[Bayesian optimization for partially known systems using hybrid models](https://arxiv.org/abs/2603.11199).
It optimizes a system whose equations are partly known: the known equations are used
as they are, and a Gaussian process learns the unknown ones from measurements. It is
built on [CasADi](https://web.casadi.org/).

The library fits GPs and solves one iteration of BO. You write the loop.

## Install

```bash
pip install "hybrid-bo[plots] @ git+https://github.com/jj2249/hybrid-bo.git@v0.7.0"
```

`plots` adds matplotlib. The tag fixes the version, so the library can't change under your
work.

To work on the library, install [pixi](https://pixi.sh):

```bash
pixi run -e dev test    # pytest
pixi run -e dev check   # lint, format and type check
pixi run -e dev docs    # the API docs, in docs/_build/html
```

## Use

State the problem, and how to measure the system. `measure(u, rng)` returns the measured
variables as an array, and can run anything: a plant, an external simulator, or an oracle
of your own.

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


def measure(u, rng):
    return np.sin(u)


def evaluate(u, rng):
    """A one-row dataset: measure the system at u, and solve for the other states."""
    return hb.reconstruct_states(
        problem, u=u[None], measured=measure(u, rng)[None], rng=rng
    )


rng = np.random.default_rng(0)
gp = hb.gp.GP(mean=hb.gp.Zero(), kernel=hb.gp.Matern52(), noise_std=0.0)
points = hb.lhs_points(system.u, 5, rng)
initial = hb.reconstruct_states(
    problem, u=points, measured=measure(points, rng), rng=rng
)
```

**Hybrid BO** fits a GP to each unknown state. The scenarios are draws `z` and weights `w`.
Draw them once, before the loop, and keep them: nothing checks that they stay the same.

```python
dataset = initial
z = rng.standard_normal((25, system.n_y))
w = np.full(25, 1 / 25)

for _ in range(10):
    fitted = hb.fit_surrogate(system, dataset, gp=gp, rng=rng)
    acquisition = hb.hybrid_acquisition(
        problem,
        fitted,
        z,
        w,
        hb.incumbents(dataset)[-1],
        approximation=hb.SAA(),  # or hb.SoftSAA(temperature=...)
    )
    solution = acquisition.solve(rng)
    dataset = hb.extend(dataset, evaluate(solution.u, rng))
```

**Standard BO** fits one GP to the objective:

```python
dataset = initial

for _ in range(10):
    fitted = gp.fit(dataset.u, dataset.f, rng)
    acquisition = hb.standard_acquisition(problem, fitted, hb.incumbents(dataset)[-1])
    solution = acquisition.solve(rng)
    dataset = hb.extend(dataset, evaluate(solution.u, rng))
```

**A baseline** is points through the same steps, from `hb.lhs_points` or
`hb.uniform_points`:

```python
dataset = initial

for point in hb.uniform_points(system.u, 10, rng):
    dataset = hb.extend(dataset, evaluate(point, rng))
```

A fit or a `solve` raises `hb.NoSolutionError` if every start fails, and
`hb.reconstruct_states` raises `hb.RootFindingError` if the known equations have no root
or a value isn't finite. Your loop decides what to do: nothing is caught for you.

The library logs what it fits and how many starts of each solve succeed, at `INFO` on the
`hybrid_bo` logger; `logging.getLogger("hybrid_bo").setLevel(logging.INFO)` shows it.

A dataset can be saved, and extended later. Nothing records how it was produced.

```python
hb.save_dataset(dataset, "dataset.npz")
dataset = hb.load_dataset("dataset.npz")
```

Plots draw on the axes you give them: `hb.plots.convergence` and `hb.plots.regret` take
incumbent curves per label, `hb.plots.surrogate` draws a fitted GP of one input, and
`hb.plots.iteration` draws what an iteration minimized, from
`hb.plots.acquisition_on_grid`, which gives EI over a grid of `u` and a `Dataset` per scenario.
`examples/illustrative_example.py` compares hybrid and standard BO and draws both.
`plt.style.use("hybrid_bo.default")` gives the library's look; change it as any matplotlib
style, with `plt.rcParams` or a style of your own.

## License

This project is licensed under the MIT license.
Be aware that dependencies are not necessarily licensed under the MIT license.
