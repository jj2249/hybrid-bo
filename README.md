# Hybrid-BO

Hybrid-BO is a Python package for
[Bayesian optimization for partially known systems using hybrid models](https://arxiv.org/abs/2603.11199).
It minimizes `f(x, y, u)` over the decision variables `u`, where the known equations
`0 = g(x, y, u)` are used as they are, and a Gaussian process learns the unknown ones,
`y = h(x, u)`, from measurements. It is built on [CasADi](https://web.casadi.org/).

The library fits GPs and solves the hybrid model once. You write the loop and the
measurement.

## Install

```bash
pip install "hybrid-bo @ git+https://github.com/jj2249/hybrid-bo.git"
```

Add `@<tag>` to the URL to fix the version, so the library can't change under your work.

## The illustrative example

The known equation is `x + exp(x) = y`, and the unknown one is `y = sin(u)`. State what you
know as a `HybridNLP`: the variables, `g`, and `f`. Each is read from `v` by name.

```python
import casadi as cas
import numpy as np

import hybrid_bo as hb

u = hb.Var("u", lower=-2.0, upper=2.0)
x = hb.Var("x", start=(-2.5, 1.0))
y = hb.Var("y", start=(-2.0, 2.0))
nlp = hb.HybridNLP(
    u=[u],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],
    f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4),
)
```

The measurement is yours: a plant, a simulator, or here an oracle. It measures `y` at `u`,
and `Dataset.from_measurements` solves `g` for `x` and computes `f`. `u` and `y` may be a
point or a set of points.

```python
def measure(u, rng):
    return hb.Dataset.from_measurements(nlp, u=u, y=np.sin(u), rng=rng)


rng = np.random.default_rng(0)
data = measure(hb.lhs_points([u], 2, rng), rng)
```

A GP of each unknown state is fitted to the data. It records its `inputs` and `output`.

```python
gp = hb.gp.GP(kernel=hb.gp.Matern52(), mean=hb.gp.Zero(), noise_std=0.0)
surrogate = [gp.fit(data, inputs=[u], output=y, rng=rng)]
```

## Minimize E[f] with SAA

`hb.SAA(z)` averages over the scenarios `z`, a row per scenario and a column per unknown
state, with equal weights unless you give `w`. The default objective is the mean, so this
minimizes the expected `f` of the hybrid model:

```python
saa = hb.SAA(rng.standard_normal((25, 1)))
solution = nlp.solve(saa, rng, surrogate=surrogate)
print(solution.u, solution.objective)
```

`solution` holds `u` (N,), `x` (S, M) and `y` (S, P), a row per scenario, and the
`objective`.

## Hybrid BO

Draw the scenarios once, before the loop, and keep them. Each iteration fits the GP,
minimizes SoftEI over the scenarios, and measures the result:

```python
saa = hb.SAA(rng.standard_normal((25, 1)))
for _ in range(10):
    surrogate = [gp.fit(data, inputs=[u], output=y, rng=rng)]
    solution = nlp.solve(
        saa,
        rng,
        surrogate=surrogate,
        objective=hb.obj.SoftEI(data.incumbents[-1], temperature=0.05),
        unknown_states_as_variables=False,
    )
    data = data.extend(measure(solution.u, rng))
```

`hb.obj.EI(incumbent)` is the plain expected improvement.

## Standard BO

Standard BO is a problem with nothing known: one unknown state, the measured `f`, which
is the objective. `hb.Exact()` gives EI in closed form, with no scenarios.

```python
fhat = hb.Var("fhat", start=(-10.0, 10.0))
standard_nlp = hb.HybridNLP(u=[u], y=[fhat], f=lambda v: v.fhat)
observed = hb.Dataset.from_measurements(
    standard_nlp, u=data.u, y=data.f[:, None], rng=rng
)
surrogate = [gp.fit(observed, inputs=[u], output=fhat, rng=rng)]
solution = standard_nlp.solve(
    hb.Exact(), rng, surrogate=surrogate, objective=hb.obj.EI(data.incumbents[-1])
)
```

`examples/illustrative_example.py` runs both loops over several seeds and compares them.

## Errors, logs and saving

A fit or a `solve` raises `hb.NoSolutionError` if every start fails, and
`Dataset.from_measurements` raises `hb.RootFindingError` if `g` has no root or a value
isn't finite. Nothing is caught for you.

The library logs at `INFO` on the `hybrid_bo` logger:
`logging.getLogger("hybrid_bo").setLevel(logging.INFO)` shows it.

```python
data.save("dataset.npz")
data = hb.Dataset.load("dataset.npz")
```

## Develop

To work on the library, install [pixi](https://pixi.sh):

```bash
pixi run -e dev test    # pytest
pixi run -e dev check   # lint, format and type check
pixi run -e dev docs    # the API docs, in docs/_build/html
```

## License

This project is licensed under the MIT license.
Be aware that dependencies are not necessarily licensed under the MIT license.
