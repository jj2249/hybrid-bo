# GP Embedded NLPs
This library is a Python package for uncertainty aware nonlinear programming in engineering. We embed data-driven relationships into the nonlinear program (NLP). The relationships represent epistemic uncertainty and missing knowledge in our system. The resulting uncertainty aware *hybrid* NLP is a stochastic problem. We implement the models in [Kutschat et al. (2026)](https://arxiv.org/abs/2603.11199) using [CasADi](https://web.casadi.org/).

With the library you can:
- Define an NLP from known mechanistic equations and unknown relationships
- Define GP priors and fit them to data
  - by maximum likelihood, or maximum a posteriori with the lengthscale prior of [Hvarfner et al. (2024)](https://arxiv.org/abs/2402.02229)
  - and make predictions conditioned on the data
- Embed the fitted GPs in the NLP
- Solve the resulting stochastic problem by sample-average approximation
- Choose the next experiment in Bayesian optimization by maximizing expected improvement
- Plot and save results

```math
\begin{aligned}
\min_{u \in \mathcal{U}} \quad & \mathbb{E}_\xi \left[ f(x, y, u) \right] \\
\text{s.t.} \quad & g(x, y, u) = 0 \quad \text{a.s.} \\
& y = \mu_h(x, u) + L_h(x, u)\,\xi \quad \text{a.s.}
\end{aligned}
```
## Getting started
### Install
With [pip](https://pip.pypa.io/):
```bash
pip install "hybrid-bo @ git+https://github.com/jj2249/hybrid-bo.git@v0.3.0"
```
With [pixi](https://pixi.sh/):
```bash
pixi add --pypi "hybrid-bo @ git+https://github.com/jj2249/hybrid-bo.git@v0.3.0"
```
With [uv](https://docs.astral.sh/uv/):
```bash
uv add "hybrid-bo @ git+https://github.com/jj2249/hybrid-bo.git@v0.3.0"
```
### Building docs
The API reference is not publicly hosted at the moment. It can be built locally by cloning the repository and running

```bash
pixi run -e dev docs
```

then opening `docs/_build/html/autoapi/index.html`.

## Using the library

The illustrative example: the known equation is `x + exp(x) = y`, and the unknown one, `y = sin(u)`, appears only in the measurements.

### Define the hybrid NLP

`g` and `f` take the variables `v` and read them by name.

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

### Collect data

Pass the measured states; `g` is solved for the rest.

```python
rng = np.random.default_rng(0)
u_samples = hb.lhs_points([u], 2, rng)
data = hb.Dataset.from_measurements(
    nlp, u=u_samples, measured={y: np.sin(u_samples[:, 0])}, rng=rng
)
```

### Define and fit a GP

Use `Matern52(lengthscale_prior=True)` for a MAP fit.

```python
gp = hb.gp.GP(kernel=hb.gp.Matern52(), mean=hb.gp.Zero(), noise_std=0.0)
fitted = gp.fit(data, inputs=[u], output=y, rng=rng)
mean, std = fitted.predict(np.array([[0.0], [1.0]]))
```

### Embed the GP and solve with SAA

```python
surrogate = [fitted]
saa = hb.SAA(rng.standard_normal((25, 1)))
solution = nlp.solve(saa, rng, surrogate=surrogate)
print(solution.u, solution.value)
```

### Bayesian optimization with expected improvement

Pass an acquisition function as the `objective`. You write the loop and the measurement.

```python
for _ in range(10):
    surrogate = [gp.fit(data, inputs=[u], output=y, rng=rng)]
    solution = nlp.solve(
        saa,
        rng,
        surrogate=surrogate,
        objective=hb.obj.EI(data.incumbents[-1]),
        unknown_states_as_variables=False,
    )
    new = hb.Dataset.from_measurements(
        nlp, u=solution.u, measured={y: np.sin(solution.u)}, rng=rng
    )
    data = data.extend(new)
```

Standard BO: no `x`, and `f` is the unknown state. `hb.Exact()` uses closed-form EI.

```python
fhat = hb.Var("fhat", start=(-10.0, 10.0))
standard_nlp = hb.HybridNLP(u=[u], y=[fhat], f=lambda v: v.fhat)
observed = hb.Dataset.from_measurements(
    standard_nlp, u=data.u, measured={fhat: data.f}, rng=rng
)
standard_surrogate = [gp.fit(observed, inputs=[u], output=fhat, rng=rng)]
standard_solution = standard_nlp.solve(
    hb.Exact(),
    rng,
    surrogate=standard_surrogate,
    objective=hb.obj.EI(observed.incumbents[-1]),
)
```

`examples/illustrative_example.py` runs both loops over several seeds and compares them.

### Plot and save results

Plotting needs matplotlib.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, sharex=True)
hb.plots.iteration(nlp, solution, axes, data)
fig.savefig("iteration.png")
```

Saving won't overwrite an existing file.

```python
data.save("dataset.npz")
data = hb.Dataset.load("dataset.npz")
```

## A fixed configuration

Values that enter `g` or `f` and are known for each experiment but not chosen, such as a reactor's inlet flow, go in `c`. Give them for each measured point and for the next experiment.

```python
a = hb.Var("a")  # known, never sampled: needs no bounds or start
configured_nlp = hb.HybridNLP(
    u=[u],
    c=[a],
    x=[x],
    y=[y],
    g=lambda v: [v.x + v.a * cas.exp(v.x) - v.y],
    f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4),
)
configured = hb.Dataset.from_measurements(
    configured_nlp,
    u=u_samples,
    c=np.array([[1.0], [1.5]]),  # each experiment's configuration, (n, K)
    measured={y: np.sin(u_samples[:, 0])},
    rng=rng,
)
surrogate = [gp.fit(configured, inputs=[u], output=y, rng=rng)]
solution = configured_nlp.solve(saa, rng, surrogate=surrogate, c=np.array([1.2]))
```

A problem without a configuration never mentions `c`.

## Errors and logs

A fit or a `solve` raises `hb.NoSolutionError` if every start fails, and
`Dataset.from_measurements` raises `hb.RootFindingError` if `g` has no root or a value
isn't finite.

The library logs at `INFO` on the `hybrid_bo` logger:
`logging.getLogger("hybrid_bo").setLevel(logging.INFO)` shows it.

## Develop

To work on the library, install [pixi](https://pixi.sh):

```bash
pixi run -e dev test    # pytest
pixi run -e dev check   # lint, format and type check
pixi run -e dev docs    # the API docs, from docs/_build/html/autoapi/index.html
```

## License

This project is licensed under the MIT license.
Be aware that dependencies are not necessarily licensed under the MIT license.
