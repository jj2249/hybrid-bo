# GP Embedded NLPs
This library is a Python package for uncertainty aware nonlinear programming in engineering. We embed data-driven relationships into the nonlinear program (NLP). The relationships represent epistemic uncertainty and missing knowledge in our system. The resulting uncertainty aware *hybrid* NLP is a stochastic problem. We implement the models in [Kutschat et al. (2026)](https://arxiv.org/abs/2603.11199) using [CasADi](https://web.casadi.org/).

The library provides the means to:
- Define a nonlinear program in terms of known and unknown mechanistic equations
- Define GP priors
  - Maximum likelihood and maximum a posteriori parameter estimation with the lengthscale prior of [Hvarfner et al. (2024)](https://arxiv.org/abs/2402.02229)
  - Condition GP on data and make conditional predictions
- Embed the GP surrogate models into the NLP
- Solve the resulting stochastic problem using sample-average approximation discretisation
- Solve acquisition problems in a bayesian optimisation context using expected improvement
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

We walk through an illustrative example with one decision variable `u`, one mechanistic state `x` and one unknown state `y`. The known equation is `x + exp(x) = y`, the unknown one is `y = sin(u)`, and the objective is `f = (6x - 2)^2 sin(12x - 4)`. The library never sees `sin(u)`: it only appears in our measurements.

### Define the hybrid NLP

Declare each variable with `hb.Var`, giving bounds for the decision variables and a range of starting points for the states. Then create a `HybridNLP` from the variables, the known equations `g` and the objective `f`. `g` and `f` are Python callables that take `v`, the variables, and read each one by name. CasADi traces them once, so derivatives are exact.

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

A `Dataset` holds the measured points. `Dataset.from_measurements` takes `u` and the measured states, solves `g` for the other states, and computes `f`. `measured` maps each measured variable to its values `(n,)`: any P of the states `x` and `y`, where P is the number of unknown states. Here we measure `y` at two points from a Latin hypercube. A system that measured `x` instead would pass `measured={x: ...}` and have `y` solved from `g`.

```python
rng = np.random.default_rng(0)
u_samples = hb.lhs_points([u], 2, rng)
data = hb.Dataset.from_measurements(
    nlp, u=u_samples, measured={y: np.sin(u_samples[:, 0])}, rng=rng
)
```

### Define and fit a GP

A `GP` is a prior: a kernel, a mean and a known noise level. `GP.fit` learns the hyperparameters by maximum likelihood and returns a `FittedGP`, conditioned on the data, for the given `inputs` and `output`. With `Matern52(lengthscale_prior=True)` the lengthscales are found by maximum a posteriori estimation instead. `FittedGP.predict` gives the conditional mean and standard deviation at new points.

```python
gp = hb.gp.GP(kernel=hb.gp.Matern52(), mean=hb.gp.Zero(), noise_std=0.0)
fitted = gp.fit(data, inputs=[u], output=y, rng=rng)
mean, std = fitted.predict(np.array([[0.0], [1.0]]))
```

### Embed the GP and solve with SAA

`HybridNLP.solve` embeds the fitted GPs, one per unknown state, as `y = mu_h(x, u) + L_h(x, u) xi`, and solves the resulting stochastic problem. `hb.SAA(z)` discretizes it by sample-average approximation over scenarios `z`, an array `(S, P)` of draws of `xi`, equally weighted unless you give weights `w`. By default `solve` minimizes the mean objective, E[f].

```python
surrogate = [fitted]
saa = hb.SAA(rng.standard_normal((25, 1)))
solution = nlp.solve(saa, rng, surrogate=surrogate)
print(solution.u, solution.value)
```

The returned `Solution` holds `u` `(N,)`, `x` `(S, M)` and `y` `(S, P)`, one row per scenario, and the minimized `value`, along with the `method`, `surrogate` and `objective` it was solved by.

### Bayesian optimization with expected improvement

Pass an acquisition function as the `objective` to choose the next experiment instead. `hb.obj.EI(incumbent)` gives expected improvement over the best measured `f`, and `hb.obj.SoftLogEI` a smooth log version that is easier to optimize. The library solves one iteration; you write the loop and the measurement. Here we keep the scenarios fixed and run ten iterations of hybrid BO.

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

Standard black-box BO is the special case with no mechanistic states and a single unknown state that is the measured objective itself, `f = y`. Solving with `hb.Exact()` uses closed-form EI rather than scenarios.

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

`hb.plots` draws a solution over a grid of `u` on matplotlib axes you provide. For one `u` and an EI objective, that is three panels: the surrogate, `f`, and the acquisition. `hb.plots.iteration` also marks the measured points and the incumbent. Plotting needs matplotlib, which the library doesn't install.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, sharex=True)
hb.plots.iteration(nlp, solution, axes, data)
fig.savefig("iteration.png")
```

Datasets save to and load from `.npz` files. Saving fails if the file already exists.

```python
data.save("dataset.npz")
data = hb.Dataset.load("dataset.npz")
```

## A fixed configuration

Some values in the known equations change from one experiment to the next but aren't yours to choose. In a reactor, the inlet flow rate and composition are measured for each run, and they enter the mass balances. These values are the configuration `c`, (K,). The decision variables `u` are free, and `solve` chooses them. The configuration is fixed: you give it for each measured point, and for the experiment you are planning next. `g` and `f` read it by name, as they read the other variables. The GPs don't take it as an input, because `h` depends on x and u, and the configuration reaches it through the x that `g` solves for.

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
