# Hybrid-BO
This library allows you to define and solve hybrid-NLPs
$$
\begin{aligned}
\min_{u\in\mathcal U}&\quad \mathbb E_\xi\left[f(x, y, u)\right]&\\ \text{s.t. }&\quad g(x, y, u)=0&\text{ a.s.}&\\ &\quad y = \mu_h(x, u) + L_h(x, u)\xi&\text{ a.s.}&\\
\end{aligned}
$$

We provide methods for maximum likelihood fitting of GPs and solving the NLP with the GP embedded, implemented in CasADi.

## Install

```bash
pip install "hybrid-bo @ git+https://github.com/jj2249/hybrid-bo.git@v0.1.0"
```

Change `@v0.1.0` to install another version.

## The illustrative example

The known equation is `x + exp(x) = y`, and the unknown one is `y = sin(u)`. Create a `HybridNLP`, specifying the variables along with the known model structure `g` and `f`. The functions must be python callables that take `v`, a list of variables as parameters. The variables can then be accessed by name for improved readability.

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

We provide a helper class for handling data `Dataset`. Given measured data $$\mathcal D = \{(u_i, y_i)\}_i$$, the library allows you to reconstruct the full state of the system using `Dataset.from_measurements`. This solves `g` for `x` and computes `f`.
```python
rng = np.random.default_rng(0)
u_samples = hb.lhs_points([u], 2, rng)
data = hb.Dataset.from_measurements(nlp, u=u_samples, y=np.sin(u_samples), rng=rng)
```

You can specify a GP prior and fit it to the data by specifying its `inputs` and `output`.

```python
gp = hb.gp.GP(kernel=hb.gp.Matern52(), mean=hb.gp.Zero(), noise_std=0.0)
surrogate = [gp.fit(data, inputs=[u], output=y, rng=rng)]
```

## Minimize E[f] with SAA

We provide an implementation of sample-average approximation (SAA) to solve the stochastic problem. This is accessed through `hb.SAA(z)`, where `z` is an array of sampled scenarios. The scenarios are equally weighted unless you specify `w`.
```python
saa = hb.SAA(rng.standard_normal((25, 1)))
solution = nlp.solve(saa, rng, surrogate=surrogate)
print(solution.u, solution.value)
```
You can call `HybridNLP.solve` with a discretisation method and a fitted surrogate to solve the problem. The default behaviour is to solve the problem for the mean objective, $$\mathbb E [f]$$. The returned solution holds `u` (N,), `x` (S, M) and `y` (S, P), a row per scenario, and the
minimized `value`, with the `method`, `surrogate` and `objective` it was solved by.

## Hybrid BO

Alternatively, you can specify an acquisition function as the objective to solve a Bayesian optimisation problem. This can be implemented as a loop with a fixed set of scenarios. Here, for example, we minimise -EI over the scenarios and build a sequence of BO trials.
```python
saa = hb.SAA(rng.standard_normal((25, 1)))
for _ in range(10):
    surrogate = [gp.fit(data, inputs=[u], output=y, rng=rng)]
    solution = nlp.solve(
        saa,
        rng,
        surrogate=surrogate,
        objective=hb.obj.EI(data.incumbents[-1]),
        unknown_states_as_variables=False,
    )
    new = hb.Dataset.from_measurements(nlp, u=solution.u, y=np.sin(solution.u), rng=rng)
    data = data.extend(new)
```

## Standard BO

We can recover non-hybrid problems as the case with no mechanistic states `x` and a single unknown state `y`, which is the measured objective itself, `f = y`. This is standard black-box BO. Solving with `hb.Exact()` tells the method to use closed-form EI rather than taking scenarios as in SAA.

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
isn't finite.

The library logs at `INFO` on the `hybrid_bo` logger:
`logging.getLogger("hybrid_bo").setLevel(logging.INFO)` shows it.

You can also save and load datasets. This command will fail if the file already exists.

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
