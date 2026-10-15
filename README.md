# Hybrid-BO

Hybrid-BO is a package to perform [Bayesian optimization for partially known systems using hybrid models](https://arxiv.org/abs/2603.11199).  
It is mainly based on [CasADi](https://web.casadi.org/).

## Installation

Currently, we do not provide Hybrid-BO in a package repository like [PyPI](https://pypi.org/).  
Instead, you have to clone the repository or download the code and perform a local installation of the package.

### pyproject.toml

The file [`pyproject.toml`](./pyproject.toml) can be used by installation tools to install the package and its dependencies.  
It defines the core dependencies of the package (see `[project.dependencies]`).  
Additionally, it defines multiple optional dependency groups (see `[dependency-groups]`).

### pip

The package can be installed via pip.  
If you want to install the package and its core dependencies, in a terminal, change directory to this project directory and type:

```bash
pip install .
```

If you additionally want to install a dependency group, type:

```bash
pip install --group {group-name} .
```

If you want to install in editable mode, type:

```bash
pip install -e .
```

### Pixi

The package can be installed via Pixi.  
Actually, we use Pixi for dependency management.  
This is why we provide the [`./pixi.lock`](./pixi.lock) file and you find some Pixi specific stuff in [`./pyproject.toml`](./pyproject.toml).  
If you want to install the package and its core dependencies, in a terminal, change directory to this project directory and type

```bash
pixi install
```

The Hybrid-BO package will be installed in editable mode automatically.

If you additionally want to install a dependency group, type

```bash
pixi install --environment {group-name}
```

Pixi handles different dependency groups by creating different environments.

## Usage

The problem is equation (4) of the [publication](https://arxiv.org/abs/2603.11199):
minimize `f(x, y, u)` over the decision variables `u`, subject to the known equations
`0 = g(x, y, u)` and the unknown equations `y = h(x, u)`, which a GP learns from
measurements. Describe it in Python, then run a method:

```python
import casadi as cas
import numpy as np

import hybrid_bo as hb

# Problem: the variables, the known equations and the objective
u = hb.Var("u", lower=-2.0, upper=2.0)
x = hb.Var("x", start=(-2.5, 1.0))  # unbounded, so start bounds are required
y = hb.Var("y", start=(-2.0, 2.0))
system = hb.System(
    u=[u],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],  # 0 = g(x, y, u)
    h_inputs=[u],  # y = h(u)
    measured=[y],
)
problem = hb.Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))

# Experiment: here an oracle plays the real system
experiment = hb.oracle_experiment(system, oracle=lambda v: [cas.sin(v.u)])

# Method
rng = np.random.default_rng(0)
initial_points = hb.initial_points(system.u, n=2, rng=rng)
run = hb.hybrid_bo(problem, experiment, initial_points, 5, hb.HybridBOSettings(), rng)
print(run.history.f)
```

- **Methods**: `hybrid_bo` and, to compare with, `standard_bo`, `latin_hypercube_sampling`
  and `uniform_sampling`. Each returns a `Run`: the evaluated points (`run.history`),
  the fitted GPs and a record of how the run was produced. A `Run` is plain data, so it
  pickles and loads without your code.
- **Experiments**: a real application passes a function `(u, rng) -> measured` that
  returns the measured variables at `u`. `oracle_experiment` simulates one from known
  unknown equations, for test cases.
- **Settings**: `HybridBOSettings` and `StandardBOSettings` choose the acquisition
  function (`EI()` or `LCB(std_factor)`), the GP (`default_gp` unless given; see `GP`,
  `Hyper` and `hybrid_bo.kernels`), the number of scenarios and the solver starts.
- **Plots**: `hb.plots` draws a finished `Run`, so saved runs are plotted without
  running them again (see Examples).
- **Errors** are `HybridBOError` subclasses; one raised during a method carries the
  partial run as `error.run`.

**Important**: Currently, only one unknown equation is supported. For more information, see the [publication](https://arxiv.org/abs/2603.11199).

## Examples

[`./examples/illustrative_example.py`](./examples/illustrative_example.py) defines the
illustrative example of the [publication](https://arxiv.org/abs/2603.11199) and the
methods to compare. The CLI in [`./examples/main.py`](./examples/main.py) runs an
example's methods, saving the runs to `examples/results/`, and plots saved runs:

```bash
pixi run -e dev python examples/main.py run illustrative_example --n-runs 10 --seed 0
pixi run -e dev python examples/main.py plot examples/results/illustrative_example/<UTC time>
```

`run` takes `--method` (repeatable; default every method), `--n-runs`, `--seed` and
`--n-iterations`, defaulting to the example's values. `plot` saves `convergence.png`
(the incumbents of every method's runs) and one figure per method of `--run`'s
iterations, showing `--iteration` (default: run 0, the last iteration), next to the
runs. `--show` opens them, each with a slider over the iterations. In your own
scripts, `hb.plots.convergence(runs)` and
`hb.plots.iterations(run, problem, rng, settings, experiment)` draw the same figures;
they need matplotlib (in the `examples` and `dev` pixi environments). Figures use the
repo's matplotlib style, `hb.plots.STYLE` (`src/hybrid_bo/hybrid_bo.mplstyle`).

[`./examples/flash_unit/`](./examples/flash_unit/) still targets the old API (tag
`pre-refactor`) until it is ported.

## API documentation

You can create an API documentation locally.  
For that, make sure you have installed the dependency group `docs`, see [`./pyproject.toml`](./pyproject.toml).  
In a terminal, type:

```bash
cd ./docs/
sphinx-build -b html . _build
```

Now, there exists a file `./docs/_build/index.html` that contains the API documentation.

## License

This project is licensed under the MIT license.  
Be aware that dependencies are not necessarily licensed under the MIT license.
