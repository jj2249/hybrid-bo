# Hybrid-BO

Hybrid-BO is a package to perform [Bayesian optimization for partially known systems using hybrid models](https://arxiv.org/abs/2603.11199).  
It is mainly based on [CasADi](https://web.casadi.org/).

## Installation

Currently, we do not provide Hybrid-BO in a package repository like [PyPI](https://pypi.org/).  
Instead, you have to clone the repository or download the code and perform a local installation of the package.

### pyproject.toml

The file [`pyproject.toml`](./pyproject.toml) can be used by installation tools to install the package and its dependencies.  
It defines the core dependencies of the package (see `[project.dependencies]`).  
Additionaly, it defines multiple optional dependency groups (see `[dependency-groups]`).

### pip

The package can be installed via pip.  
If you want to install the package and its core dependencies, in a terminal, change directory to this project directory and type:

```bash
pip install .
```

If you additionaly want to install a dependency group, type:

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

To perform Bayesian optimization for partially known systems using hybrid models, you have to call the function
`do_hybrid_bo()` in [`./src/hybrid_bo/routines/hybrid_bo.py`](./src/hybrid_bo/routines/hybrid_bo.py)).

To be able to compare the performance with other approaches, we also provide

* Standard Bayesian optimization (function `do_standard_bo()` in [`./src/hybrid_bo/routines/standard_bo.py`](./src/hybrid_bo/routines/standard_bo.py))
* Latin hypercube sampling (function `do_latin_hybercube_sampling_bo()` in [`./src/hybrid_bo/routines/latin_hypercube_sampling.py`](./src/hybrid_bo/routines/latin_hypercube_sampling.py))
* Uniform sampling (function `do_uniform_sampling_bo()` in [`./src/hybrid_bo/routines/uniform_sampling.py`](./src/hybrid_bo/routines/uniform_sampling.py))

For the function `do_hybrid_bo()`, you need to provide

* A `Problem` (see [`./src/hybrid_bo/problem.py`](./src/hybrid_bo/problem.py)) as in equation (4) in the [publication](https://arxiv.org/abs/2603.11199)
* A `Config` (see [`./src/hybrid_bo/config.py`](./src/hybrid_bo/config.py))
* A `GP` (Gaussian process, see [`./src/hybrid_bo/gp.py`](./src/hybrid_bo/gp.py))
* An `Optimizer` (see [`./src/hybrid_bo/optimizers.py`](./src/hybrid_bo/optimizers.py))
* Initial points

A `GP` needs

* A `Kernel` (see [`./src/hybrid_bo/kernels.py`](./src/hybrid_bo/kernels.py))
* A `Mean` (see [`./src/hybrid_bo/means.py`](./src/hybrid_bo/means.py))
* A standard deviation `Parameter` (see [`./src/hybrid_bo/parameterization.py`](./src/hybrid_bo/parameterization.py))
* An `Optimizer` (see [`./src/hybrid_bo/optimizers.py`](./src/hybrid_bo/optimizers.py))

All hyperparameters of a `GP` are objects of `Parameter` (see [`./src/hybrid_bo/parameterization.py`](./src/hybrid_bo/parameterization.py)).

**Important**: Currently, only one unknown equation is supported. For more information, see the [publication](https://arxiv.org/abs/2603.11199).

## Examples

Currently, we have implemented two examples

* [`./examples/illustrative_example/`](./examples/illustrative_example/)
* [`./examples/flash_unit/`](./examples/flash_unit/)

The examples are described in detail in the [publication](https://arxiv.org/abs/2603.11199).  
You can run the examples with [`./examples/main.py`](./examples/main.py).  
Before running the examples, make sure you have installed the dependency group `examples`, see [`./pyproject.toml`](./pyproject.toml).

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
