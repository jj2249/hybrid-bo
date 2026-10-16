# Hybrid-BO

Hybrid-BO is a Python package for
[Bayesian optimization for partially known systems using hybrid models](https://arxiv.org/abs/2603.11199).
It optimizes a system whose equations are partly known: the known equations are used
as they are, and a Gaussian process learns the unknown ones from measurements. It is
built on [CasADi](https://web.casadi.org/).

## Install

Install [pixi](https://pixi.sh), then fork this repository and clone your fork. The
first `pixi run` installs everything the project needs.

## Run

Run every method on the illustrative example of the paper and plot the results:

```bash
pixi run experiment illustrative_example --n-runs 3 --plot
```

The study and its figures are saved in `results/illustrative_example/`.

## Docs

```bash
pixi run docs
```

then open `docs/_build/html/index.html`. Getting started explains what you just ran,
Writing an experiment how to set up your own problem, and Reading results what the
results mean.

## Citing

If you use Hybrid-BO in your work, please cite the paper,
[arXiv:2603.11199](https://arxiv.org/abs/2603.11199).

## License

This project is licensed under the MIT license.
Be aware that dependencies are not necessarily licensed under the MIT license.
