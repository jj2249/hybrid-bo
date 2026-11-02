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

The study and its figures are saved in `results/illustrative_example/`. Add
`--n-workers 4` to run the runs in 4 processes at once, with the same results. Each
run is saved as it finishes, so a study stopped by Ctrl-C or a crash finishes with
`pixi run experiment illustrative_example resume STUDY`.

Any script that ends with `hb.cli.main(...)` runs the same way, as
`python my_study.py [--n-runs 3 --plot ...]`. Keep your own in `scripts/`, which git
ignores; [scripts/README.md](scripts/README.md) has an example.

## Benchmarks

`benchmarks/` builds benchmark experiments by rules from an external problem library
(ADR 0012). `pixi run benchmark s2mpj HS27 [--n-runs 3 --plot ...]` fetches
[S2MPJ](https://github.com/GrattonToint/S2MPJ) at a pinned commit, builds a hybrid
problem from its CUTEst problem HS27 and runs it as a study in
`results/s2mpj/HS27/`. Any equality-constrained S2MPJ problem the rules support
runs the same way; one they don't support is refused with the reason.

## Docs

```bash
pixi run docs
```

then open `docs/_build/html/index.html`. Getting started explains what you just ran,
Writing an experiment how to set up your own problem, and Reading results what the
results mean.

## License

This project is licensed under the MIT license.
Be aware that dependencies are not necessarily licensed under the MIT license.
