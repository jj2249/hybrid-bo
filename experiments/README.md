# Experiments

Each file here is an experiment: a problem, how it is measured, and the methods to
compare. This folder is also a [Benchopt](https://benchopt.github.io) benchmark, which
runs them.

- `illustrative_example.py`: the illustrative example of the
  [paper](https://arxiv.org/abs/2603.11199), with an oracle.
- `illustrative_example_2d.py`: the illustrative example with two decision
  variables, y = sin(u1) cos(u2), with an oracle, comparing SAA-EI, Lin-EI,
  SoftSAA-EI, standard BO and Latin hypercube sampling.
- `rosenbrock.py`: the 2D Rosenbrock function as a composite function, with no known
  equations and two unknown states (adapted from Astudillo & Frazier,
  [arXiv 1906.01537](https://arxiv.org/abs/1906.01537), App. D.2), with an oracle,
  comparing SAA-EI, SoftSAA-EI, standard BO and Latin hypercube sampling.
- `langermann.py`: the Langermann function as a composite function, with no known
  equations and five unknown states (Astudillo & Frazier, App. D.1), with an oracle,
  comparing the same methods.

## Running

```bash
pixi run benchmark -r 3 -j 4                                     # every experiment, 3 runs each
pixi run benchmark -r 3 -d "experiment[module=rosenbrock]"       # one experiment
pixi run benchmark -r 3 -s "method[label=hybrid_bo]"             # one method
```

`-r` is the number of runs and `-j` the number of processes, and `--seed` sets the
study's seed. Run i of every method starts from the same initial points. Add
`--no-plot --no-html` to skip the report, which takes much longer than a small study.

Results go to `outputs/benchopt_run_<time>.parquet`, one row per evaluation, and each
run in full to `outputs/runs/<module>/<code>/<label>/seed-<seed>.json`, which
`hb.load_run` reads. A stopped study finishes by running the same command again: Benchopt
skips the runs it has. `code` hashes the experiments and the library, so a change to
either runs everything again rather than returning old results.

The report has `regret` (the median regret over the evaluations, with its interquartile
range) and `summary` (each
method's runs, complete runs and final regret). A run that fails with an unexpected
exception prints its traceback, is missing from the results and doesn't change the
exit code, so check the runs in `summary`.

## Adding an experiment

A new file here is found by its name. It needs these names at its top level:

- `problem` and `measure`;
- `optimum`, the oracle optimum that regret is measured from;
- `methods`, each method by its label;
- `n_evaluations` and `n_initial`, every method's budget.

Benchopt only imports the file, so anything it should run directly goes under
`if __name__ == "__main__":`.

## The benchmark

- `objective.py`: scores a run, one row per evaluation.
- `datasets/experiment.py`: every experiment in this folder, found by file name.
- `solvers/method.py`: every method of the experiments, by label.
- `plots/`: the `regret` and `summary` views.
- `benchmark_utils/`: what they share, including the hash of the code.
