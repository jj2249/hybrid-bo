# Experiments

Each file here is an experiment: a problem, how it is measured, and the methods to
compare, passed to the library's command line, `hb.cli.main`. For your own
throwaway studies, use `scripts/` instead ([scripts/README.md](../scripts/README.md)).

- `illustrative_example.py`: the illustrative example of the
  [paper](https://arxiv.org/abs/2603.11199), with an oracle.
- `illustrative_example_2d.py`: the illustrative example with two decision
  variables, y = sin(u1) cos(u2), with an oracle, comparing SAA-EI, Lin-EI,
  SoftSAA-EI, standard BO and Latin hypercube sampling. Its iteration figures show every 2D panel,
  including the GP of y.
- `rosenbrock.py`: the 2D Rosenbrock function as a composite function, with no known
  equations and two unknown states (adapted from Astudillo & Frazier,
  [arXiv 1906.01537](https://arxiv.org/abs/1906.01537), App. D.2), with an oracle,
  comparing SAA-EI, SoftSAA-EI, standard BO and Latin hypercube sampling.
- `langermann.py`: the Langermann function as a composite function, with no known
  equations and five unknown states (Astudillo & Frazier, App. D.1), with an oracle,
  comparing the same methods.

With more than one unknown state, hybrid BO's iteration figures are drawn without
the GP panels.

```bash
pixi run experiment <name> [--n-runs 3 --plot ...]   # run a study; --help lists the flags
pixi run plot <name> [STUDY]                         # plot the newest (or named) study
pixi run experiment <name> plot [STUDY] --plot-runs all   # iteration figures of every run
```

A study's runs are its tasks, each run in its own process, `--n-workers` at a time
(default 1), with the same results whatever the number. A study shows its progress on
the console and saves it as `study.log` in its folder, and each run's lines in
`logs/<method>/run-000.log`. Each run is saved as it finishes, so a study stopped by
Ctrl-C or a crash keeps its finished runs and prints the command that finishes it:

```bash
pixi run experiment <name> resume STUDY [--n-workers 4] [--plot]
```

A `--name` that is already taken, by a saved or an unfinished study, is refused
before anything runs.

On a cluster, `prepare` writes a study's task list without running anything and
prints its number of tasks, each task runs as `run-task STUDY --task-index K` (e.g.
one SLURM array element each, on one core with `OMP_NUM_THREADS=1`), and `resume`
assembles the study. `docs/running_large_studies.rst` has a full `sbatch` script.

To write your own, see
[Writing an experiment](../docs/writing_an_experiment.rst) (build the docs with
`pixi run docs`).

## With Benchopt

This folder is also a [Benchopt](https://benchopt.github.io) benchmark, which runs the
same experiments and their configurations, unchanged:

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

The report has `regret` (the median regret over the evaluations) and `summary` (each
method's runs, complete runs and final regret). A run that fails with an unexpected
exception prints its traceback, is missing from the results and doesn't change the
exit code, so check the runs in `summary`.

- `objective.py`: scores a run, one row per evaluation.
- `datasets/experiment.py`: every experiment in this folder, found by file name.
- `solvers/method.py`: every configuration of the experiments, by label.
- `plots/`: the `regret` and `summary` views.
- `benchmark_utils/`: what they share, including the hash of the code.
