# Experiments

Each file here is an experiment: a problem, how it is measured, and the methods to
compare. `cli.py` is the shared command line, not an experiment.

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
pixi run experiment <name> resume STUDY [--n-workers 4]
```

A `--name` that is already taken, by a saved or an unfinished study, is refused
before anything runs.

On a cluster, `prepare` writes a study's task list without running anything, each
task runs as `run-task STUDY --task-index K`, and `resume` assembles the study:

```bash
pixi run experiment langermann prepare --name sweep      # prints the number of tasks
sbatch --array=0-17 --wrap \
    'pixi run experiment langermann run-task sweep --task-index $SLURM_ARRAY_TASK_ID'
pixi run experiment langermann resume sweep              # after the array finishes
```

To write your own, see
[Writing an experiment](../docs/writing_an_experiment.rst) (build the docs with
`pixi run docs`).
