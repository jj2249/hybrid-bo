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
- `langermann_partial.py`: the same Langermann function, box and optimum with known
  and unknown equations: two of its five squared distances are known equations and
  three are unknown states, with an oracle, comparing the same methods. Standard BO
  and Latin hypercube sampling behave exactly as on `langermann.py`.
- `langermann_sweep.py`: a sweep over initial fractions on `langermann_partial.py`'s
  problem for SoftSAA-EI, n_initial = 2, 4, ..., 38, each run to 40 evaluations. It
  plots budget heatmaps of the median regret and of the fraction of runs that reach
  the global optimum's basin, the regret against the initial fraction at budgets of
  10, 20, 30 and 40, and the regret over evaluations by number of initial points
  instead of the
  convergence and regret figures. The full study is about 18 hours on one core, so
  run it with `--n-workers`.
- `cstr_series_reaction.py`: a 1D chemical engineering example for students. A CSTR
  runs A → B → C at a reactor temperature to choose, for the most B at the outlet.
  The mole balances and the kinetics of B → C are known; ln k_1 of the catalytic step
  A → B is the unknown state, with an oracle, comparing SoftSAA-EI, standard BO and
  Latin hypercube sampling.

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

On a cluster, `prepare` writes a study's task list without running anything and
prints its number of tasks, each task runs as `run-task STUDY --task-index K` (e.g.
one SLURM array element each, on one core with `OMP_NUM_THREADS=1`), and `resume`
assembles the study. `docs/running_large_studies.rst` has a full `sbatch` script.

To write your own, see
[Writing an experiment](../docs/writing_an_experiment.rst) (build the docs with
`pixi run docs`).
