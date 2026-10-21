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

A study shows its progress on the console and saves it as `study.log` in its folder;
a study that doesn't finish leaves `<study id>.log` beside the study folders. A
`--name` that is already taken is refused before anything runs.

To write your own, see
[Writing an experiment](../docs/writing_an_experiment.rst) (build the docs with
`pixi run docs`).
