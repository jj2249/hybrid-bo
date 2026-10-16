# Experiments

Each file here is an experiment: a problem, how it is measured, and the methods to
compare. `cli.py` is the shared command line, not an experiment.

- `illustrative_example.py`: the illustrative example of the
  [paper](https://arxiv.org/abs/2603.11199), with an oracle.

```bash
pixi run experiment <name> [--n-runs 3 --plot ...]   # run a study; --help lists the flags
pixi run plot <name> [STUDY]                         # plot the newest (or named) study
```

To write your own, see
[Writing an experiment](../docs/writing_an_experiment.rst) (build the docs with
`pixi run docs`).
